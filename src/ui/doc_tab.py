from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import Config
from src.loaders.pdf_loader import PDFLoader
from src.loaders.qa_loader import QALoader
from src.loaders.text_loader import TextLoader
from src.processor.chunker import TextChunker, TitleBasedChunker
from src.processor.cleaner import TextCleaner
from src.ui.services import _init_vectorstore, _init_embedder, _init_bm25_retriever
from src.ui.constants import _PROJECT_ROOT, _get_file_extension, _get_doc_type, _format_file_size

logger = logging.getLogger(__name__)
config = Config()


def _index_file(file_path: Path, original_name: str = "") -> tuple:
    ext = _get_file_extension(file_path.name)

    try:
        if ext == ".pdf":
            docs = PDFLoader().load(file_path)
        elif ext in (".txt", ".md"):
            docs = TextLoader().load(file_path)
        elif ext in (".json", ".csv"):
            docs = QALoader().load(file_path)
        else:
            return 0, f"不支持的文件格式: {ext}"
    except Exception as e:
        return 0, f"文件加载失败: {e}"

    if not docs:
        return 0, "文件内容为空"

    if config.chunker.strategy == "title":
        chunker = TitleBasedChunker(
            chunk_size=config.chunker.title_chunk_size,
            title_patterns=config.chunker.title_patterns,
        )
    else:
        chunker = TextChunker(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
    cleaner = TextCleaner()
    chunks = []
    for doc in docs:
        cleaned = cleaner.clean(doc.content)
        doc_chunks = chunker.chunk(cleaned, doc.metadata)
        chunks.extend(doc_chunks)

    if not chunks:
        return 0, "分块结果为空"

    try:
        api_key = st.session_state.api_key
        embedder = _init_embedder(api_key)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
    except Exception as e:
        return 0, f"向量化失败: {e}"

    try:
        store = _init_vectorstore()
        stem = original_name or file_path.stem
        ids = [f"{ext[1:]}_{Path(stem).stem}_{i}" for i in range(len(chunks))]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        store.add_documents(ids, documents, embeddings, metadatas)
    except Exception as e:
        return 0, f"向量存储失败: {e}"

    return len(chunks), ""


def render_doc_management_tab() -> None:
    st.markdown(
        '<div class="main-header">'
        '<h1>Document Management</h1>'
        '<div class="subtitle">Upload, index, and manage financial documents</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.4rem 0 1rem 0;">', unsafe_allow_html=True)

    st.subheader("UPLOAD")
    uploaded_files = st.file_uploader(
        "拖拽文件到此处，或点击上传",
        type=["pdf", "txt", "md", "json", "csv"],
        accept_multiple_files=True,
        help="支持 PDF、TXT、Markdown、JSON(Q&A)、CSV(Q&A) 格式",
    )

    api_key = st.session_state.api_key
    if not api_key:
        st.warning("Please configure API Key in sidebar before uploading")
    elif uploaded_files:
        if st.button("START INDEXING", type="primary", use_container_width=True):
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Preparing...")
                status_text = st.empty()

            total = len(uploaded_files)
            success_count = 0
            error_list = []

            for idx, uploaded_file in enumerate(uploaded_files):
                ext = _get_file_extension(uploaded_file.name)
                progress_bar.progress(
                    idx / total,
                    text=f"Processing ({idx + 1}/{total}): {uploaded_file.name}...",
                )

                raw_dir = _PROJECT_ROOT / "data" / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    suffix=ext, delete=False, dir=str(raw_dir)
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)

                try:
                    chunk_count, error_msg = _index_file(tmp_path, original_name=uploaded_file.name)
                    if error_msg:
                        error_list.append(f"{uploaded_file.name}: {error_msg}")
                        status_text.warning(f"[WARN] {uploaded_file.name}: {error_msg}")
                    else:
                        success_count += 1
                        status_text.success(
                            f"[OK] {uploaded_file.name}: indexed {chunk_count} chunks"
                        )
                        st.session_state.indexed_files.append({
                            "filename": uploaded_file.name,
                            "type": _get_doc_type(ext),
                            "size": _format_file_size(uploaded_file.size),
                            "chunks": chunk_count,
                            "index_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                except Exception as e:
                    logger.error("索引文件失败 %s: %s", uploaded_file.name, e, exc_info=True)
                    error_list.append(f"{uploaded_file.name}: 处理异常")
                    status_text.error(f"[ERR] {uploaded_file.name}: {e}")
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            progress_bar.progress(1.0, text="Complete")
            time.sleep(1)
            progress_bar.empty()

            if success_count > 0:
                _init_bm25_retriever.clear()

            st.divider()
            if success_count > 0:
                st.success(f"Indexed {success_count}/{total} files successfully")
            if error_list:
                st.error("Failed files:")
                for err in error_list:
                    st.error(f"  · {err}")

    st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:1rem 0;">', unsafe_allow_html=True)
    st.subheader("INDEXED DOCUMENTS")

    store = None
    try:
        store = _init_vectorstore()
        all_ids = store.get_all_ids()
        stats = store.get_stats()
        st.caption(f"Total indexed chunks: **{stats['document_count']}**")
    except Exception as e:
        logger.error("Vector store connection failed: %s", e, exc_info=True)
        st.warning("Unable to connect to vector store")
        all_ids = []

    if all_ids:
        file_groups = {}
        for doc_id in all_ids:
            parts = doc_id.split("_", 2)
            if len(parts) >= 2:
                file_key = f"{parts[0]}_{parts[1]}"
                file_groups.setdefault(file_key, []).append(doc_id)
            else:
                file_groups.setdefault(doc_id, []).append(doc_id)

        rows = []
        for file_key, chunk_ids in file_groups.items():
            ext = f".{file_key.split('_')[0]}" if "_" in file_key else ""
            rows.append({
                "文件名": file_key,
                "类型": _get_doc_type(ext),
                "文档块数": len(chunk_ids),
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:1rem 0;">', unsafe_allow_html=True)
        st.subheader("DELETE DOCUMENT")
        selected_file = st.selectbox(
            "选择要删除的文件",
            options=list(file_groups.keys()),
            format_func=lambda x: x,
        )

        if st.button("删除选中文件的所有文档块", type="secondary"):
            if selected_file and selected_file in file_groups and store:
                ids_to_delete = file_groups[selected_file]
                try:
                    store.delete_by_ids(ids_to_delete)
                    _init_bm25_retriever.clear()
                    st.success(f"Deleted {len(ids_to_delete)} chunks for {selected_file}")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")

        st.markdown('<hr style="border-color:rgba(201,168,76,0.15);margin:1rem 0;">', unsafe_allow_html=True)
        st.subheader("RE-INDEX")
        if st.button("CLEAR ALL AND RE-INDEX", type="secondary"):
            if st.checkbox("Confirm (irreversible)"):
                try:
                    if store is None:
                        st.error("无法连接向量数据库")
                        return
                    store.delete_collection()
                    _init_bm25_retriever.clear()
                    st.session_state.indexed_files = []
                    st.success("All documents cleared. Please re-upload.")
                    st.rerun()
                except Exception as e:
                    st.error(f"清空失败: {e}")
    else:
        st.info("No indexed documents yet. Upload files and click START INDEXING.")
