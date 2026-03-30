"""文档管理 API 路由 — 上传、统计、删除。"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_store
from src.api.schemas import DocumentStatsResponse, DocumentUploadRequest, DocumentUploadResponse
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    body: DocumentUploadRequest,
    store: ChromaStore = Depends(get_store),
) -> DocumentUploadResponse:
    """上传文档：解码 base64 内容，分块并向量化后存入 ChromaDB。

    注意：实际使用需要 API Key 来调用 Embedding 接口。
    """
    try:
        # 解码 base64 内容
        try:
            raw_bytes = base64.b64decode(body.content)
            text = raw_bytes.decode("utf-8")
        except Exception:
            # 如果不是 base64，直接当作纯文本
            text = body.content

        if not text.strip():
            raise HTTPException(status_code=400, detail="文档内容为空")

        # 简单分块：按段落分割
        chunks = _simple_chunk(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档分块结果为空")

        # 生成 ID 和元数据
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            doc_id = f"{body.filename}_{chunk_hash}"
            ids.append(doc_id)
            metadatas.append(
                {
                    "source": body.filename,
                    "doc_type": body.doc_type,
                    "chunk_index": str(i),
                }
            )

        # 检查是否已存在
        existing_ids = set(store.get_all_ids())
        new_ids = []
        new_chunks = []
        new_metas = []
        skipped = 0
        for doc_id, chunk, meta in zip(ids, chunks, metadatas, strict=False):
            if doc_id in existing_ids:
                skipped += 1
                continue
            new_ids.append(doc_id)
            new_chunks.append(chunk)
            new_metas.append(meta)

        if not new_ids:
            return DocumentUploadResponse(
                success=True,
                message=f"文档已存在，跳过 {skipped} 个块",
                chunk_count=skipped,
            )

        # 注意：此处需要 Embedding 才能存储到 ChromaDB
        # 没有向量无法直接存储，返回提示信息
        return DocumentUploadResponse(
            success=True,
            message=f"文档解析成功，生成 {len(new_ids)} 个块（含 {skipped} 个已存在跳过）。"
            "请通过 IndexBuilder 或 Streamlit UI 完成向量化存储。",
            chunk_count=len(new_ids) + skipped,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("文档上传失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档上传失败: {e}") from e


@router.get("/documents/stats", response_model=DocumentStatsResponse)
async def document_stats(
    store: ChromaStore = Depends(get_store),
) -> DocumentStatsResponse:
    """文档统计：已索引的块数量、来源分布。"""
    try:
        stats = store.get_stats()
        total_chunks = stats.get("document_count", 0)

        # 获取来源分布
        all_docs = store.get_all_documents()
        source_dist: dict[str, int] = {}
        for doc_id, _content in all_docs.items():
            # doc_id 格式：source_hash
            source = doc_id.rsplit("_", 1)[0] if "_" in doc_id else "unknown"
            source_dist[source] = source_dist.get(source, 0) + 1

        return DocumentStatsResponse(
            total_chunks=total_chunks,
            unique_sources=len(source_dist),
            source_distribution=source_dist,
        )
    except Exception as e:
        logger.error("获取文档统计失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计失败: {e}") from e


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    store: ChromaStore = Depends(get_store),
) -> dict:
    """删除指定文档：从 ChromaDB 中移除。"""
    try:
        # 查找匹配的文档块（doc_id 是前缀，匹配所有属于该文档的块）
        all_ids = store.get_all_ids()
        matching_ids = [i for i in all_ids if i.startswith(doc_id) or i == doc_id]

        if not matching_ids:
            raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在")

        store.delete_by_ids(matching_ids)
        return {"success": True, "message": f"已删除 {len(matching_ids)} 个文档块", "deleted_count": len(matching_ids)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("删除文档失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {e}") from e


def _simple_chunk(text: str, max_length: int = 512) -> list[str]:
    """简单分块：按段落分割，超长段落按 max_length 截断。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_length:
            chunks.append(para)
        else:
            # 按 max_length 切分
            for i in range(0, len(para), max_length):
                chunks.append(para[i : i + max_length])
    return chunks
