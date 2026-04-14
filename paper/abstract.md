# 基于混合检索的金融领域RAG知识库问答系统设计与实现

## 摘要

大语言模型在金融问答场景中面临"幻觉"问题，可能生成看似合理但实际错误的信息。检索增强生成(RAG)通过引入外部知识检索缓解该问题，但单一检索方式难以同时满足精确匹配和语义理解的需求。本文设计并实现了一个基于混合检索的金融领域RAG知识库问答系统，采用BM25关键词检索与向量语义检索双路并行策略，通过RRF(Reciprocal Rank Fusion)算法融合两路检索结果排名，并结合Cross-Encoder重排序、Query Rewriting多轮对话改写和语义级查询缓存等优化技术。系统支持PDF研报、文本新闻等多种文档格式，提供Streamlit Web界面和FastAPI REST API双入口。在50条金融问答评估数据集上的实验表明，混合检索在忠实度上达到0.6224，相比纯向量检索提升9.9%，在上下文召回率上达到0.5455，提升23.1%，验证了混合检索策略在金融领域的有效性。

**关键词**：RAG；混合检索；RRF融合；金融问答；大语言模型

## Abstract

Large Language Models (LLMs) face the challenge of "hallucination" in financial question-answering scenarios, potentially generating plausible but incorrect information. Retrieval-Augmented Generation (RAG) addresses this by incorporating external knowledge retrieval, yet single retrieval methods struggle to simultaneously satisfy exact keyword matching and semantic understanding. This paper designs and implements a financial domain RAG knowledge base Q&A system based on hybrid retrieval. The system employs dual-parallel retrieval with BM25 keyword search and vector semantic search, fusing results through Reciprocal Rank Fusion (RRF) algorithm, combined with Cross-Encoder reranking, Query Rewriting for multi-turn dialogue, and semantic query caching. The system supports multiple document formats including PDF research reports and text news, providing both Streamlit Web UI and FastAPI REST API interfaces. Experimental results on a 50-sample financial QA dataset demonstrate that hybrid retrieval achieves a Faithfulness score of 0.6224 (9.9% improvement over pure vector retrieval) and a Context Recall of 0.5455 (23.1% improvement), validating the effectiveness of hybrid retrieval in the financial domain.

**Keywords**: RAG; Hybrid Retrieval; RRF Fusion; Financial QA; Large Language Model
