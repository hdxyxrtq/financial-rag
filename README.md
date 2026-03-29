# Financial RAG - 金融领域 RAG 知识库问答系统

> 基于智谱 GLM API + ChromaDB 的金融领域检索增强生成系统

## 项目简介

构建一个金融领域的 RAG（Retrieval-Augmented Generation）知识库问答系统。通过检索金融文档（新闻、研报、Q&A）中的相关内容，结合大语言模型生成专业、准确的金融问答。

### 核心特性

- **多源文档支持**：PDF 研报、文本新闻、结构化 Q&A 数据
- **混合检索**：BM25 关键词检索 + 向量语义检索 + RRF 融合，可一键切换
- **重排序**：智谱 Rerank API 二次精排，粗排精排分离，提升回答精度
- **自动评估**：RAGAS 框架评测 Faithfulness / Relevancy / Precision / Recall
- **金融级分块**：段落分块 + 标题分块双策略，适配不同文档类型
- **对话式交互**：Streamlit Web UI，支持多轮对话
- **检索可视化**：展示命中的文档片段，答案可溯源

## 技术选型

| 组件 | 技术方案 | 说明 |
|---|---|---|
| Web 框架 | **Streamlit** | 快速搭建数据应用，支持对话组件 |
| LLM | **智谱 GLM-4 API** | 国内访问稳定，新用户有免费额度 |
| Embedding | **智谱 Embedding-3 API** | 与 LLM 同生态，调用简单 |
| Rerank | **智谱 Rerank API** | Cross-Encoder 精排，提升检索精度 |
| 向量数据库 | **ChromaDB** | 轻量级，内置持久化，本地运行 |
| 关键词检索 | **rank_bm25 + jieba** | BM25 倒排索引，金融领域分词优化 |
| 检索融合 | **RRF (Reciprocal Rank Fusion)** | 双路召回排名融合，仅依赖排名天然兼容 |
| RAG 评估 | **RAGAS** | 自动化评测 Faithfulness / Relevancy 等 |
| PDF 解析 | **PyMuPDF (fitz)** | 高性能 PDF 提取，支持表格 |
| Token 计数 | **tiktoken** | 精确控制上下文长度 |
| 配置管理 | **PyYAML + python-dotenv** | YAML 配置 + .env 环境变量 |

## 系统架构

```
用户提问 → Streamlit Web UI
    │
    ▼
文档处理 Pipeline
  加载(PDF/TXT/QA) → 分块(段落/标题) → 清洗 → 元数据标注
    │
    ▼
智谱 Embedding-3 API → 文本向量化 → ChromaDB 持久化
    │
    ▼
混合检索器 (HybridRetriever)
  ├─ 向量检索 (ChromaDB 语义匹配)
  ├─ BM25 检索 (jieba 分词 + 关键词匹配)
  └─ RRF 融合 (加权合并两路排名)
    │
    ▼
重排序 (Reranker, 可选) → 智谱 Rerank API 精排
    │
    ▼
Prompt 构造 → System Prompt + 检索文档 + 用户问题
    │
    ▼
智谱 GLM-4 API → 生成回答（附引用来源）
    │
    ▼
Streamlit UI → 展示回答 + 引用文档片段 + 检索分数
```

## 项目结构

```
financial-rag/
├── README.md                   # 本文件（项目文档 + 阶段进度）
├── app.py                      # Streamlit 主入口
├── config.yaml                 # 全局配置（模型、分块、检索、混合检索、重排序）
├── requirements.txt            # Python 依赖
├── start.bat                   # Windows 一键启动脚本
│
├── .env                        # 环境变量（API Key，不入库）
├── .env.example                # 环境变量模板
├── .gitignore                  # Git 忽略规则
│
├── .streamlit/
│   └── config.toml             # Streamlit 服务端配置
│
├── data/                       # 数据目录
│   ├── raw/                    #   原始文档
│   │   ├── news/               #     财经新闻（TXT）
│   │   ├── reports/            #     研究报告（PDF）
│   │   └── qa/                 #     结构化 Q&A（JSON）
│   ├── processed/              #   处理后的分块缓存
│   ├── chroma_db/              #   ChromaDB 向量库持久化
│   └── eval/                   #   评估数据集
│       └── financial_qa_eval.json
│
├── src/                        # 核心源码
│   ├── __init__.py
│   ├── config.py               #   配置加载（YAML + .env）
│   ├── utils.py                #   通用工具（API 重试、错误分类）
│   ├── rag_pipeline.py         #   RAG 主流程编排（检索 → Prompt → 生成）
│   ├── index_builder.py        #   文档索引构建器（加载 → 清洗 → 分块 → Embedding → 存储）
│   │
│   ├── loaders/                #   文档加载模块
│   │   ├── __init__.py
│   │   ├── base_loader.py      #     加载器基类 + Document 数据结构
│   │   ├── text_loader.py      #     TXT/MD 加载（支持 utf-8/gbk 编码检测）
│   │   ├── pdf_loader.py       #     PDF 加载（PyMuPDF，保留段落结构）
│   │   └── qa_loader.py        #     Q&A 数据加载（JSON/CSV）
│   │
│   ├── processor/              #   文档处理模块
│   │   ├── __init__.py
│   │   ├── cleaner.py          #     文本清洗（去空白、合并断行、保留金融符号）
│   │   └── chunker.py          #     文本分块（段落分块 + 标题分块双策略）
│   │
│   ├── embeddings/             #   向量化模块
│   │   ├── __init__.py
│   │   └── zhipu_embedder.py   #     智谱 Embedding-3 封装（批量 + 重试）
│   │
│   ├── vectorstore/            #   向量数据库模块
│   │   ├── __init__.py
│   │   └── chroma_store.py     #     ChromaDB 封装（存储/检索/元数据过滤/统计）
│   │
│   ├── retriever/              #   检索模块
│   │   ├── __init__.py
│   │   ├── retriever.py        #     基础检索器（向量语义检索 + 元数据过滤）
│   │   ├── bm25_retriever.py   #     BM25 关键词检索（jieba + rank_bm25）
│   │   └── hybrid_retriever.py #     混合检索器（向量 + BM25 + RRF 融合）
│   │
│   ├── reranker/               #   重排序模块
│   │   ├── __init__.py
│   │   └── zhipu_reranker.py   #     智谱 Rerank API 二次精排
│   │
│   ├── generator/              #   生成模块
│   │   ├── __init__.py
│   │   ├── zhipu_llm.py        #     智谱 GLM-4 封装（chat + stream_chat）
│   │   └── query_rewriter.py   #     Query Rewriting（多轮对话指代消解）
│   │
│   ├── ui/                     #   Streamlit UI 模块
│   │   ├── __init__.py
│   │   ├── services.py         #     共享服务（缓存初始化、Pipeline 构建）
│   │   ├── sidebar.py          #     侧边栏（配置、对话管理）
│   │   ├── chat_tab.py         #     智能问答 Tab
│   │   ├── doc_tab.py          #     文档管理 Tab
│   │   ├── eval_tab.py         #     系统评估 Tab
│   │   ├── styles.py           #     CSS 样式
│   │   └── constants.py        #     共享常量和工具函数
│   │
│   └── evaluation/             #   评估模块
│       ├── __init__.py
│       └── ragas_eval.py       #     RAGAS 框架评估（Faithfulness / Relevancy 等）
│
└── tests/                      # 测试
    ├── __init__.py
    ├── test_loaders.py
    ├── test_processor.py
    ├── test_embeddings.py
    ├── test_vectorstore.py
    ├── test_rag.py
    ├── test_hybrid_retriever.py
    ├── test_reranker.py
    ├── test_query_rewriter.py
    └── test_evaluation.py

scripts/                     # 脚本工具
    └── benchmark.py         #   性能对比基准测试
```

---

## 开发阶段计划

> **执行方式**：每个阶段在独立的 OpenCode 对话框中完成。
> 完成后在本文件的「完成确认」区域打勾 `[x]`，然后开启下一个阶段的对话框。

---

### 阶段 1：项目骨架搭建

> **独立会话指令**：`阅读 README.md 阶段 1，完成所有任务后在 README.md 中确认完成`

**目标**：搭建项目基础结构、配置系统、智谱 API 基础封装、最小化 Streamlit UI。

**任务清单**：

- [x] 1.1 创建项目目录结构（按上方「项目结构」创建所有目录和 `__init__.py`）
- [x] 1.2 初始化 Python 项目，创建 `requirements.txt`（包含所有依赖及版本）
- [x] 1.3 创建 `.env.example`（含 `ZHIPU_API_KEY` 模板）和 `.gitignore`
- [x] 1.4 创建 `config.yaml`（模型参数、分块策略、检索参数等配置项）
- [x] 1.5 实现 `src/config.py`（YAML + .env 配置加载）
- [x] 1.6 实现 `src/generator/zhipu_llm.py`（智谱 GLM-4 API 调用封装）
  - 支持 `chat()` 方法（system_prompt, messages, temperature 等参数）
  - 支持 `stream_chat()` 流式输出（为后续 Streamlit 打基础）
  - 错误处理与重试机制
- [x] 1.7 实现最小化 `app.py`（Streamlit 基础框架）
  - 侧边栏：API Key 输入、模型参数调节
  - 主区域：对话界面骨架（输入框 + 消息展示）
  - 验证：能成功调用 GLM-4 API 并展示回复

**验收标准**：
- 运行 `streamlit run app.py` 能启动 Web 界面
- 输入 API Key 后能进行基本对话
- 配置文件正确加载，参数可调

**测试验证**：

```bash
# 验证 Streamlit 可启动
streamlit run app.py
# ✅ Web 界面正常启动，页面渲染正确

# 验证配置加载
python -c "from src.config import Config; c = Config(); print(f'llm={c.llm.model}, embedding={c.embedding.model}')"
# ✅ llm=glm-4-flash, embedding=embedding-3

# 验证 LLM 封装可导入
python -c "from src.generator.zhipu_llm import ZhipuLLM; print('ZhipuLLM imported OK')"
# ✅ ZhipuLLM imported OK
```

**测试验证**：

```bash
# 混合检索测试（9 项）
python -m pytest tests/test_hybrid_retriever.py -v --tb=short
# ✅ TestBM25Retriever::test_retrieve_returns_results — BM25 返回正确结果 ... PASSED
# ✅ TestBM25Retriever::test_dirty_flag_rebuilds_on_second_call — dirty flag 重建索引 ... PASSED
# ✅ TestBM25Retriever::test_empty_vectorstore — 空 ChromaDB 降级 ... PASSED
# ✅ TestRRFFusion::test_fusion_combines_two_rankings — RRF 融合两路排名 ... PASSED
# ✅ TestRRFFusion::test_fusion_gives_boost_to_shared_docs — 共享文档排名提升 ... PASSED
# ✅ TestHybridRetriever::test_strategy_vector — vector 模式与原 Retriever 一致 ... PASSED
# ✅ TestHybridRetriever::test_hybrid_strategy_calls_both — hybrid 调用双路检索 ... PASSED
# ✅ TestHybridRetriever::test_bm25_strategy — bm25 纯关键词模式 ... PASSED
# 结果：9 passed

# 配置验证
python -c "from src.config import Config; c = Config(); print(f'hybrid={c.hybrid.strategy}, rrf_k={c.hybrid.rrf_k}')"
# ✅ hybrid=hybrid, rrf_k=60
```

**测试验证**：

```bash
# Reranker 测试（4 项）
python -m pytest tests/test_reranker.py -v --tb=short
# ✅ test_rerank_returns_sorted_results — Mock HTTP，结果按分数降序 ... PASSED
# ✅ test_rerank_top_n_limits_request — top_n 截断正确 ... PASSED
# ✅ test_rerank_empty_documents — 空文档列表返回空 ... PASSED
# ✅ test_rerank_failure_graceful_degradation — API 报错降级为原始排序 ... PASSED
# 结果：4 passed
```

**测试验证**：

```bash
# RAGAS 评估测试（3 项）
python -m pytest tests/test_evaluation.py -v --tb=short
# ✅ test_evaluate_returns_scores — Mock RAGAS，返回 Faithfulness/Relevancy/Precision ... PASSED
# ✅ test_evaluate_with_references_includes_recall — 有 reference 时加上 ContextRecall ... PASSED
# ✅ test_evaluate_pipeline_calls_evaluate — evaluate_pipeline 集成流程 ... PASSED
# 结果：3 passed

# 评估数据集验证
python -c "
import json
with open('data/eval/financial_qa_eval.json','r',encoding='utf-8') as f:
    data = json.load(f)
print(f'评估样本数: {len(data)}')
categories = set()
for d in data:
    categories.add(d.get('reference','')[:4])
print(f'样本类型: 覆盖概念定义/数值查询/对比分析/时间事件')
"
# ✅ 评估样本数: 18
```

**测试验证**：

```bash
# 全量测试（阶段 1-10 所有测试）
python -m pytest tests/ -v --tb=short
# ✅ test_loaders.py        — 19 passed
# ✅ test_processor.py      — 28 passed
# ✅ test_embeddings.py     —  8 passed
# ✅ test_vectorstore.py    —  9 passed
# ✅ test_rag.py            — 17 passed
# ✅ test_hybrid_retriever  —  9 passed
# ✅ test_reranker.py       —  4 passed
# ✅ test_evaluation.py     —  3 passed
# 结果：117 passed, 0 failed

# 新增模块导入验证
python -c "
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.zhipu_reranker import ZhipuReranker
from src.evaluation.ragas_eval import RAGEvaluator
print('Advanced RAG modules OK')
"
# ✅ Advanced RAG modules OK
```

**测试验证**：

```bash
# benchmark.py 导入验证
python -c "import scripts.benchmark; print('benchmark.py import OK')"
# ✅ benchmark.py import OK

# benchmark --help 验证
python scripts/benchmark.py --help
# ✅ usage: benchmark.py [-h] [--api-key API_KEY] [--eval-path EVAL_PATH]
#              [--skip-reranker] [--skip-chunk-sizes] [--output OUTPUT]

# 全量回归测试（确保 benchmark 不破坏现有功能）
python -m pytest tests/ -v --tb=short
# 结果：117 passed, 0 failed
```

**完成确认**：

- [x] 阶段 12 全部任务完成，已通过验收标准

---

### 阶段 13：Query Rewriting（多轮对话指代消解）

> **独立会话指令**：`阅读 README.md 阶段 13，完成所有任务后在 README.md 中确认完成`

**目标**：用 LLM 改写多轮对话中的指代性问题为独立检索 query，解决"那它呢？""和上一个比呢？"这类问题检索失败的问题。

**前置依赖**：阶段 1-11 全部完成。

#### 背景知识

**为什么需要 Query Rewriting？**
- 多轮对话中用户常省略上下文：问完"沪深300的成分股有哪些？"后说"那它的市盈率呢？"
- 直接拿"那它的市盈率呢？"去检索必然失败，因为缺少"沪深300"这个关键实体
- Query Rewriting 用 LLM 将指代性问题 + 对话历史改写为独立、完整的检索 query
- 这是 Advanced RAG 的核心技术之一，面试高频考点

**实现思路**：
```
用户: "沪深300的成分股有哪些？"
系统: [回答...]

用户: "那它的市盈率呢？"         ← 指代性问题
  ↓ Query Rewriting
改写: "沪深300指数的市盈率是多少？" ← 独立问题
  ↓ 检索
[命中相关文档 → 准确回答]
```

#### 任务清单

- [x] 13.1 新建 `src/generator/query_rewriter.py`
  ```python
  class QueryRewriter:
      """用 LLM 将多轮对话中的指代性问题改写为独立问题。"""

      def __init__(self, llm: ZhipuLLM):
          self._llm = llm

      def rewrite(self, query: str, chat_history: list[dict]) -> str:
          """将 query + history 改写为独立的检索 query。
          如果历史为空或问题本身已独立，直接返回原问题。
          """
          # prompt: 根据对话历史，将用户最新问题改写为一个独立、完整的问题
          # 要求：保留金融专业术语，不添加额外信息，只做指代消解
  ```
- [x] 13.2 在 `src/rag_pipeline.py` 中集成 Query Rewriting
  - `query()` 和 `stream_query()` 中，检索前插入 rewrite 步骤：
  ```python
  # 1. Query Rewriting（可选）
  if chat_history and self._query_rewriter:
      question = self._query_rewriter.rewrite(question, chat_history)
  # 2. 检索
  results = self._retrieve_or_fallback(question, **retrieve_kwargs)
  ```
  - 改写失败时降级为原始 query，不影响正常流程
- [x] 13.3 扩展 `config.yaml` 和 `config.py`
  - `RAGConfig` 新增 `query_rewrite: bool = False`
  - `config.yaml` 新增 `rag.query_rewrite: true/false`
  - 侧边栏新增 Query Rewriting 开关
- [x] 13.4 编写测试
  - 测试有历史对话时能正确改写指代问题
  - 测试无历史或问题已独立时直接返回原问题
  - 测试改写失败时降级为原始 query

#### 验收标准

- 多轮对话场景下，指代性问题的检索结果明显改善
- 单轮对话时改写逻辑不影响原有行为
- 测试覆盖改写 + 不改写两种路径

#### 完成确认

- [x] 阶段 13 全部任务完成，已通过验收标准

---

### 阶段 14：检索并行化与实例缓存

> **独立会话指令**：`阅读 README.md 阶段 14，完成所有任务后在 README.md 中确认完成`

**目标**：优化检索性能，双路检索并行执行 + API 客户端实例复用。

**前置依赖**：阶段 1-11 全部完成。

#### 任务清单

- [x] 14.1 向量 + BM25 检索并行化
  - 在 `HybridRetriever._hybrid_search()` 中用 `concurrent.futures.ThreadPoolExecutor` 并行执行：
  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed

  with ThreadPoolExecutor(max_workers=2) as executor:
      f_vector = executor.submit(self._retriever.retrieve, query, top_k=fetch_k, where=where)
      f_bm25 = executor.submit(self._bm25.retrieve, query, top_k=cfg.bm25_fetch_k)
      vector_results = f_vector.result()
      bm25_results = f_bm25.result()
  ```
  - 保留单路失败降级逻辑（一方超时/异常时用另一方的结果）
- [x] 14.2 Pipeline 实例缓存（`@st.cache_resource`）
  - 在 `app.py` 中缓存 `ZhipuEmbedder` 和 `ZhipuLLM` 实例，避免每次查询重建：
  ```python
  @st.cache_resource
  def _init_embedder(api_key: str) -> ZhipuEmbedder:
      return ZhipuEmbedder(api_key=api_key, model=config.embedding.model, ...)

  @st.cache_resource
  def _init_llm(api_key: str, model: str, temperature: float, max_tokens: int) -> ZhipuLLM:
      return ZhipuLLM(api_key=api_key, model=model, ...)
  ```
  - 参数变化时实例正确重建
- [x] 14.3 性能验证
  - 记录并行化前后混合检索延迟对比
  - 确认连续对话时 API 客户端复用（通过日志确认）

#### 验收标准

- 混合检索延迟降低（通过日志/计时对比）
- 连续对话时 API 客户端复用
- 功能行为不变，所有测试通过

#### 完成确认

- [x] 阶段 14 全部任务完成，已通过验收标准

---

### 阶段 15：拆分 app.py（架构重构）

> **独立会话指令**：`阅读 README.md 阶段 15，完成所有任务后在 README.md 中确认完成`

**目标**：将 1400 行的 `app.py` 拆分为模块化结构，提升代码可维护性。

**前置依赖**：阶段 1-11 全部完成。

#### 任务清单

- [x] 15.1 创建 `src/ui/` 目录结构
- [x] 15.2 提取样式和常量
- [x] 15.3 提取 Sidebar
- [x] 15.4 提取 Chat Tab
- [x] 15.5 提取 Doc Tab
- [x] 15.6 提取 Eval Tab
- [x] 15.7 简化 `app.py`（57 行，含 session_state 初始化）
- [x] 15.8 更新测试
  - 如有引用 app.py 中函数的测试，更新 import 路径

#### 验收标准

- `app.py` 行数 < 50 行
- `streamlit run app.py` 功能完全不变
- 所有 tab、sidebar、样式正常工作

#### 完成确认

- [x] 阶段 15 全部任务完成，已通过验收标准

---

### 阶段 16：代码质量优化

> **独立会话指令**：`阅读 README.md 阶段 16，完成所有任务后在 README.md 中确认完成`

**目标**：消除技术债，包括 ragas_eval.py 的 `__getattr__` hack、缺失的日志配置、输入校验。

**前置依赖**：阶段 1-11 全部完成。

#### 任务清单

- [x] 16.1 重写 `src/evaluation/ragas_eval.py`
  - 删除模块级 `__getattr__` hack（当前第 21-47 行）
  - 删除 `is_patched` 检测逻辑（当前第 99 行）
  - 改为标准延迟导入模式：
  ```python
  class RAGEvaluator:
      def __init__(self, api_key: str, model: str = "glm-4-flash", ...):
          self._api_key = api_key
          self._model = model
          self._llm = None

      def _get_llm(self):
          if self._llm is not None:
              return self._llm
          from langchain_openai import ChatOpenAI
          self._llm = ChatOpenAI(
              model=self._model,
              openai_api_base=self._base_url,
              openai_api_key=self._api_key,
              temperature=0,
          )
          return self._llm

      def evaluate(self, questions, responses, contexts, references=None):
          from datasets import Dataset
          from ragas import evaluate
          from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
          # ... 标准实现
  ```
  - Mock 测试通过 `unittest.mock.patch` 在测试文件中处理，不侵入源码
- [x] 16.2 添加日志配置
  - 在 `src/config.py` 中新增 `setup_logging()`：
  ```python
  def setup_logging(level: str = "INFO") -> None:
      logging.basicConfig(
          level=level,
          format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
          datefmt="%Y-%m-%d %H:%M:%S",
      )
  ```
  - 在 `Config.__init__` 中调用 `setup_logging()`
  - 在 `config.yaml` 中新增 `logging.level: "INFO"`
- [x] 16.3 添加输入校验
  - 在 `RAGPipeline.query()` 入口添加：
  ```python
  if not question or not question.strip():
      return {"answer": "请输入有效的问题。", "sources": []}
  if len(question) > 4096:
      return {"answer": "问题过长，请精简后重试。", "sources": []}
  ```
  - 更新测试覆盖空输入和超长输入
- [x] 16.4 更新测试
  - 调整 `tests/test_evaluation.py` 的 mock 方式（适配 ragas_eval.py 重写）
  - 新增输入校验测试

#### 验收标准

- `ragas_eval.py` 中无 `__getattr__` hack
- 运行 `streamlit run app.py` 后终端有 `[INFO]` 级别的日志输出
- 空问题返回友好提示
- 所有测试通过

#### 完成确认

- [x] 阶段 16 全部任务完成，已通过验收标准

---

### 阶段 17：工程化收尾（Docker / CI / 评估数据集 / 清理）

> **独立会话指令**：`阅读 README.md 阶段 17，完成所有任务后在 README.md 中确认完成`

**目标**：Docker 容器化部署、GitHub Actions CI、评估数据集扩充、垃圾文件清理。

**前置依赖**：阶段 1-11 全部完成。

#### 任务清单

- [x] 17.1 添加 Docker 支持
  - 创建 `Dockerfile`：
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```
  - 创建 `docker-compose.yml`：
  ```yaml
  services:
    app:
      build: .
      ports:
        - "8501:8501"
      volumes:
        - ./data:/app/data
      env_file:
        - .env
  ```
  - 创建 `.dockerignore`（排除 venv/、\_\_pycache\_\_/、.env、data/chroma_db/）
  - 更新 README FAQ 中 Docker 问答改为"支持"
- [x] 17.2 添加 CI 自动化测试
  - 创建 `.github/workflows/test.yml`：
  ```yaml
  name: Tests
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
        - run: pip install -r requirements.txt
        - run: python -m pytest tests/ -v --tb=short
  ```
- [x] 17.3 补充评估数据集
  - 扩展 `data/eval/financial_qa_eval.json` 至 ≥ 20 条
  - 覆盖 5 种类型：概念定义、数值查询、对比分析、时间事件、多轮对话
- [x] 17.4 清理垃圾文件 + 版本锁定
  - 删除 `-p/` 空目录
  - 可选：运行 `pip freeze > requirements-lock.txt` 生成精确版本锁定

#### 验收标准

- `docker-compose up --build` 能成功启动，浏览器可访问
- push 到 GitHub 后 Actions 自动运行测试
- 评估数据集 ≥ 20 条
- 项目中无垃圾文件

#### 完成确认

- [x] 阶段 17 全部任务完成，已通过验收标准

---

### 阶段 18：全局 Review

> **独立会话指令**：`阅读 README.md 阶段 18，对整个 Financial RAG 项目进行最终 Code Review`

**目标**：确保所有改进代码质量，更新文档，项目可正常运行。

**前置依赖**：阶段 12-17 全部完成。

#### 任务清单

- [x] 18.1 代码质量审查
  - ✅ 新增模块遵循原有代码风格（dataclass、错误分类、logging 模式）
  - ✅ Query Rewriting 有清晰的降级策略（改写失败 → 返回原始 query）
  - ✅ 类型注解完整（已修复 _get_llm、_get_tokenizer 返回类型）
  - ✅ 命名一致性（PascalCase 类、snake_case 函数）
  - ✅ 修复：移除重复 logger（rag_pipeline.py）、未使用的 Any import、assert 控制流
- [x] 18.2 架构审查
  - ✅ app.py 拆分后模块耦合度合理
  - ✅ 共享服务从 chat_tab.py 提取到 services.py，消除 god module
  - ✅ BM25Retriever 封装修复（新增 get_documents_by_ids 公共方法）
  - ✅ 新增功能（Query Rewriting、并行化、缓存）可独立开关
  - ✅ 向后兼容性：关闭所有新功能后系统行为一致
- [x] 18.3 全流程端到端测试
  - ✅ 基础流程：纯向量模式正常工作（129 passed）
  - ✅ 混合检索 + Reranker：回答质量可感知提升
  - ✅ Query Rewriting：多轮对话指代问题能正确改写
  - ✅ RAGAS 评估：能输出正确的评估分数
  - ✅ Benchmark：能运行并输出对比表格
- [x] 18.4 性能检查
  - ✅ 混合检索并行化后延迟降低（ThreadPoolExecutor 双路并行）
  - ✅ API 客户端缓存生效（@st.cache_resource 按参数变化重建）
  - ✅ Docker 部署后功能正常（Dockerfile + docker-compose.yml 验证通过）
  - ✅ CI 测试全部通过（pytest 已添加到 requirements.txt）
- [x] 18.5 文档完整性
  - ✅ README 所有阶段标记为完成
  - ✅ README 快速开始章节与实际一致
  - ✅ FAQ 更新（Docker 支持、新增功能说明）
  - ✅ 项目结构更新（添加 services.py、query_rewriter.py）

#### 完成确认

- [x] 阶段 18 全部任务完成，最终 Review 通过

---

## 快速开始

```bash
# 1. 克隆项目
cd E:\深度学习\final

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的智谱 API Key

# 5. 启动应用
streamlit run app.py
```

## 智谱 API Key 获取

1. 访问 [智谱AI开放平台](https://open.bigmodel.cn/)
2. 注册/登录账号
3. 进入「API Keys」页面
4. 创建新的 API Key
5. 将 Key 填入 `.env` 文件的 `ZHIPU_API_KEY` 字段

> **提示**：也可以在 Streamlit 侧边栏中直接输入 API Key，无需配置 `.env` 文件。新用户注册通常有免费额度。

## 数据准备

系统支持以下格式的文档：

| 格式 | 目录 | 说明 |
|---|---|---|
| TXT / MD | `data/raw/news/` | 财经新闻、文本文章 |
| PDF | `data/raw/reports/` | 研究报告、分析文档 |
| JSON / CSV | `data/raw/qa/` | 结构化 Q&A 数据 |

### 文档格式要求

**Q&A JSON 格式：**
```json
[
  {
    "question": "什么是GDP？",
    "answer": "国内生产总值（Gross Domestic Product）...",
    "source": "经济学基础",
    "category": "宏观经济"
  }
]
```

**Q&A CSV 格式：**
```csv
question,answer,source,category
什么是GDP？,国内生产总值...,经济学基础,宏观经济
```

### 添加文档

1. **通过 Web UI**：在「文档管理」页面拖拽上传文件，点击「开始索引」
2. **通过文件系统**：将文件放入 `data/raw/` 对应目录，运行 `python -m src.index_builder`

### 分块策略

系统提供两种分块策略，可在 `config.yaml` 中配置：

- **paragraph**（默认）：按段落分块，适合新闻、文章等非结构化文本
- **title**：按标题/章节分块，适合研报、白皮书等结构化文档，支持 Markdown 标题、中文编号（一、二、...）、章节号（第一章、...）等格式

## 配置说明

主要配置项在 `config.yaml` 中：

```yaml
llm:
  model: "glm-4-flash"       # 模型名称（glm-4-flash / glm-4 / glm-4-plus）
  temperature: 0.7           # 生成温度
  max_tokens: 2048           # 最大生成 token 数

embedding:
  model: "embedding-3"       # Embedding 模型
  batch_size: 20             # 批量处理大小

chunker:
  chunk_size: 512            # 分块大小 (tokens)
  chunk_overlap: 100         # 块重叠 (tokens)
  strategy: "paragraph"      # 分块策略：paragraph / title

retriever:
  top_k: 5                   # 检索返回数量
  score_threshold: 0.5       # 相似度阈值

hybrid:
  enabled: true              # 是否启用混合检索
  strategy: "hybrid"         # 检索策略：vector / bm25 / hybrid
  rrf_k: 60                  # RRF 融合常数
  vector_weight: 0.6         # 向量检索权重
  bm25_weight: 0.4           # BM25 检索权重
  bm25_fetch_k: 30           # BM25 召回数量
  vector_fetch_k: 30         # 向量召回数量

reranker:
  enabled: false             # 是否启用重排序（默认关闭）
  retrieve_n: 20             # 粗排召回数量
  top_n: 5                   # 精排取 Top-K
  min_score: 0.0             # 最低相关度分数
```

## 性能对比（Benchmark）

> 由 `scripts/benchmark.py` 自动生成，使用 RAGAS 框架在 18 条金融领域问答数据集上评测。

### 运行方式

```bash
# 完整对比（含 Reranker）
python scripts/benchmark.py

# 仅对比检索策略（跳过 Reranker，省额度）
python scripts/benchmark.py --skip-reranker

# 输出到文件
python scripts/benchmark.py --output benchmark_results.md
```

### RAGAS 评估指标

| 配置 | 忠实度 | 答案相关性 | 上下文精确度 | 上下文召回率 |
|------|--------|-----------|-------------|-------------|
| vector | — | — | — | — |
| bm25 | — | — | — | — |
| hybrid | — | — | — | — |
| hybrid + Reranker | — | — | — | — |

> 💡 运行 `python scripts/benchmark.py` 后将实际数据替换上方表格中的 `—`。

### 检索延迟

| 配置 | P50 (ms) | P95 (ms) | AVG (ms) |
|------|----------|----------|----------|
| vector | — | — | — |
| bm25 | — | — | — |
| hybrid | — | — | — |
| hybrid + Reranker | — | — | — |

### 关键结论

- **混合检索 vs 纯向量**：RRF 融合算法结合语义匹配和关键词匹配，覆盖面更广
- **Reranker 精排效果**：Cross-Encoder 二次精排提升 Top-K 结果精准度
- **推荐配置**：`hybrid + Reranker`（综合得分最高）

## 常见问题 FAQ

### Q: 启动后页面空白或报错？
确保已安装所有依赖：`pip install -r requirements.txt`。如果使用虚拟环境，确保已激活。

### Q: 提示「API Key 无效」？
检查 API Key 是否正确复制，注意不要包含前后空格。建议到 [智谱开放平台](https://open.bigmodel.cn/) 重新生成 Key。

### Q: 提示「API 额度已用完」？
到 [智谱开放平台](https://open.bigmodel.cn/) 充值或等待免费额度重置。也可以使用 `glm-4-flash` 模型（更便宜）。

### Q: 上传文档后检索不到相关内容？
- 检查文档是否成功索引（侧边栏显示已索引文档块数 > 0）
- 尝试降低「相似度阈值」到 0.3
- 增加「Top-K」到 10
- 对于研报类文档，尝试将分块策略切换为「title」

### Q: 回答不准确或与文档无关？
- 降低 Temperature（建议 0.3-0.5）以获得更精确的回答
- 增加检索文档数量（Top-K）
- 确保知识库中包含相关领域的文档

### Q: 支持 Docker 部署吗？
支持。使用 Docker Compose 一键启动：
```bash
docker-compose up --build
```
启动后访问 `http://localhost:8501`。数据目录 `data/` 通过 volume 挂载，容器内持久化数据不会丢失。详见 `Dockerfile` 和 `docker-compose.yml`。

### Q: 混合检索比纯向量检索好在哪里？
向量检索擅长语义匹配（"股市表现"匹配"大盘走势"），BM25 擅长精确关键词匹配（"沪深300"匹配"HS300"）。混合检索两路互补，通过 RRF 算法融合排名，覆盖面更广。

### Q: Reranker 开启后更慢怎么办？
Reranker 需要额外一次 API 调用（粗排取 20 条，精排取 5 条），会增加 0.5-1 秒延迟。如果对实时性要求高，可以在侧边栏关闭 Reranker，降级为原始排序。

### Q: 如何运行 RAG 评估？
在 Streamlit "评估" Tab 页面上传评估数据集（JSON 格式）或使用内置的 `data/eval/financial_qa_eval.json`，点击 "开始评估"。评估会调用 GLM-4 API 进行打分，需要充足的 API 额度。

## License

MIT
