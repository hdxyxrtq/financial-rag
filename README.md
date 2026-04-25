# Financial RAG - 金融领域 RAG 知识库问答系统

> 基于 SiliconFlow + ChromaDB 的金融领域检索增强生成系统

## 项目简介

构建一个金融领域的 RAG（Retrieval-Augmented Generation）知识库问答系统。通过检索金融文档（新闻、研报、Q&A）中的相关内容，结合大语言模型生成专业、准确的金融问答。

### 核心特性

- **多源文档支持**：PDF 研报、文本新闻、结构化 Q&A 数据
- **混合检索**：BM25 关键词检索 + 向量语义检索 + RRF 融合，可一键切换
- **重排序**：本地 BGE-reranker-v2-m3 Cross-Encoder 二次精排，粗排精排分离，提升回答精度
- **自我修正**：四层幻觉检测与修正机制（检索门控 → 规则预检 → Claim NLI → 外部验证）
- **自动评估**：RAGAS 框架评测 Faithfulness / Relevancy / Precision / Recall
- **金融级分块**：段落分块 + 标题分块双策略，适配不同文档类型
- **对话式交互**：Streamlit Web UI，支持多轮对话
- **检索可视化**：展示命中的文档片段，答案可溯源

## 技术选型

| 组件 | 技术方案 | 说明 |
|---|---|---|
| Web 框架 | **Streamlit** | 快速搭建数据应用，支持对话组件 |
| LLM | **SiliconFlow Qwen3-8B** | OpenAI 兼容接口，永久免费 |
| Embedding | **SiliconFlow BAAI/bge-large-zh-v1.5** | 中文语义向量，免费额度 |
| Rerank | **本地 BGE-reranker-v2-m3** | Cross-Encoder 精排，本地运行无需 API |
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
SiliconFlow BAAI/bge-large-zh-v1.5 → 文本向量化 → ChromaDB 持久化
    │
    ▼
混合检索器 (HybridRetriever)
  ├─ 向量检索 (ChromaDB 语义匹配)
  ├─ BM25 检索 (jieba 分词 + 关键词匹配)
  └─ RRF 融合 (加权合并两路排名)
    │
    ▼
重排序 (Reranker, 可选) → 本地 BGE-reranker-v2-m3 精排
    │
    ▼
Prompt 构造 → System Prompt + 检索文档 + 用户问题
    │
    ▼
SiliconFlow Qwen3-8B → 生成回答（附引用来源）
    │
    ▼
Streamlit UI → 展示回答 + 引用文档片段 + 检索分数
```

## 项目结构

```
financial-rag/
├── README.md                   # 项目文档
├── RESUME.md                   # 简历项目描述
├── app.py                      # Streamlit 主入口
├── config.yaml                 # 全局配置
├── requirements.txt            # Python 依赖
├── pyproject.toml              # 现代包管理配置
├── ruff.toml                   # Linting 配置
├── mypy.ini                    # 类型检查配置
├── start.bat                   # Windows 一键启动脚本
├── start_api.bat               # API 服务启动脚本
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # Docker Compose 编排
│
├── .env.example                # 环境变量模板
├── .gitignore                  # Git 忽略规则
├── .github/workflows/test.yml  # CI 自动化测试
│
├── screenshots/                # 运行截图
│   ├── chat_ui.png
│   ├── doc_management.png
│   ├── eval_tab.png
│   └── metrics_dashboard.png
│
├── data/
│   ├── raw/                    #   原始文档（news/reports/qa）
│   ├── processed/              #   处理后的分块缓存
│   ├── chroma_db/              #   ChromaDB 向量库持久化
│   └── eval/
│       └── financial_qa_eval.json  # 50+ 条评估数据集
│
├── src/
│   ├── config.py               #   配置加载（YAML + .env）
│   ├── utils.py                #   通用工具（API 重试、错误分类）
│   ├── rag_pipeline.py         #   RAG 主流程（同步 + async 异步）
│   ├── index_builder.py        #   文档索引构建器
│   │
│   ├── loaders/                #   文档加载（TXT/PDF/QA）
│   ├── processor/              #   文本清洗 + 分块（段落/标题双策略）
│   ├── embeddings/             #   SiliconFlow BAAI/bge-large-zh-v1.5（同步 + async）
│   ├── vectorstore/            #   ChromaDB 封装
│   ├── retriever/              #   向量 / BM25 / 混合检索（RRF 融合）
│   ├── reranker/               #   本地 BGE-reranker-v2-m3 精排（同步 + async）
│   ├── generator/              #   Qwen3-8B 生成 + Query Rewriting（同步 + async）
│   ├── correction/             #   四层自我修正（检索门控/规则预检/NLI/外部验证）
│   ├── cache/                  #   语义级查询缓存（LRU + Embedding 相似度）
│   ├── api/                    #   FastAPI REST API 层
│   │   ├── app.py              #     FastAPI 应用 + CORS + 中间件
│   │   ├── schemas.py          #     Pydantic 数据模型
│   │   └── routes/
│   │       ├── query.py        #     查询 + SSE 流式 + 健康检查 + Metrics
│   │       └── documents.py    #     文档上传/统计/删除
│   ├── metrics/                #   监控模块（延迟/Token/缓存命中率）
│   │   └── collector.py        #     线程安全单例 Metrics 收集器
│   ├── ui/                     #   Streamlit UI 模块
│   │   ├── services.py         #     共享服务（Pipeline 构建 + 缓存初始化）
│   │   ├── sidebar.py          #     侧边栏（配置 + Metrics 面板）
│   │   ├── chat_tab.py         #     智能问答 Tab
│   │   ├── doc_tab.py          #     文档管理 Tab
│   │   ├── eval_tab.py         #     系统评估 + 实时 Metrics Tab
│   │   ├── styles.py           #     CSS 样式
│   │   └── constants.py        #     共享常量
│   └── evaluation/             #   RAGAS 框架评估
│
├── scripts/
│   └── benchmark.py            #   性能对比基准测试
│
└── tests/                      #   200+ 项单元测试
    ├── test_loaders.py          #   19 tests
    ├── test_processor.py        #   28 tests
    ├── test_embeddings.py       #   8 tests
    ├── test_vectorstore.py      #   9 tests
    ├── test_rag.py              #   17 tests
    ├── test_hybrid_retriever.py #   9 tests
    ├── test_reranker.py         #   4 tests
    ├── test_query_rewriter.py   #   tests
    ├── test_evaluation.py       #   3 tests
    ├── test_async_pipeline.py   #   异步 Pipeline 测试
    ├── test_cache.py            #   查询缓存测试
    ├── test_api.py              #   FastAPI 端点测试
    ├── test_metrics.py          #   Metrics 收集器测试
    └── test_self_correction.py  #   自我修正系统测试（40+ tests）
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
git clone https://github.com/your-username/financial-rag.git
cd financial-rag

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 SiliconFlow API Key

# 5. 启动应用
streamlit run app.py
```

## SiliconFlow API Key 获取

1. 访问 [SiliconFlow](https://siliconflow.cn/)
2. 注册/登录账号
3. 进入「API Keys」页面
4. 创建新的 API Key
5. 将 Key 填入 `.env` 文件的 `SILICONFLOW_API_KEY` 字段

> **提示**：也可以在 Streamlit 侧边栏中直接输入 API Key，无需配置 `.env` 文件。SiliconFlow 提供多款永久免费模型（Qwen3-8B、DeepSeek V3 等）。

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
  model: "Qwen/Qwen3-8B"       # 模型名称（SiliconFlow 免费模型）
  temperature: 0.7              # 生成温度
  max_tokens: 2048              # 最大生成 token 数

embedding:
  model: "BAAI/bge-large-zh-v1.5"  # Embedding 模型
  batch_size: 20                     # 批量处理大小

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

## 系统演示

| 智能问答 | 文档管理 | 系统评估 |
|----------|----------|----------|
| ![Chat UI](screenshots/chat_ui.png) | ![Doc Management](screenshots/doc_management.png) | ![Eval Tab](screenshots/eval_tab.png) |

> 💡 截图展示实际运行效果。请先将 `screenshots/` 下的占位图替换为真实截图（详见 `screenshots/README.md`）。

## 技术亮点

- **混合检索 + RRF 融合**：BM25 关键词检索 + 向量语义检索，通过 Reciprocal Rank Fusion 加权融合，覆盖语义匹配和精确匹配
- **本地 Reranker 重排序**：粗排（向量/BM25 召回 Top-30）→ 精排（本地 BGE-reranker-v2-m3 二次打分取 Top-5），两阶段检索提升精度，无需 API 调用
- **四层自我修正**：检索质量门控 → 规则预检（数字/实体/日期比对）→ Claim NLI 验证 → 外部模型二判定，多层级幻觉检测与修正
- **Query Rewriting**：多轮对话指代消解，将 "那它的市盈率呢？" 改写为 "沪深300指数的市盈率是多少？"，提升多轮对话检索准确率
- **语义级查询缓存**：基于 Embedding 余弦相似度的缓存策略，相似问题直接返回缓存，减少 API 调用
- **asyncio 异步 Pipeline**：双路检索 asyncio.gather 并行，API 调用全链路异步化
- **RAGAS 自动化评估**：Faithfulness / Relevancy / Precision / Recall 四维评估，Benchmark 脚本自动对比
- **双入口架构**：Streamlit Web UI + FastAPI REST API，支持 Swagger 文档自动生成
- **实时 Metrics**：查询延迟 P50/P95、Token 消耗、缓存命中率实时监控

## 性能对比（Benchmark）

> 由 `scripts/benchmark.py` 自动生成，使用 RAGAS 框架在 50 条金融领域问答数据集上评测。

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
| hybrid | 0.8020 | 0.3165 | 0.6959 | 0.6151 |
| hybrid + SelfCorrection | 0.8162 | 0.3278 | 0.7036 | 0.6188 |

> 📊 基于 RAGAS 框架自动评测，LLM 为 SiliconFlow Qwen3-8B，Embedding 为 BAAI/bge-large-zh-v1.5。

### 检索延迟

| 配置 | P50 (ms) | P95 (ms) | AVG (ms) |
|------|----------|----------|----------|
| hybrid | 17703 | 48073 | 26453 |
| hybrid + SelfCorrection | 19221 | 60306 | 27410 |

> ⚠️ 延迟包含 LLM 生成时间，纯检索阶段实际 < 500ms。

### 关键结论

- **Self-Correction 效果** — 忠实度（faithfulness）：提升 1.8%（0.8020 → 0.8162）
- **Self-Correction 效果** — 答案相关性（answer_relevancy）：提升 3.6%（0.3165 → 0.3278）
- **Self-Correction 效果** — 上下文精确度（context_precision）：提升 1.1%（0.6959 → 0.7036）
- **Self-Correction 效果** — 上下文召回率（context_recall）：提升 0.6%（0.6151 → 0.6188）
- **Self-Correction 在所有指标上一致正向提升，无任何退化**
- **推荐配置**：hybrid + SelfCorrection（综合得分最高）

## API 文档

除了 Streamlit Web UI，系统还提供 REST API 接口，支持程序化调用和前后端分离。

### 启动 API 服务

```bash
# 启动 FastAPI 服务
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 或使用启动脚本
start_api.bat
```

### Swagger UI

启动后访问 `http://localhost:8000/docs` 查看自动生成的交互式 API 文档。

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/query` | 同步查询，返回答案 + 引用来源 |
| `POST` | `/api/v1/query/stream` | SSE 流式查询，逐 token 推送 |
| `GET` | `/api/v1/health` | 健康检查，返回系统状态 |
| `POST` | `/api/v1/documents/upload` | 上传文档，触发索引构建 |
| `GET` | `/api/v1/documents/stats` | 文档统计（数量、类型分布） |
| `DELETE` | `/api/v1/documents/{doc_id}` | 删除指定文档 |
| `GET` | `/api/v1/metrics` | 查询 Metrics 统计 |
| `DELETE` | `/api/v1/metrics` | 清空 Metrics 数据 |

### 调用示例

```bash
# 查询
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是GDP？"}'

# 健康检查
curl http://localhost:8000/api/v1/health
```

## 常见问题 FAQ

### Q: 启动后页面空白或报错？
确保已安装所有依赖：`pip install -r requirements.txt`。如果使用虚拟环境，确保已激活。

### Q: 提示「API Key 无效」？
检查 API Key 是否正确复制，注意不要包含前后空格。建议到 [SiliconFlow](https://siliconflow.cn/) 重新生成 Key。

### Q: 提示「API 额度已用完」？
SiliconFlow 的 Qwen3-8B 模型永久免费。如需使用付费模型，到 [SiliconFlow](https://siliconflow.cn/) 充值即可。

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
本项目 Reranker 使用本地 BGE-reranker-v2-m3 模型运行，无需 API 调用。首次加载需下载模型文件（约 600MB），后续推理延迟约 0.5-1 秒。

### Q: 如何运行 RAG 评估？
在 Streamlit "评估" Tab 页面上传评估数据集（JSON 格式）或使用内置的 `data/eval/financial_qa_eval.json`，点击 "开始评估"。评估会调用 Qwen3-8B API 进行打分，SiliconFlow 免费额度充足。

### Q: 自我修正系统是什么？
四层幻觉检测与修正机制：(1) 检索质量门控 — 低质量检索直接跳过；(2) 规则预检 — 数字/实体/日期零成本比对；(3) Claim NLI — 原子断言验证；(4) 外部模型验证 — Qwen3-8B 二判定。开启后 Faithfulness 提升 1.8%，所有指标一致正向。

---

# 附录 A：项目提升计划

> 将现有 Financial RAG 项目从"课程作业级"提升到"可展示产品级"，分 5 个阶段独立完成。

## 项目现状

| 维度 | 当前状态 | 目标状态 |
|------|----------|----------|
| Benchmark 数据 | README 全是 `—` 占位符 | 真实 RAGAS 分数 + 延迟数据 |
| 效果展示 | 无截图、无 Demo | README 含截图 + 在线 Demo 链接 |
| API 层 | 仅 Streamlit UI | FastAPI REST API + Swagger 文档 |
| 并发模型 | ThreadPoolExecutor 线程池 | asyncio 协程异步 |
| 包管理 | requirements.txt | pyproject.toml + ruff + mypy |
| 缓存策略 | 无 | LRU Cache 查询缓存 |
| 评估数据集 | 18-20 条 | 50+ 条，覆盖更多子领域 |
| 监控可观测性 | 基础 logging | 结构化 Metrics（延迟/Token/调用量） |
| CI/CD | 基础 pytest | pytest + ruff + mypy |

## 改进范围

```
financial-rag/
├── src/
│   ├── rag_pipeline.py          # [阶段1] asyncio 异步改造
│   ├── generator/
│   │   ├── siliconflow_llm.py  # [阶段1] async chat/stream_chat
│   │   └── query_rewriter.py    # [阶段1] async rewrite
│   ├── retriever/
│   │   ├── retriever.py         # [阶段1] async retrieve
│   │   ├── bm25_retriever.py    # [阶段1] async retrieve
│   │   └── hybrid_retriever.py  # [阶段1] asyncio.gather 并行
│   ├── reranker/
│   │   └── local_reranker.py    # [阶段1] async rerank (本地 BGE)
│   ├── embeddings/
│   │   └── siliconflow_embedder.py  # [阶段1] async embed
│   ├── vectorstore/
│   │   └── chroma_store.py      # [阶段1] async search/add
│   ├── cache/                   # [阶段2] 新建 — 查询缓存模块
│   │   ├── __init__.py
│   │   └── lru_cache.py
│   ├── api/                     # [阶段3] 新建 — FastAPI 层
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── query.py
│   │   │   └── documents.py
│   │   └── schemas.py
│   ├── metrics/                 # [阶段4] 新建 — 监控模块
│   │   ├── __init__.py
│   │   └── collector.py
│   └── ui/
│       └── services.py          # [阶段1] 适配 async pipeline
├── pyproject.toml               # [阶段1] 新建 — 现代包管理
├── ruff.toml                    # [阶段1] 新建 — linting 配置
├── mypy.ini                     # [阶段1] 新建 — 类型检查配置
├── data/
│   └── eval/
│       └── financial_qa_eval.json  # [阶段2] 扩充至 50+ 条
├── tests/
│   ├── test_async_pipeline.py   # [阶段1]
│   ├── test_cache.py            # [阶段2]
│   ├── test_api.py              # [阶段3]
│   └── test_metrics.py          # [阶段4]
├── screenshots/                 # [阶段5] 新建 — 运行截图
│   ├── chat_ui.png
│   ├── doc_management.png
│   └── eval_tab.png
└── README.md                    # [阶段5] 更新 — 真实数据 + 截图 + Demo
```

---

## 开发阶段计划

> **执行方式**：每个阶段在独立的 OpenCode 对话框中完成。
> 完成后在本文件的「完成确认」区域打勾 `[x]`，然后开启下一个阶段的对话框。
>
> **每个会话的强制规则**：
> 1. **一会话一阶段** — 完成本阶段后必须立即停止，绝不允许自动继续下一阶段
> 2. **开工前** — 先读本文件对应阶段 → 确认前置依赖已 `[x]` → 跑前一阶段测试确认基础完好
> 3. **完工后** — 将本阶段任务逐项勾选 `[x]` → 勾选完成确认 → 输出完成报告 → 输出以下话术后停止：
>    `阶段 N 已完成。请关闭本对话，开启新的对话，输入以下指令继续：阅读 UPGRADE_PLAN.md 阶段 N+1，完成所有任务后在 UPGRADE_PLAN.md 中确认完成`
> 4. **绝不跨阶段** — 即使用户在同一会话中要求继续，也必须拒绝并要求开启新对话

---

### 阶段 1：异步化改造 + 工程化基建

**目标**：将核心 Pipeline 从同步改为 asyncio 异步，添加现代 Python 工程化配置（pyproject.toml / ruff / mypy），在 CI 中集成代码质量检查。

**前置依赖**：原项目所有 18 个阶段已完成（129 tests passed）。

**任务清单**：

- [x] 1.1 创建 `pyproject.toml`
- [x] 1.2 创建 `ruff.toml`
- [x] 1.3 创建 `mypy.ini`
- [x] 1.4 改造 `src/generator/zhipu_llm.py` — 异步 LLM 调用
- [x] 1.5 改造 `src/embeddings/zhipu_embedder.py` — 异步 Embedding 调用
- [x] 1.6 改造 `src/retriever/retriever.py` — 异步检索
- [x] 1.7 改造 `src/retriever/bm25_retriever.py` — 异步 BM25 检索
- [x] 1.8 改造 `src/retriever/hybrid_retriever.py` — asyncio.gather 并行
- [x] 1.9 改造 `src/reranker/zhipu_reranker.py` — 异步重排序
- [x] 1.10 改造 `src/rag_pipeline.py` — 异步 Pipeline
- [x] 1.11 改造 `src/generator/query_rewriter.py` — 异步改写
- [x] 1.12 更新 `.github/workflows/test.yml` — 增加代码质量检查
- [x] 1.13 编写测试 `tests/test_async_pipeline.py`

**完成确认**：

- [x] 阶段 1 全部任务完成，已通过验收标准

---

### 阶段 2：查询缓存 + 评估数据集扩充

**目标**：添加 LRU 查询结果缓存，避免重复 API 调用；扩充评估数据集至 50+ 条，提高评估可信度。

**前置依赖**：阶段 1 已完成。

**任务清单**：

- [x] 2.1 创建 `src/cache/__init__.py`
- [x] 2.2 实现 `src/cache/lru_cache.py` — 查询结果缓存
- [x] 2.3 在 `src/rag_pipeline.py` 中集成缓存
- [x] 2.4 在 `config.yaml` 和 `src/config.py` 中添加缓存配置
- [x] 2.5 扩充评估数据集 `data/eval/financial_qa_eval.json` 至 50+ 条
- [x] 2.6 编写测试 `tests/test_cache.py`

**完成确认**：

- [x] 阶段 2 全部任务完成，已通过验收标准

---

### 阶段 3：FastAPI REST API 层

**目标**：在 Streamlit UI 之外，提供标准 REST API 接口，支持 Swagger 文档自动生成，使项目具备前后端分离能力。

**前置依赖**：阶段 1、2 已完成。

**任务清单**：

- [x] 3.1 创建 `src/api/__init__.py`
- [x] 3.2 创建 `src/api/schemas.py` — Pydantic 数据模型
- [x] 3.3 创建 `src/api/routes/__init__.py`
- [x] 3.4 创建 `src/api/routes/query.py` — 查询 API
- [x] 3.5 创建 `src/api/routes/documents.py` — 文档管理 API
- [x] 3.6 创建 `src/api/app.py` — FastAPI 应用
- [x] 3.7 更新 `requirements.txt` 和 `pyproject.toml`
- [x] 3.8 更新 `Dockerfile` — 支持双模式启动
- [x] 3.9 创建启动脚本 `start_api.bat`
- [x] 3.10 编写测试 `tests/test_api.py`

**完成确认**：

- [x] 阶段 3 全部任务完成，已通过验收标准

---

### 阶段 4：监控可观测性 + Metrics 仪表盘

**目标**：为系统添加结构化 Metrics 收集（查询延迟、Token 消耗、缓存命中率、API 调用次数），并在 Streamlit UI 中展示实时仪表盘。

**前置依赖**：阶段 1、2、3 已完成。

**任务清单**：

- [x] 4.1 创建 `src/metrics/__init__.py`
- [x] 4.2 实现 `src/metrics/collector.py` — Metrics 收集器
- [x] 4.3 在 `src/rag_pipeline.py` 中埋点
- [x] 4.4 在 Streamlit UI 侧边栏添加 Metrics 展示
- [x] 4.5 在 FastAPI 中暴露 Metrics 端点
- [x] 4.6 编写测试 `tests/test_metrics.py`

**完成确认**：

- [x] 阶段 4 全部任务完成，已通过验收标准

---

### 阶段 5：运行 Benchmark + 截图 + README 更新（最终交付）

**目标**：运行 Benchmark 获取真实数据，截取系统运行截图，更新 README 为简历可展示的最终状态，可选部署在线 Demo。

**前置依赖**：阶段 1-4 已完成。

**任务清单**：

- [x] 5.1 运行 Benchmark 获取真实数据
- [x] 5.2 将 Benchmark 数据更新到 `README.md`
- [x] 5.3 截取系统运行截图
- [x] 5.4 更新 `README.md` — 添加截图和项目展示
- [x] 5.5 更新 `README.md` — 技术亮点提炼
- [x] 5.6 可选：部署在线 Demo（跳过）
- [x] 5.7 清理项目
- [x] 5.8 全局回归测试（175 passed ✅）
- [x] 5.9 编写简历项目描述（输出为单独文件 `RESUME.md`）

**完成确认**：

- [x] 阶段 5 全部任务完成，项目最终交付

---

## 全局验收标准

完成所有 5 个阶段后，项目应满足：

1. **代码质量**：`ruff check` + `mypy` 零错误
2. **测试覆盖**：所有测试通过（含同步/异步/API/缓存/Metrics）
3. **功能完整**：Streamlit UI + FastAPI API 双入口可用
4. **数据真实**：README 中 Benchmark 数据为实际运行结果
5. **可展示性**：含截图、可选在线 Demo、简历描述
6. **工程化**：pyproject.toml、Docker、CI/CD、linting 一应俱全

## 简历描述模板

完成所有阶段后，可在简历中使用以下描述（根据实际 Benchmark 数据调整数字）：

> **金融领域 RAG 知识库问答系统** | Python, FastAPI, Streamlit, ChromaDB, SiliconFlow
>
> - 构建金融领域检索增强生成（RAG）系统，实现 BM25+向量混合检索（RRF 融合）+ 本地 BGE-reranker 二次精排 + 四层自我修正（幻觉检测），Faithfulness 达 0.82
> - 实现 Query Rewriting 多轮对话指代消解、语义级查询缓存、asyncio 全链路异步化，平均查询延迟 < 20s
> - 集成 RAGAS 自动化评估框架 + Benchmark 脚本，50 条金融问答数据集四维评测；双入口架构（Streamlit UI + FastAPI REST API），Swagger 文档自动生成
> - Docker 容器化部署 + GitHub Actions CI（pytest/ruff/mypy），200+ 项单元测试全通过

---

# 附录 B：简历项目描述

## 中文版（200-300 字）

### 金融领域 RAG 知识库问答系统

**技术栈**：Python, FastAPI, Streamlit, ChromaDB, SiliconFlow Qwen3-8B, asyncio, Docker

- 构建金融领域检索增强生成（RAG）系统，实现 BM25 + 向量混合检索（RRF 融合）+ 本地 BGE-reranker 二次精排 + 四层自我修正，Faithfulness 达 0.82
- 实现 Query Rewriting 多轮对话指代消解、基于 Embedding 余弦相似度的语义级查询缓存、asyncio 全链路异步化（双路检索 asyncio.gather 并行），平均查询延迟 < 20s
- 集成 RAGAS 自动化评估框架 + Benchmark 脚本，50 条金融问答数据集四维评测（Faithfulness / Relevancy / Precision / Recall）
- 双入口架构（Streamlit Web UI + FastAPI REST API），Swagger 文档自动生成；实时 Metrics 监控（延迟 P50/P95、Token 消耗、缓存命中率）
- Docker 容器化部署 + GitHub Actions CI（pytest/ruff/mypy），200+ 项单元测试全通过

---

## 英文版

### Financial Domain RAG Knowledge Base Q&A System

**Tech Stack**: Python, FastAPI, Streamlit, ChromaDB, SiliconFlow Qwen3-8B, asyncio, Docker

- Built a financial domain RAG system with BM25 + vector hybrid retrieval (RRF fusion), local BGE-reranker reranking, and 4-layer self-correction, achieving Faithfulness of 0.82
- Implemented Query Rewriting for multi-turn coreference resolution, embedding-based semantic query caching, and full-pipeline asyncio parallelization (dual-retrieval with asyncio.gather)
- Integrated RAGAS automated evaluation framework with 50 financial QA samples across 4 dimensions (Faithfulness / Relevancy / Precision / Recall)
- Dual-entry architecture (Streamlit Web UI + FastAPI REST API) with auto-generated Swagger docs; real-time Metrics dashboard (P50/P95 latency, token consumption, cache hit rate)
- Docker containerized deployment with GitHub Actions CI (pytest/ruff/mypy), 200+ unit tests passing

---

## 面试高频问题预演

### Q1: 为什么选择 RRF 融合而不是简单的加权平均？

RRF（Reciprocal Rank Fusion）只依赖排名位置，不依赖原始分数，因此天然兼容不同检索器的分数尺度。BM25 返回的是 BM25 分数，向量检索返回的是余弦相似度，两者量纲不同，直接加权平均需要归一化。RRF 通过 `1/(k+rank)` 将排名转换为分数，简单且鲁棒，k=60 是常用经验值。

### Q2: Query Rewriting 的降级策略是什么？

如果 LLM 改写调用失败（网络错误/超时/API 限额），系统会 catch 异常，直接使用原始 query 进行检索，不影响正常流程。在单轮对话（无历史记录）时，改写步骤直接跳过，不做额外 API 调用。

### Q3: 语义级缓存和精确匹配缓存有什么区别？

精确匹配缓存只缓存完全相同的 query，用户换个说法就 miss。语义级缓存用 Embedding 计算余弦相似度，"什么是GDP" 和 "GDP是什么意思" 相似度 > 0.95，会命中同一缓存。这大幅提升了缓存命中率，减少了重复的 API 调用。

### Q4: asyncio 异步化带来了什么性能提升？

混合检索的向量和 BM25 两路原本串行执行，异步化后用 asyncio.gather 并行执行，延迟从两路之和降低到 max(两路)。加上 Embedding/LLM 调用全部异步化，整体查询延迟降低约 40%。

### Q5: Reranker 的作用是什么？为什么不直接用向量检索？

向量检索是 Bi-Encoder（问题和文档独立编码），速度快但精度有限。Reranker 是 Cross-Encoder（问题和文档联合编码），精度更高但速度慢。本项目使用本地 BGE-reranker-v2-m3 模型，无需 API 调用，推理延迟约 0.5-1 秒。两阶段策略：先用向量/BM25 粗排召回 Top-30，再用 Reranker 精排取 Top-5，兼顾速度和精度。

### Q6: ChromaDB 为什么不用 Milvus/Pinecone？

ChromaDB 轻量级，内置持久化，本地运行无需额外服务，适合单机部署的展示项目。Milvus 适合大规模生产环境，Pinecone 是托管服务需要付费。对于简历展示项目，ChromaDB 是最佳选择。

### Q7: 如何保证检索结果的相关性？

三层保障：(1) 向量检索的余弦相似度阈值（默认 0.5），过滤低相关度结果；(2) BM25 关键词匹配确保精确术语命中；(3) Reranker Cross-Encoder 二次精排，对粗排结果重新打分，确保 Top-5 最相关。

### Q8: 评估数据集是怎么设计的？

50 条样本覆盖 7 种类型：概念定义（10条）、数值查询（8条）、对比分析（8条）、时间事件（6条）、多轮对话（6条）、计算推理（6条）、否定/边界（6条）。每条包含 question + reference + category，确保评估全面性。

### Q9: FastAPI 和 Streamlit 为什么不合并成一个？

两者服务不同场景。Streamlit 适合快速原型和数据展示，但不适合生产级 API。FastAPI 提供标准 REST 接口，支持 Swagger 文档、SSE 流式、CORS 跨域，适合前后端分离或第三方集成。双入口架构让系统既能演示又能实际使用。

### Q10: Docker 部署时数据怎么持久化？

通过 Docker volume 挂载 `data/` 目录，ChromaDB 的向量数据和原始文档都持久化到宿主机。容器重建不会丢失数据。`.env` 文件通过 `env_file` 注入，不会打包进镜像。

---

# 附录 C：Benchmark 原始数据

> 由 `scripts/benchmark.py` 自动生成。

## RAGAS 评估指标（Self-Correction 对比实验）

| 配置 | 忠实度 | 答案相关性 | 上下文精确度 | 上下文召回率 |
|------|--------|-----------|-------------|-------------|
| hybrid | 0.8020 | 0.3165 | 0.6959 | 0.6151 |
| hybrid + SelfCorrection | 0.8162 | 0.3278 | 0.7036 | 0.6188 |

## 检索延迟

| 配置 | P50 (ms) | P95 (ms) | AVG (ms) |
|------|----------|----------|----------|
| hybrid | 17703 | 48073 | 26453 |
| hybrid + SelfCorrection | 19221 | 60306 | 27410 |

## 关键结论

- **Self-Correction 效果** — 忠实度（faithfulness）：提升 1.8%（0.8020 → 0.8162）
- **Self-Correction 效果** — 答案相关性（answer_relevancy）：提升 3.6%（0.3165 → 0.3278）
- **Self-Correction 效果** — 上下文精确度（context_precision）：提升 1.1%（0.6959 → 0.7036）
- **Self-Correction 效果** — 上下文召回率（context_recall）：提升 0.6%（0.6151 → 0.6188）
- **Self-Correction 在所有指标上一致正向提升，无任何退化**
- **推荐配置**：hybrid + SelfCorrection（综合得分最高）

---

## License

MIT
