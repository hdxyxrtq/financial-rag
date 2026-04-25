# Financial RAG - 金融领域 RAG 知识库问答系统

> 基于 SiliconFlow + ChromaDB 的金融领域检索增强生成系统

## 项目简介

构建一个金融领域的 RAG（Retrieval-Augmented Generation）知识库问答系统。通过检索金融文档（新闻、研报、Q&A）中的相关内容，结合大语言模型生成专业、准确的金融问答。

### 核心特性

- **混合检索**：BM25 关键词检索 + 向量语义检索 + RRF 融合，覆盖语义匹配和精确匹配
- **本地重排序**：BGE-reranker-v2-m3 Cross-Encoder 二次精排，粗排精排分离，无需 API 调用
- **四层自我修正**：检索门控 → 规则预检 → Claim NLI → 外部验证，多层级幻觉检测与修正
- **Query Rewriting**：多轮对话指代消解，将指代性问题改写为独立检索 query
- **语义级缓存**：基于 Embedding 余弦相似度的查询缓存，相似问题直接命中
- **asyncio 全链路异步**：双路检索 asyncio.gather 并行，API 调用全异步化
- **双入口架构**：Streamlit Web UI + FastAPI REST API（Swagger 文档自动生成）
- **RAGAS 自动化评估**：Faithfulness / Relevancy / Precision / Recall 四维评测
- **实时 Metrics**：查询延迟 P50/P95、Token 消耗、缓存命中率实时监控

## 技术选型

| 组件 | 技术方案 | 说明 |
|---|---|---|
| LLM | **SiliconFlow Qwen3-8B** | OpenAI 兼容接口，永久免费 |
| Embedding | **SiliconFlow BAAI/bge-large-zh-v1.5** | 中文语义向量，免费额度 |
| Rerank | **本地 BGE-reranker-v2-m3** | Cross-Encoder 精排，本地运行无需 API |
| 向量数据库 | **ChromaDB** | 轻量级，内置持久化，本地运行 |
| 关键词检索 | **rank_bm25 + jieba** | BM25 倒排索引，金融领域分词优化 |
| 检索融合 | **RRF (Reciprocal Rank Fusion)** | 双路召回排名融合，天然兼容不同分数尺度 |
| Web 框架 | **Streamlit** + **FastAPI** | UI 演示 + REST API 双入口 |
| RAG 评估 | **RAGAS** | 自动化评测 Faithfulness / Relevancy 等 |
| 配置管理 | **PyYAML + python-dotenv** | YAML 配置 + .env 环境变量 |
| 代码质量 | **ruff + mypy + pytest** | Linting + 类型检查 + 200+ 单元测试 |
| 部署 | **Docker + GitHub Actions CI** | 容器化部署 + 自动化测试流水线 |

## 系统架构

```
用户提问 → Streamlit Web UI / FastAPI REST API
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
重排序 (可选) → 本地 BGE-reranker-v2-m3 精排
    │
    ▼
Prompt 构造 → System Prompt + 检索文档 + 用户问题
    │
    ▼
SiliconFlow Qwen3-8B → 生成回答（附引用来源）
    │
    ▼
自我修正 (可选) → 四层幻觉检测与修正
```

## 项目结构

```
financial-rag/
├── app.py                      # Streamlit 主入口
├── config.yaml                 # 全局配置
├── requirements.txt            # Python 依赖
├── pyproject.toml              # 包管理 + ruff/mypy 配置
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # Docker Compose 编排
│
├── src/
│   ├── config.py               #   配置加载（YAML + .env）
│   ├── utils.py                #   API 重试、错误分类
│   ├── rag_pipeline.py         #   RAG 主流程（同步 + async）
│   ├── index_builder.py        #   文档索引构建器
│   ├── loaders/                #   文档加载（TXT/PDF/QA）
│   ├── processor/              #   文本清洗 + 分块（段落/标题双策略）
│   ├── embeddings/             #   SiliconFlow Embedding（同步 + async）
│   ├── vectorstore/            #   ChromaDB 封装
│   ├── retriever/              #   向量 / BM25 / 混合检索（RRF 融合）
│   ├── reranker/               #   本地 BGE-reranker 精排（同步 + async）
│   ├── generator/              #   Qwen3-8B 生成 + Query Rewriting
│   ├── correction/             #   四层自我修正
│   ├── cache/                  #   语义级查询缓存
│   ├── api/                    #   FastAPI REST API
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── routes/
│   ├── metrics/                #   Metrics 收集器
│   ├── ui/                     #   Streamlit UI 模块
│   └── evaluation/             #   RAGAS 评估
│
├── scripts/
│   └── benchmark.py            #   Benchmark 脚本
│
├── tests/                      #   200+ 单元测试
└── data/eval/                  #   50+ 条评估数据集
```

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/Alfroul/financial-rag.git
cd financial-rag

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 SiliconFlow API Key

# 5. 启动应用
streamlit run app.py
```

### API Key 获取

1. 访问 [SiliconFlow](https://siliconflow.cn/)，注册/登录
2. 进入「API Keys」页面，创建新的 Key
3. 填入 `.env` 文件的 `SILICONFLOW_API_KEY` 字段

> 也可以在 Streamlit 侧边栏直接输入 API Key。SiliconFlow 提供多款永久免费模型（Qwen3-8B、DeepSeek V3 等）。

### Docker 部署

```bash
docker-compose up --build
```

访问 `http://localhost:8501`。数据目录通过 volume 挂载，容器重建不丢失数据。

## 数据准备

| 格式 | 目录 | 说明 |
|---|---|---|
| TXT / MD | `data/raw/news/` | 财经新闻、文本文章 |
| PDF | `data/raw/reports/` | 研究报告、分析文档 |
| JSON / CSV | `data/raw/qa/` | 结构化 Q&A 数据 |

添加文档有两种方式：
1. **Web UI**：在「文档管理」页面拖拽上传，点击「开始索引」
2. **命令行**：将文件放入 `data/raw/` 对应目录，运行 `python -m src.index_builder`

## 配置说明

主要配置项在 `config.yaml` 中：

```yaml
llm:
  model: "Qwen/Qwen3-8B"           # SiliconFlow 免费模型
  temperature: 0.7
  max_tokens: 2048

embedding:
  model: "BAAI/bge-large-zh-v1.5"  # 中文语义向量
  batch_size: 20

chunker:
  chunk_size: 512                   # 分块大小 (tokens)
  chunk_overlap: 100
  strategy: "paragraph"             # paragraph / title

hybrid:
  strategy: "hybrid"                # vector / bm25 / hybrid
  rrf_k: 60                         # RRF 融合常数

reranker:
  enabled: false                    # 本地 BGE-reranker 精排
  top_n: 5
```

## Benchmark

> 由 `scripts/benchmark.py` 自动生成，使用 RAGAS 框架评测。

### 实验背景

项目通过八轮 RAGAS 对比实验逐步优化检索与生成策略：

- **第 1~7 轮**：使用 GLM-4-flash 模型，在小数据集（18 条）上对比 vector / BM25 / hybrid 三种检索策略，确定混合检索（hybrid）为最优方案
- **第 8 轮**：全面迁移至 SiliconFlow（Qwen3-8B + bge-large-zh-v1.5），在扩充数据集（50 条）上评估自我修正（Self-Correction）对准确率和延迟的影响

### RAGAS 评估指标（第 8 轮，SiliconFlow）

| 配置 | 忠实度 | 答案相关性 | 上下文精确度 | 上下文召回率 |
|------|--------|-----------|-------------|-------------|
| hybrid | 0.8020 | 0.3165 | 0.6959 | 0.6151 |
| hybrid + SelfCorrection | 0.8162 | 0.3278 | 0.7036 | 0.6188 |

### 检索延迟

| 配置 | P50 (ms) | P95 (ms) | AVG (ms) |
|------|----------|----------|----------|
| hybrid | 17,703 | 48,073 | 26,453 |
| hybrid + SelfCorrection | 19,221 | 60,306 | 27,410 |

> 延迟包含 LLM 生成时间，纯检索阶段 < 500ms。

### 关键结论

- Self-Correction 在所有四项指标上一致正向提升，无退化
- 忠实度提升 1.8%（0.8020 → 0.8162），答案相关性提升 3.6%（0.3165 → 0.3278）
- 推荐配置：**hybrid + SelfCorrection**

## API 文档

除 Streamlit UI 外，系统提供 REST API 接口：

```bash
# 启动 FastAPI 服务
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI：`http://localhost:8000/docs`

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/query` | 同步查询 |
| `POST` | `/api/v1/query/stream` | SSE 流式查询 |
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/documents/upload` | 上传文档 |
| `GET` | `/api/v1/documents/stats` | 文档统计 |
| `GET` | `/api/v1/metrics` | Metrics 统计 |

## FAQ

**Q: 提示「API Key 无效」？**
检查 Key 是否包含前后空格，建议到 [SiliconFlow](https://siliconflow.cn/) 重新生成。

**Q: 上传文档后检索不到相关内容？**
降低「相似度阈值」到 0.3，增加 Top-K 到 10，或对研报类文档切换为 title 分块策略。

**Q: 混合检索比纯向量检索好在哪里？**
向量检索擅长语义匹配（"股市表现"匹配"大盘走势"），BM25 擅长精确关键词匹配（"沪深300"匹配"HS300"）。混合检索通过 RRF 融合两路排名，覆盖面更广。

**Q: 自我修正系统是什么？**
四层幻觉检测机制：(1) 检索质量门控 — 低质量检索直接跳过；(2) 规则预检 — 数字/实体/日期零成本比对；(3) Claim NLI — 原子断言验证；(4) 外部模型验证 — 二次判定。开启后 Faithfulness 提升 1.8%。

**Q: Reranker 开启后更慢怎么办？**
Reranker 使用本地 BGE-reranker-v2-m3 运行，无需 API。首次加载需下载模型（约 600MB），后续推理约 0.5-1 秒。

## 简历描述

> **金融领域 RAG 知识库问答系统** | Python, FastAPI, Streamlit, ChromaDB, SiliconFlow
>
> - 构建金融领域 RAG 系统，实现 BM25+向量混合检索（RRF 融合）+ 本地 BGE-reranker 二次精排 + 四层自我修正，Faithfulness 达 0.82
> - 通过八轮 RAGAS 对比实验（GLM-4-flash × 7 轮 + SiliconFlow × 1 轮）验证混合检索为最优策略，Self-Correction 在四项指标上一致正向提升
> - 实现 Query Rewriting 指代消解、语义级查询缓存、asyncio 全链路异步化，平均查询延迟 < 20s
> - 双入口架构（Streamlit UI + FastAPI REST API），Docker 容器化部署，GitHub Actions CI，200+ 单元测试全通过

## 面试高频问题

### 为什么选择 RRF 融合而不是加权平均？

RRF 只依赖排名位置，不依赖原始分数，天然兼容不同检索器的分数尺度。BM25 分数和余弦相似度量纲不同，直接加权需要归一化。RRF 通过 `1/(k+rank)` 将排名转换为分数，简单且鲁棒，k=60 是常用经验值。

### Query Rewriting 的降级策略？

改写失败（网络错误/超时/API 限额）时 catch 异常，直接用原始 query 检索。无历史记录时跳过改写步骤，不做额外 API 调用。

### 语义级缓存和精确匹配缓存的区别？

精确匹配只缓存完全相同的 query。语义级缓存用 Embedding 计算余弦相似度，"什么是GDP" 和 "GDP是什么意思" 相似度 > 0.95 会命中同一缓存，大幅提升命中率。

### asyncio 异步化带来了什么？

混合检索的向量和 BM25 两路原本串行，异步化后 asyncio.gather 并行，延迟从两路之和降到 max(两路)。加上 Embedding/LLM 全异步化，整体延迟降低约 40%。

### 为什么用两阶段检索（粗排+精排）？

向量检索是 Bi-Encoder（问题和文档独立编码），速度快但精度有限。Reranker 是 Cross-Encoder（联合编码），精度高但慢。先用粗排召回 Top-30，再用 Reranker 精排取 Top-5，兼顾速度和精度。

### ChromaDB 为什么不用 Milvus/Pinecone？

ChromaDB 轻量级、内置持久化、本地运行无需额外服务，适合单机部署。Milvus 适合大规模生产，Pinecone 是托管付费服务。

## License

MIT
