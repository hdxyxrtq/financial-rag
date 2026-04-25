# 自我修正系统开发计划

> 为 Financial RAG 添加 CRAG + Self-Reflection 多层自我修正能力

## 项目简介

在现有 RAG Pipeline 基础上，添加四层幻觉检测与修正机制：
- **Layer 0**：检索质量门控（零成本，复用已有分数信号）
- **Layer 2**：规则预检（零 API 成本，正则提取数字/实体/日期比对）
- **Layer 3**：Claim 分解 + NLI 验证（本地模型，零 API 成本）
- **Layer 4**：外部模型二判定（SiliconFlow Qwen3-8B，永久免费）

## 技术选型

| 组件 | 技术方案 | 说明 |
|---|---|---|
| 架构模式 | Wrapper（包装 RAGPipeline） | 不修改现有代码，向后兼容 |
| Layer 0 | 已有检索分数融合 | top-1 score / reranker score / 检索结果数 |
| Layer 2 | 正则 + 金融术语字典 | 复用 bm25_retriever.py 51 个术语 |
| Layer 3 | Claim 分解 + 本地 NLI | 本地运行，候选：HHEM 或 BGE-reranker 改造 |
| Layer 4 | SiliconFlow Qwen3-8B | OpenAI 兼容，永久免费 |
| 配置 | config.yaml + config.py 扩展 | SelfCorrectionConfig dataclass |

## 系统架构

```
用户提问
  → Rewrite → Hybrid Retrieve → Rerank (已有流程)
  │
  ▼
【Layer 0】检索质量门控
  │ 信号: top-1 retrieval_score, reranker_score, num_sources
  │ ├─ GOOD (top-1 rerank ≥ 0.7) → 正常生成
  │ ├─ WEAK (top-1 rerank < 0.3) → 返回"知识库覆盖不足"
  │ └─ MARGINAL (0.3-0.7) → 生成但标记为低置信
  │
  ▼
【Layer 1】GLM-4-flash 生成回答 (已有，不改动)
  │
  ▼
【Layer 2】规则预检 (零 API 成本)
  │ ① 数字比对: 答案中的数字是否出现在原文？
  │ ② 金融实体比对: 51 个术语是否在原文？
  │ ③ 日期比对: 答案中的日期是否在原文？
  │ ④ 词汇重叠率: answer vs source 词汇重合度
  │ ├─ 全部通过 → 放行
  │ └─ 有疑点 → 进入 Layer 3
  │
  ▼
【Layer 3】Claim 分解 + NLI 验证 (本地运行)
  │ 把回答拆成原子断言 → 每条 claim 用 NLI 判定是否被原文支撑
  │ ├─ 全部支撑 → 放行
  │ └─ 有不支撑的 claim → 标记，进入 Layer 4
  │
  ▼
【Layer 4】外部模型二判定 (SiliconFlow Qwen3-8B)
  │ 把有疑点的 claim + 原文证据送给 Qwen3-8B
  │ Prompt: "以下断言是否被原文支撑？逐条判定并引用原文。"
  │ ├─ 判定为支撑 → 放行
  │ └─ 判定为不支撑 → 注入反馈 → 重新生成（最多 2 次）
```

## 项目结构（新增部分）

```
src/correction/                  # 自我修正模块
├── __init__.py                  # 导出 SelfCorrectingPipeline
├── types.py                     # 共享数据类型 (CorrectionResult, ClaimVerdict 等)
├── retrieval_gate.py            # Layer 0: 检索质量门控
├── rule_checker.py              # Layer 2: 规则预检
├── nli_verifier.py              # Layer 3: Claim 分解 + NLI 验证
├── external_verifier.py         # Layer 4: SiliconFlow Qwen3-8B 验证
└── pipeline.py                  # SelfCorrectingPipeline 主编排器

tests/
└── test_self_correction.py      # 自我修正系统测试

# 修改的文件
src/config.py                    # 新增 SelfCorrectionConfig
config.yaml                      # 新增 self_correction 配置段
.env.example                     # 新增 SILICONFLOW_API_KEY
src/ui/services.py               # 可选：集成 SelfCorrectingPipeline
```

---

## 开发阶段计划

> **执行方式**：每个阶段在独立的 OpenCode 对话框中完成。
> 完成后在本文件的「完成确认」区域打勾 `[x]`，然后开启下一个阶段的对话框。
>
> **每个会话的强制规则（无需额外文档，本段即为完整执行手册）**：
>
> **开工前必做**：
> 1. 读本项目结构 + 当前阶段的全部内容 + 测试命令（不读其他阶段的任务清单）
> 2. 确认前置依赖阶段的完成确认已 `[x]`，未完成则拒绝开始
> 3. 如当前阶段有 Checkpoint 文件（`docs/handoff-stage-N-checkpoint.md`），必须先读 Checkpoint，从中断点继续
> 4. 跑前一阶段的测试命令，确认基础完好再动手：`python -m pytest tests/ -v --tb=short`
> 5. 如前置阶段有遗留问题或测试失败 → 先修复，不带着坏基础继续
>
> **执行中**：
> 6. 一会话一个完整单元（阶段或子阶段），完成后立即停止，绝不允许自动继续下一阶段
> 7. 上下文快满时 → 触发 Checkpoint 机制，禁止硬撑
> 8. 需要回溯修改前序阶段时：
>    - 小修（修 typo、加参数、修 bug）：在当前阶段内直接修复
>    - 大改（改接口、改数据结构、删文件）：停止当前阶段，插入回溯修复阶段
>
> **完工后必做**：
> 9. 将任务清单逐项勾选 `[x]`，将阶段末尾的完成确认勾选 `[x]`
> 10. 删除本阶段的 Checkpoint 文件（如存在）
> 11. 输出阶段完成报告
> 12. 输出以下话术后结束：
>     `阶段 N 已完成。请关闭本对话，开启新的对话，输入以下指令继续：阅读 SELF_CORRECTION_PLAN.md 阶段 N+1，完成所有任务后在 SELF_CORRECTION_PLAN.md 中确认完成`
>
> **绝不允许**：跨阶段推进、跳过前置依赖检查、忽略 Checkpoint、上下文不足时硬撑、在同一会话中做下一阶段

---

### 阶段 1：架构 Spike — 接口签名 + 配置 + 目录结构

> **独立会话指令**：`阅读 SELF_CORRECTION_PLAN.md 阶段 1，完成所有任务后在 SELF_CORRECTION_PLAN.md 中确认完成`
>
> **开工前只需阅读**：项目结构、阶段 1、测试命令。不需要读其他阶段。

**目标**：铺设所有模块的骨架（接口签名 + 空实现），添加配置项，确认目录结构和 import 无报错。

**任务清单**：

- [x] 1.1 创建 `src/correction/__init__.py`
  - 导出 `SelfCorrectingPipeline`
- [x] 1.2 创建 `src/correction/types.py` — 共享数据类型
  - `@dataclass CorrectionResult`: passed (bool), flagged_claims (list[str]), layer_results (dict), confidence (float)
  - `@dataclass ClaimVerdict`: claim (str), supported (bool), evidence (str), source (str)
  - `@dataclass RetrievalQuality`: level ("GOOD"/"MARGINAL"/"WEAK"), top_score (float), avg_score (float), num_sources (int)
- [x] 1.3 创建 `src/correction/retrieval_gate.py` — Layer 0 骨架
  - `class RetrievalGate`：`__init__(rerank_threshold_good, rerank_threshold_weak)`
  - `def assess(self, results: list[RetrievalResult], reranked_results: list[RetrievalResult] | None) -> RetrievalQuality` — 空实现，返回 GOOD
- [x] 1.4 创建 `src/correction/rule_checker.py` — Layer 2 骨架
  - `class RuleChecker`：`__init__(financial_terms: list[str])`
  - `def check(self, answer: str, source_texts: list[str]) -> list[dict]` — 空实现，返回空列表
- [x] 1.5 创建 `src/correction/nli_verifier.py` — Layer 3 骨架
  - `class NLIVerifier`：`__init__()`
  - `def decompose_claims(self, answer: str) -> list[str]` — 空实现，返回空列表
  - `def verify(self, claims: list[str], context: str) -> list[ClaimVerdict]` — 空实现，返回空列表
- [x] 1.6 创建 `src/correction/external_verifier.py` — Layer 4 骨架
  - `class ExternalVerifier`：`__init__(api_key: str, base_url: str, model: str)`
  - `def verify(self, claims: list[str], context: str) -> list[ClaimVerdict]` — 空实现，返回空列表
- [x] 1.7 创建 `src/correction/pipeline.py` — 主编排器骨架
  - `class SelfCorrectingPipeline`：
    - `__init__(self, pipeline: RAGPipeline, config: SelfCorrectionConfig, ...)`
    - `def query(self, question, chat_history=None, **kwargs) -> dict` — 委托给内部 pipeline.query()，不做修正
- [x] 1.8 更新 `src/config.py` — 新增 SelfCorrectionConfig
  - `@dataclass(frozen=True) SelfCorrectionConfig`：enabled (bool), max_retries (int=2), rerank_threshold_good (float=0.7), rerank_threshold_weak (float=0.3), siliconflow_api_key (从 .env 读取)
  - 在 `Config.__init__` 中解析 `self_correction` 配置段
  - 新增 `@property self_correction -> SelfCorrectionConfig`
  - 新增 `@property siliconflow_api_key -> str | None`
- [x] 1.9 更新 `config.yaml` — 新增 self_correction 配置段
  ```yaml
  self_correction:
    enabled: false
    max_retries: 2
    rerank_threshold_good: 0.7
    rerank_threshold_weak: 0.3
    siliconflow_base_url: "https://api.siliconflow.cn/v1"
    siliconflow_model: "Qwen/Qwen3-8B"
  ```
- [x] 1.10 更新 `.env.example` — 新增 `SILICONFLOW_API_KEY`
- [x] 1.11 编写测试 `tests/test_self_correction.py` — 骨架导入测试
  - 测试所有模块可正常导入
  - 测试 Config 能加载 self_correction 配置
  - 测试 SelfCorrectingPipeline 委托 query 不报错

**验收标准**：
- `python -m pytest tests/test_self_correction.py -v` 全部通过
- `python -c "from src.correction import SelfCorrectingPipeline; print('OK')"` 无报错
- `python -c "from src.config import Config; c = Config(); print(c.self_correction)"` 打印配置
- 现有测试 `python -m pytest tests/ -v --tb=short` 全部通过（回归无破坏）

**完成确认**：

- [x] 阶段 1 全部任务完成，已通过验收标准

---

### 阶段 2：功能 Slice — Layer 0 检索质量门控 + Layer 2 规则预检

> **独立会话指令**：`阅读 SELF_CORRECTION_PLAN.md 阶段 2，完成所有任务后在 SELF_CORRECTION_PLAN.md 中确认完成`
>
> **开工前只需阅读**：项目结构、阶段 2、测试命令。

**目标**：实现两个零 API 成本的检测层（Layer 0 和 Layer 2），它们是自修正系统的第一道防线。

**前置依赖**：阶段 1 已完成。

**任务清单**：

- [x] 2.1 实现 `src/correction/retrieval_gate.py` — Layer 0 检索质量门控
  - `assess()` 方法实现：
    - 计算 top-1 score（原始检索分）
    - 计算 top-1 rerank_score（重排序分，如有）
    - 计算 avg_score（所有结果平均分）
    - 统计 num_sources
    - 按 rerank_score 阈值分级：GOOD (≥0.7) / MARGINAL (0.3-0.7) / WEAK (<0.3)
    - 当无 rerank 结果时降级使用原始检索分数
  - 注意：`results` 参数类型为 `list[RetrievalResult]`（`from src.retriever.retriever import RetrievalResult`）
- [x] 2.2 实现 `src/correction/rule_checker.py` — Layer 2 规则预检
  - 构造函数接收金融术语列表（默认从 `bm25_retriever._FINANCIAL_TERMS` 获取）
  - `check()` 方法实现 4 项检查：
    - ① `check_numbers(answer, sources)` — 提取答案中的数字（含百分比、金额），验证是否出现在原文中
    - ② `check_entities(answer, sources)` — 检查金融术语是否在原文中
    - ③ `check_dates(answer, sources)` — 提取日期（YYYY年MM月DD日等），验证是否在原文中
    - ④ `check_overlap(answer, sources)` — 计算词汇重叠率，低于阈值报警
  - 每项检查返回 `list[dict]`，每项含 type / value / severity / message
  - `check()` 汇总所有检查结果
- [x] 2.3 编写测试 `tests/test_self_correction.py`（追加）
  - **RetrievalGate 测试**：
    - test_gate_good_retrieval — 高 rerank 分数 → GOOD
    - test_gate_weak_retrieval — 低 rerank 分数 → WEAK
    - test_gate_marginal_retrieval — 中间分数 → MARGINAL
    - test_gate_no_rerank_fallback — 无 rerank 结果时用原始分
    - test_gate_empty_results — 空结果 → WEAK
  - **RuleChecker 测试**：
    - test_number_in_source — 答案数字在原文中 → 通过
    - test_number_not_in_source — 答案数字不在原文中 → 报 HIGH
    - test_entity_check — 金融术语在/不在原文中
    - test_date_check — 日期在/不在原文中
    - test_low_overlap — 词汇重叠率低 → 报 MEDIUM
    - test_clean_answer — 完全基于原文的回答 → 空列表

**验收标准**：
- `python -m pytest tests/test_self_correction.py -v` 全部通过（新增约 12 个测试）
- 现有测试全部通过（回归无破坏）

**完成确认**：

- [x] 阶段 2 全部任务完成，已通过验收标准

---

### 阶段 3：功能 Slice — Layer 3 Claim 分解 + NLI 验证

> **独立会话指令**：`阅读 SELF_CORRECTION_PLAN.md 阶段 3，完成所有任务后在 SELF_CORRECTION_PLAN.md 中确认完成`

**目标**：实现 Claim 分解和本地 NLI 验证。这是 Layer 2 和 Layer 4 之间的关键桥梁。

**前置依赖**：阶段 1、2 已完成。

**任务清单**：

- [x] 3.1 实现 `src/correction/nli_verifier.py` — `decompose_claims()` 方法
  - 用正则 + 启发式规则将回答拆分为原子断言
  - 按句号、分号分割
  - 过滤掉短句（< 10 字符）和纯疑问句
  - 过滤掉不含任何实质性内容的句子（如"根据现有资料"等套话）
  - 返回 `list[str]`
- [x] 3.2 实现 `src/correction/nli_verifier.py` — `verify()` 方法
  - 初版方案：**基于词汇重叠的轻量 NLI**（无需下载额外模型）
    - 对每条 claim，计算与原文的字符级 n-gram 重叠率
    - 重叠率 > 0.5 → SUPPORTED
    - 重叠率 < 0.3 → UNSUPPORTED
    - 0.3-0.5 → 需要进一步验证（标记为 uncertain）
  - 高级方案（可选）：加载 `vectara/hallucination_evaluation_model` 或复用本地 BGE-reranker-v2-m3
    - 在 `verify()` 中检查模型是否可用，不可用则降级为轻量方案
  - 返回 `list[ClaimVerdict]`
- [x] 3.3 编写测试 `tests/test_self_correction.py`（追加）
  - **Claim 分解测试**：
    - test_decompose_single_claim — 单句拆分
    - test_decompose_multiple_claims — 多句拆分
    - test_decompose_filters_short — 过滤短句
    - test_decompose_filters_boilerplate — 过滤套话
    - test_decompose_preserves_financial_terms — 保留金融术语完整性
  - **NLI 验证测试**：
    - test_verify_supported_claim — 有支撑的 claim → SUPPORTED
    - test_verify_unsupported_claim — 无支撑的 claim → UNSUPPORTED
    - test_verify_mixed_claims — 混合场景，部分支撑部分不支撑
    - test_verify_empty_claims — 空 claim 列表
    - test_verify_exact_quote — 完全引用原文 → SUPPORTED

**验收标准**：
- `python -m pytest tests/test_self_correction.py -v` 全部通过
- Claim 分解能正确拆分含金融术语的长句
- NLI 验证能区分支撑/不支撑的 claim
- 现有测试全部通过

**完成确认**：

- [x] 阶段 3 全部任务完成，已通过验收标准

---

### 阶段 4：功能 Slice — Layer 4 外部模型验证 + Pipeline 编排

> **独立会话指令**：`阅读 SELF_CORRECTION_PLAN.md 阶段 4，完成所有任务后在 SELF_CORRECTION_PLAN.md 中确认完成`

**目标**：实现 SiliconFlow Qwen3-8B 外部验证，并将四层检测整合到 SelfCorrectingPipeline 编排器中。

**前置依赖**：阶段 1、2、3 已完成。

**任务清单**：

- [x] 4.1 实现 `src/correction/external_verifier.py` — Layer 4 外部模型验证
  - 使用 `httpx`（项目已有依赖）调用 SiliconFlow API
  - API 兼容 OpenAI 格式：`POST {base_url}/chat/completions`
  - Prompt 设计（关键：与生成 prompt 完全不同的任务框架）：
    ```
    你是金融文档事实核查审计员。逐条判断以下断言是否被证据文本直接支撑。
    严格标准：仅当证据包含直接支持断言的具体信息时判定为 YES。

    证据：
    {context}

    待判定断言：
    {claims_formatted}

    逐条输出 JSON 数组，每项含 claim, supported(true/false), evidence_quote(引用原文或"UNSUPPORTED")
    ```
  - `temperature=0`，`max_tokens=1024`
  - 错误处理：API 调用失败时降级为全部 SUPPORTED（不阻塞正常流程）
  - 返回 `list[ClaimVerdict]`
- [x] 4.2 实现 `src/correction/pipeline.py` — SelfCorrectingPipeline 完整编排
  - `query()` 方法完整流程：
    1. 调用 `self._pipeline.query()` 获取原始结果
    2. Layer 0：评估检索质量，WEAK 时直接返回原始结果 + 警告
    3. Layer 2：规则预检，无问题直接返回
    4. Layer 3：Claim 分解 + NLI，全部支撑直接返回
    5. Layer 4：外部模型验证有疑点的 claim
    6. 如有不支撑的 claim → 构造反馈 → 重试生成（最多 max_retries 次）
    7. 返回最终结果（含 correction_metadata）
  - `stream_query()` 方法：**禁用自修正**，直接委托给内部 pipeline（流式无法中途修正）
  - `aquery()` 方法：同 `query()` 的异步版本（如时间不够可留空实现）
  - 返回值在原始 `{"answer": str, "sources": list}` 基础上增加 `correction` 字段
- [x] 4.3 编写测试 `tests/test_self_correction.py`（追加）
  - **ExternalVerifier 测试**（Mock httpx 调用）：
    - test_verify_supported — API 返回全部 YES → 全部 SUPPORTED
    - test_verify_unsupported — API 返回 NO → UNSUPPORTED
    - test_verify_api_failure — API 调用失败 → 降级为全部 SUPPORTED
    - test_verify_mixed_results — 部分支撑部分不支撑
  - **SelfCorrectingPipeline 测试**（Mock 内部 pipeline + 所有 layer）：
    - test_query_passes_through — 无问题时直接返回
    - test_query_weak_retrieval — 检索弱时返回警告
    - test_query_rule_check_catches — 规则预检发现问题时触发后续层
    - test_query_correction_retry — 检测到幻觉后重试
    - test_query_max_retries_exhausted — 重试耗尽后返回最后结果
    - test_stream_query_delegates — 流式查询直接委托不做修正

**验收标准**：
- `python -m pytest tests/test_self_correction.py -v` 全部通过（累计约 30+ 测试）
- SelfCorrectingPipeline 的 query() 能正确编排四层检测
- API 调用失败时优雅降级，不阻塞正常流程
- 现有测试全部通过

**完成确认**：

- [x] 阶段 4 全部任务完成，已通过验收标准

---

### 阶段 5：集成 + UI 开关 + 回归测试

> **独立会话指令**：`阅读 SELF_CORRECTION_PLAN.md 阶段 5，完成所有任务后在 SELF_CORRECTION_PLAN.md 中确认完成`

**目标**：将 SelfCorrectingPipeline 集成到 Streamlit UI 和 FastAPI，添加开关和展示。

**前置依赖**：阶段 1、2、3、4 已完成。

**任务清单**：

- [x] 5.1 更新 `src/ui/services.py` — 可选集成 SelfCorrectingPipeline
  - 新增 `_init_self_correction(pipeline, api_key)` 函数
  - 当 `st.session_state.self_correction_enabled` 为 True 时，用 SelfCorrectingPipeline 包装 RAGPipeline
  - 当为 False 时，直接使用原始 RAGPipeline（向后兼容）
- [x] 5.2 更新 `src/ui/sidebar.py` — 添加自修正开关
  - 在侧边栏添加 `st.checkbox("自我修正", key="self_correction_enabled")`
  - 显示当前自修正状态
- [x] 5.3 更新 `src/ui/chat_tab.py` — 展示修正信息
  - 如果返回结果包含 `correction` 字段，展示修正详情
  - 显示被标记的 claim 和修正过程
- [x] 5.4 更新 FastAPI `src/api/routes/query.py` — API 层支持
  - 在 QueryRequest 中添加 `self_correction: bool = False` 字段
  - 当请求启用自修正时，用 SelfCorrectingPipeline 包装
- [x] 5.5 全量回归测试
  - `python -m pytest tests/ -v --tb=short` — 所有测试通过
  - 确认关闭自修正时系统行为与修改前完全一致
  - 确认开启自修正时不影响流式查询

**验收标准**：
- `python -m pytest tests/ -v --tb=short` 全部通过（175+ 现有 + 30+ 新增）
- Streamlit UI 侧边栏有自修正开关
- 开启自修正后，答案区能展示修正信息
- 关闭自修正后，系统行为与修改前完全一致
- `streamlit run app.py` 正常启动

**完成确认**：

- [x] 阶段 5 全部任务完成，已通过验收标准

---

### 阶段 6：全局 Review

> **独立会话指令**：`阅读 SELF_CORRECTION_PLAN.md，对整个自我修正系统进行全面的 Code Review`

**目标**：对新增的自我修正系统进行全局审查，确保代码质量、架构合理性、安全性。

**前置依赖**：阶段 1-5 全部完成。

**任务清单**：

- [x] 6.1 代码质量审查
  - ✅ 新增模块遵循原有代码风格（`@dataclass`、`logging.getLogger(__name__)`、错误分类）
  - ✅ 类型注解完整（所有公开方法均有参数和返回值注解，`from __future__ import annotations` 统一使用）
  - ✅ 命名一致性（PascalCase 类：RetrievalGate/RuleChecker/NLIVerifier/ExternalVerifier/SelfCorrectingPipeline；snake_case 方法：assess/check/decompose_claims/verify）
  - ✅ 无 `as any`、`@ts-ignore`、空 catch 块 — **已修复**: 将 2 处 `# type: ignore[assignment]` 替换为 `cast()` 显式类型转换
  - ℹ️ 备注: `_check_overlap` 使用字符级集合（`set(answer)`）而非词级，变量名 `answer_words` 略有误导，但作为 Layer 2 粗筛可接受，Layer 3 NLI 会精确兜底
- [x] 6.2 架构审查
  - ✅ SelfCorrectingPipeline 真正零侵入 — Wrapper 模式，关闭时直接使用 RAGPipeline（services.py:126-127），行为完全一致
  - ✅ 四层检测接口清晰：每层有独立类 + 明确的输入/输出类型（`RetrievalQuality`/`list[dict]`/`list[ClaimVerdict]`）
  - ℹ️ 备注: 四层不支持独立开关（配置级），但架构上通过条件判断实现了等效的层级跳过（Layer 2 无问题跳过 Layer 3/4）
  - ✅ 降级策略完善：ExternalVerifier API 失败→全部 SUPPORTED（不阻塞流程）；JSON 解析失败→全部 SUPPORTED；RuleChecker 导入失败→空术语列表
  - ✅ 新增配置合理：`SelfCorrectionConfig` 为 frozen dataclass，`config.yaml` 有合理默认值，`self_correction.enabled: false` 默认关闭
  - ℹ️ 备注: `aquery()` 委托给内部 Pipeline 不做修正（计划中标注"如时间不够可留空实现"），`stream_query()` 同理（流式无法中途修正）
- [x] 6.3 安全性审查
  - ✅ SILICONFLOW_API_KEY 安全存储：`.env` 文件不入 Git（`.gitignore` 第 17 行），`.env.example` 有占位提示
  - ✅ 外部 API 调用有 30 秒超时（external_verifier.py:79 `timeout=30.0`），失败时优雅降级不阻塞
  - ✅ 用户输入有长度限制：RAGPipeline 已在入口校验（空输入/超 4096 字符），SelfCorrectingPipeline 委托给 RAGPipeline 后才处理
  - ℹ️ 备注: 传给 SiliconFlow 的 context 包含检索到的文档内容，属设计需要（需断言+原文比对），但应在文档中说明数据会发送至第三方
- [x] 6.4 完整性检查
  - ✅ 阶段 1-5 全部标记为 `[x]` 完成
  - ✅ 全量测试通过：210 passed（含 57 项自修正测试）
  - ✅ 新增文件与测试对应关系：
    - types.py → TestDataclassCreation (3 tests)
    - retrieval_gate.py → TestRetrievalGate (8 tests)
    - rule_checker.py → TestRuleChecker (10 tests)
    - nli_verifier.py → TestClaimDecomposition (5) + TestNLIVerification (5)
    - external_verifier.py → TestExternalVerifier (7 tests)
    - pipeline.py → TestSelfCorrectingPipeline (3) + TestSelfCorrectingPipelineOrchestration (7)
    - config.py (SelfCorrectionConfig) → TestSelfCorrectionConfig (2)
    - 集成点 (UI/API) → 已在各自模块中验证

**完成确认**：

- [x] 阶段 6 全部任务完成，项目 Review 通过

---

## 测试命令

- 全量测试：`python -m pytest tests/ -v --tb=short`
- 自修正测试：`python -m pytest tests/test_self_correction.py -v --tb=short`
- 启动验证：`streamlit run app.py`
- 配置验证：`python -c "from src.config import Config; c = Config(); print(c.self_correction)"`

## 快速开始（自修正相关）

```bash
# 1. 配置 SiliconFlow API Key
# 注册 https://siliconflow.cn 获取免费 API Key
# 编辑 .env，添加：
SILICONFLOW_API_KEY=your_siliconflow_key_here

# 2. 启动应用
streamlit run app.py

# 3. 在侧边栏开启"自我修正"开关
```

## 常见问题 FAQ

### Q: Layer 3 NLI 模型需要下载吗？
初版使用基于词汇重叠的轻量方案，无需下载模型。如需更高精度，可选加载 `vectara/hallucination_evaluation_model`（约 300MB）。

### Q: SiliconFlow 免费额度用完怎么办？
Qwen3-8B 在 SiliconFlow 上永久免费。如需备选，可切换到 DeepSeek V3（注册送 500 万 tokens）。

### Q: 自修正会增加多少延迟？
Layer 0-3 几乎零延迟（纯代码 + 本地计算）。Layer 4 每次约 1-3 秒（API 调用）。仅在 Layer 2-3 检测到疑点时才触发 Layer 4。

### Q: 关闭自修正后会影响现有功能吗？
不会。SelfCorrectingPipeline 是 Wrapper 模式，关闭后直接使用原始 RAGPipeline，行为完全一致。

## License

MIT
