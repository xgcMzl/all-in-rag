# RAG 评估指标笔记：为什么「答案相关性」不需要参考答案？

> 配套代码：`01_llamaindex_evaluation_example.py`
> LlamaIndex 版本：`llama-index-core 0.12.52`
> 整理日期：2026-11-03

## 目录

1. [先分清三个词](#一先分清三个词)
2. [用项目数据集里的真实题目举例](#二用项目数据集里的真实题目举例)
3. [切题 ≠ 正确](#三切题--正确)
4. [为什么没有参考答案，这个指标也能成立？](#四为什么没有参考答案这个指标也能成立)
5. [完整的数据流](#五完整的数据流)
6. [指标分类总表](#六指标分类总表)
7. [DatasetGenerator 为什么还要生成参考答案？](#七那-datasetgenerator-为什么还要生成参考答案)
8. [对应代码位置速查](#八对应代码位置速查)
9. [**为什么 RAG 三元组不包含答案的准确性？**](#九为什么-rag-三元组不包含答案的准确性)
10. [本项目的局限性与使用建议](#十本项目的局限性与使用建议)
11. [附：想加「正确性」评估时的接法](#附想加正确性评估时的接法)

---

## 核心结论

> **「答案相关性」这个名字里的「答案」，指的是模型答出来的那个 `response`，不是数据集里印着的标准答案 `reference`。**

判断模型「答得**切不切题**」→ 不用参考答案；
判断模型「答得**对不对**」→ 必须用参考答案。

> **「RAG 三元组」是诊断工具，不是评分工具。** 它回答「哪里坏了」；正确性回答「到底对不对」，所以留给离线验收。

---

## 一、先分清三个词

名字撞车是理解不了这个问题的根源。

| 名字 | 英文 | 是什么 | 谁产生的 |
|---|---|---|---|
| 问题 | `query` | 用户提的问题 | 数据集 |
| **回答** | **`response`** | 模型生成的回答 | **你的 RAG 系统** |
| **参考答案** | **`reference`** | 标准答案 | 数据集（人写的或自动生成） |

`AnswerRelevancyEvaluator` 里的 **Answer 指的是「回答」(`response`)**，不是「参考答案」(`reference`)。

中文把它译成「答案相关性」很容易误导；它真正想表达的是「**回答切题度**」——这段回答是不是在回应这个问题。

---

## 二、用项目数据集里的真实题目举例

数据集 `c6_response_eval_dataset.json` 中有一道题：

> **query**: `Who are the two Coordinating Lead Authors for this chapter?`

假设模型给出三种回答，看 `AnswerRelevancyEvaluator` 会怎么判：

| 模型的回答 (`response`) | 裁判判断 | 得分 | 是否需要参考答案？ |
|---|---|---|---|
| "The two Coordinating Lead Authors are Camille Parmesan and Hans-Otto Pörtner." | 切中题目、正面回答了「是谁」 | **2/2 = 1.0** | ❌ 不需要 |
| "This chapter is part of the IPCC Sixth Assessment Report." | 话题相关，但没回答「是谁」 | **1/2 = 0.5** | ❌ 不需要 |
| "Python is a popular programming language." | 完全离题 | **0/2 = 0.0** | ❌ 不需要 |

> 表中分数是**示意值**，用于说明打分逻辑，**不是本项目的实测结果**。

**关键点：上面三个判断，裁判根本不需要知道真实作者是谁。**
它只需要判断「这段文字像不像在回答这个问题」。

### 打分规则（源码 `answer_relevancy.py`）

裁判被问两个子问题，每问 1 分，满分 2 分：

1. Does the provided response match the subject matter of the user's query?
2. Does the provided response attempt to address the focus or perspective taken on by the user's query?

模型必须按 `[RESULT] 2` 的格式输出，最终 `score = 原始分 / score_threshold(默认 2.0)`，归一化到 **0 ~ 1**。

### 源码里最关键的一行

```python
async def aevaluate(self, query=None, response=None, contexts=None, ...):
    del kwargs
    del contexts        # ← 连检索到的上下文都丢弃
    ...
    eval_response = await self._llm.apredict(
        prompt=self._eval_template,
        query=query,        # ← 只喂 query
        response=response,  # ← 和 response
    )
```

既不用 `contexts`，也没有 `reference` 参数。**它只认识问题和回答这两个东西。**

---

## 三、切题 ≠ 正确

这是最容易混淆、也最重要的区别。

假设模型回答：

> "The authors are **Michael Jackson and Albert Einstein**."

| 指标 | 结果 | 原因 |
|---|---|---|
| `AnswerRelevancy` | **2/2，满分** | 它确实在正面回答「是谁」 |
| `Correctness` | **0 分** | 人名全错，必须对照 `reference` 才能发现 |

> 表中分数是**示意值**，用于说明指标差异，**不是实测结果**。

**这就是两者的本质区别：切题 ≠ 正确。**

所以 `AnswerRelevancy` 天生防不了幻觉——一个自信但错误的回答，在这个指标上照样满分。防幻觉是 `FaithfulnessEvaluator` 的职责。

---

## 四、为什么没有参考答案，这个指标也能成立？

因为「答非所问」是一种**形式上**就能看出来的缺陷，跟内容对不对无关。你能一眼看出来的东西，LLM 裁判也能。

但举例时必须让「切题度」这一维单独变化，其余保持不动，否则看不出差异。用**同一道题**的三档回答来对比：

> **query**：法国的首都是哪？

| # | 模型的回答 | 切题吗？ | 答对了吗？ | AnswerRelevancy | Correctness |
|---|---|---|---|---|---|
| 1 | 日本是一个位于东亚的岛国，由四个大岛和众多小岛组成。 | ❌ 完全没提首都，也没回应问题 | — | **0 分** | — |
| 2 | 巴黎是法国最大的城市，也是著名的旅游胜地。 | ⚠️ 相关，但没直接回答「首都是谁」 | — | **1 分** | — |
| 3 | 法国的首都是马赛。 | ✅ 正面回答，句式完全契合 | ❌ 事实错误 | **2 分（满分）** | **0 分** |

> 表中分数是**示意值**（按评分规则推演），**不是实测结果**；LLM 裁判的实际给分会有波动。

看第 3 行：**回答越切题，越说明不了它是对的。**

`AnswerRelevancyEvaluator` 只问两件事：

1. 这段回答是不是在讲用户问的那个话题？
2. 这段回答是否试图回应问题所关注的那个点？

「法国的首都是马赛」对这两问都是 YES——它讲的是法国、回答的是首都、句式严丝合缝。**但它没有能力核对「马赛」是不是真的首都**，那是 `CorrectnessEvaluator` 对照 `reference` 才做得的事。

> ⚠️ 一个反面教材（本文档初版写错了）：像「问法国的首都，答日本的首都是东京」这种句子，
> 既像在回答「首都」这个话题，又完全答错了对象，落在灰色地带，LLM 裁判给几分并不稳定。
> **好的例子里，「切题」和「正确」必须是两个清晰分离的维度**，否则例子本身就说明不了问题。

---

## 五、完整的数据流

下图只画四个代表性指标，重点在区分「要不要 reference」；六项完整对照见第六节。

```text
                    ┌─→ contexts ──┬─→ ContextRelevancy  (query + contexts)
                    │              │
query ─→ 检索器 ────┘              │
   │                               └─→ Faithfulness      (response + contexts)   ← query 不参与判定
   │
   └─→ 生成 ─→ response ─┬─→ AnswerRelevancy  (query + response)            ← 不碰 reference
                         │
                         └─→ Correctness      (query + response + reference) ← 唯一需要 reference 的
```

> 注：`FaithfulnessEvaluator` 虽然形参里有 `query`，但**判定时并不使用它**——
> 源码中是 `query_engine.aquery(response)`，即只拿 response 去和 contexts 比对。
> `query` 仅被回填进 `EvaluationResult.query` 供追溯。

`response` 是**你的系统现场生成的**，每道题都不一样；
`reference` 是**数据集里写死的**。

评估「`response` 好不好」不一定要拿 `reference` 来比——可以先从几个不需要标准答案的角度看。

---

## 六、指标分类总表

| 类型 | 需要参考答案？ | LlamaIndex 中的类 | 输出字段 | 用来干什么 | 对应三元组环节 |
|---|---|---|---|---|---|
| 无参考（reference-free） | ❌ | `FaithfulnessEvaluator` | `passing` + `score`(0/1) | **查幻觉**：整段回答是否被检索内容支撑。口径较宽松——只要上下文中有任意内容支持回答即判 YES（提示词原文：`Answer YES if any of the context supports the information, even if most of the context is unrelated`），不会逐句核查。 | **Groundedness**（扎根性） |
| 无参考（reference-free） | ❌ | `RelevancyEvaluator` | `passing` + `score`(0/1) | **查串味**：回答是否与检索到的上下文一致。注意名字有误导——它对比的是「回答 vs 上下文」，不评价「问题有没有被回答」，和忠实度高度重叠。 | **Groundedness**（扎根性的弱化版：只给 YES/NO；本笔记按提示词语义归类，非官方映射） |
| 无参考（reference-free） | ❌ | `AnswerRelevancyEvaluator` | `score` (0~1) | **查答非所问**：回答是否切中问题本身。完全不看上下文、也不判断对错，只判断"这段文字像不像在回答这个问题"。 | **Answer Relevance**（答案相关性） |
| 无参考（reference-free） | ❌ | `ContextRelevancyEvaluator` | `score` (0~1) | **查检索质量**：检索回来的片段是否切题、是否足以回答问题。检索坏了，后面模型再强也救不回来，所以这项要单独盯。 | **Context Relevance**（上下文相关性） |
| **有参考（reference-based）** | ✅ | **`CorrectnessEvaluator`** | `score` **（1~5 原始分）** + `passing` | **查对不对**：把回答与参考答案对比打分，用于版本验收。⚠️ 源码**不做归一化**：`score` 就是 1~5 的原始分，`passing = score >= 4.0`，统计时需自行除以 5。 | |
| **有参考（reference-based）** | ✅ | **`SemanticSimilarityEvaluator`** | `score` (0~1) + `passing` | **查像不像**：用向量算回答与参考答案的语义相似度（默认阈值 0.8）。便宜、快，适合大批量初筛；但只测"意思接不接近"，推理错了但措辞接近照样高分。 | |

> **空白**表示该指标不对应 RAG 三元组的任何环节。
> 三元组只问三件事——「检索的切题吗」「回答有依据吗」「回答切题吗」，**三者全部不涉及标准答案**；
> 而 `Correctness` 和 `SemanticSimilarity` 的本质就是「拿回答去和参考答案比」，所以它们天然落在三元组之外（见第 9 节）。

**三元组与指标的一一对应关系：**

| 三元组环节 | 回答的问题 | 对应的 LlamaIndex 指标 |
|---|---|---|
| Context Relevance | 检索回来的片段切题吗？ | `ContextRelevancyEvaluator` |
| Groundedness | 回答有检索内容作依据吗？ | `FaithfulnessEvaluator`；`RelevancyEvaluator`（弱化版） |
| Answer Relevance | 回答切中问题吗？ | `AnswerRelevancyEvaluator` |
| —（三元组没有） | 回答到底对不对？ | `CorrectnessEvaluator`、`SemanticSimilarityEvaluator` |

> 注意：`passing` 型指标算**通过率**，`score` 型指标算**平均分**。
> `AnswerRelevancyEvaluator` 和 `ContextRelevancyEvaluator` **都不设置 `passing`**（值为 `None`），
> 直接套用通过率统计会抛 `TypeError`，必须用平均分函数。

### 怎么组合用

- **只查"哪里坏了"**（线上监控）→ 挑三个无参考指标就够：`ContextRelevancy`（检索）+ `Faithfulness`（生成）+ `AnswerRelevancy`（端到端），正好是第 9 节讲的 RAG 三元组。
- **要确认"到底对不对"**（发版验收）→ 必须有 golden set，加 `Correctness` 兜底。
- **只想廉价大批量筛**→ `SemanticSimilarity` 先过一遍，可疑的再人工或 LLM 复核。

> 一句话记法：**无参考指标查"过程有没有跑偏"，有参考指标查"结果有没有跑错"。**

---

## 七、那 DatasetGenerator 为什么还要生成参考答案？

因为 `agenerate_dataset_from_nodes()` 内部调用时传了 `generate_response=True`，它顺手把参考答案也生成了，存进 `QueryResponseDataset.responses` 字段。

**这份数据是有备无患的**：你以后想加 `CorrectnessEvaluator` 就直接能用。
但当前脚本用的三元组三个指标（上下文相关性、忠实度、答案相关性）确实一个都用不上它。

```python
# 问题和参考答案都是 Dict[str, str]（"题目ID" -> 文本）
# 按下标（顺序）对齐不可靠；按「题目ID」配对才安全
pairs = [
    (qid, q, response_eval_dataset.responses.get(qid, ""))
    for qid, q in response_eval_dataset.queries.items()
]
queries = [q for _, q, _ in pairs]
references = [ref for _, _, ref in pairs]
```

> ⚠️ 坑点一：`QueryResponseDataset.queries` 是 `Dict[str, str]`（"题目ID" -> "问题文本"），不是 `list`。
> `BatchEvalRunner.aevaluate_queries` 内部是 `for query in queries`，
> 直接传 dict 遍历到的是 **key（题目ID）**，会拿 UUID 字符串去检索提问，分数全是垃圾。
>
> ⚠️ 坑点二：两个 dict 的 `.values()` 顺序在当前文件里恰好一致，但这是**数据的巧合，不是 API 保证**。
> 换数据集或增删条目后，按顺序配对可能错位，务必按 key 配对。
>
> ⚠️ 坑点三：`DatasetGenerator` 偶尔会把整段生成输出当成一道题存进来
> （形如 `Based on the provided context..., here are N questions...`）。
> 这类脏数据会同时拉低三个指标，评估前要过滤掉（当前脚本的 `is_valid_query()` 负责这件事）。

---

## 八、对应代码位置速查

`01_llamaindex_evaluation_example.py`：

| 行号 | 内容 |
|---|---|
| 213 | `evaluators = {...}` 三元组三个评估器的注册 |
| 229 | `is_valid_query()` 脏数据过滤规则 |
| 240–248 | 按「题目ID」配对、过滤、拆出 `queries` / `references` |
| 289 | `calc_pass_rate()` 统计 passing 型指标 |
| 300 | `calc_avg_score()` 统计 score 型指标 |
| 314–321 | 两个方案的三项分数计算 |
| 338 | 三维度对比 |

> 行号以修订时的脚本版本为准，改代码后会漂移，建议用搜索而不是行号定位。

---

## 九、为什么 RAG 三元组不包含答案的准确性？

先明确一个前提：**「RAG 三元组」（RAG Triad）不是「RAG 评估的全部」，而是一套「在线诊断工具」**。它由 TruLens 提出，包含：

1. **Context Relevance**（上下文相关性）—— 检索回来的片段切题吗？
2. **Groundedness**（扎根性 / 忠实度）—— 回答有检索内容作依据吗？
3. **Answer Relevance**（答案相关性）—— 回答切题吗？

没有「答案准确性」，不是疏忽，是设计取舍。三个原因：

### 9.1 三元组是「归因工具」，correctness 是「结论」

看每个三元组指标对应的**可修复环节**：

| 指标偏低 | 能定位到 | 该修什么 |
|---|---|---|
| Context Relevance 低 | **检索器 / 切分策略 / embedding** | 换 chunk 大小、换模型、加混合检索 |
| Groundedness 低 | **生成环节** | 收紧提示词、降低 temperature、要求引用原文 |
| Answer Relevance 低 | **提示词 / 生成** | 改模板、加 few-shot |
| **Correctness 低** | **❓ 定位不到任何具体环节** | 不知道修哪 |

**Correctness 是个症状，不是一个可归因的环节。** 分数低了你只知道「结果烂」，但不知道是检索错了、还是模型在编、还是答非所问——这时候你还是得回头去看三元组。

诊断指标的黄金标准是：**每个指标都要能指向一个可动手修的组件。** correctness 做不到，所以它进不了诊断集。

### 9.2 三元组可以「零标注、在线跑」，correctness 做不到

| | 三元组 | Correctness |
|---|---|---|
| 需要参考答案 | ❌ 不需要 | ✅ 需要 |
| 数据从哪来 | 模型自己的中间产物（query / contexts / response） | 人工标注或高质量 golden set |
| 每条 query 都能算 | ✅ | ❌ |
| 能否生产环境实时监控 | ✅ | ❌ |

现实里企业 RAG 跑的几乎都是**私有文档**，没有任何现成的 QA 对照。要算 correctness，就得先人工造一批标准答案——成本高、覆盖窄、还跟不上文档更新。

更要命的是：**参考答案本身可能是错的**。本项目里的 `c6_response_eval_dataset.json` 就是 `DatasetGenerator` 让 LLM 自动生成的参考答案，它出错时，你是在「拿一个可能错的答案去判另一个答案」——错误会叠加，而且看不出来。

### 9.3 原创业务里大量问题根本没有唯一答案

```text
"帮我总结这份报告第三部分"
"这段代码有什么问题"
"根据合同，我应该注意哪些风险"
```

这类问题的正确答案是**一个集合**，不是一条字符串。用 correctness 去比，会因为措辞不同而判错——语义对了、字面不同，照样低分。三元组不跟标准答案比，所以不受这个限制。

### 9.4 但它确实有盲区，必须承认

三元组全绿，答案仍可能是错的。两种典型情况：

1. **检索到的原文本身就是错的**（文档过期、上游数据有误）→ Context Relevance 高、Groundedness 高，但答案错。
2. **需要多跳推理**：答案不在任何一个片段里，要靠组合推理。模型老实地说"根据资料推断…"→ 忠实度高，但结论可能错。

所以严格的评估体系会补上 correctness。对比一下：

| 框架 | 定位 | 是否含 correctness |
|---|---|---|
| **TruLens RAG Triad** | 在线监控 / 归因诊断 | ❌（定义见 TruLens 官方文档） |
| **RAGAS** | 离线评测套件 | ✅ `answer_correctness`、`answer_semantic_similarity`（指标名**需核实**：本机未安装 RAGAS，无法本地验证） |
| **LlamaIndex** | 两者都提供 | ✅ `CorrectnessEvaluator`、`SemanticSimilarityEvaluator`（已对照 `0.12.52` 源码核验） |

> 外部框架信息来自公开文档，未在本地环境安装验证：
> - TruLens RAG Triad：<https://www.trulens.org/getting_started/core_concepts/rag_triad/>
> - RAGAS 指标：<https://docs.ragas.io/>（`answer_correctness` 的确切名称与打分口径请以对应版本文档为准）

RAGAS 保留 correctness，是因为它面向**离线回归测试**——有 golden set、跑一次要几分钟、用来做版本对比。
TruLens 的三元组面向**线上持续监控**——每条请求都要立刻打分。

### 9.5 实践中的组合方式

```text
线上（每条请求实时跑）  → 三元组，异常就告警，人工介入
    ↓ 发现问题
离线（发版前跑）        → 三元组定位 + correctness 验收
```

**一句话总结：**

> **三元组回答的是「哪里坏了」，correctness 回答的是「到底对不对」。**
> 前者能指导你改代码，后者不能——所以诊断集里只放前者，把后者留给离线验收。

---

## 十、本项目的局限性与使用建议

### 10.1 数据质量

- `DatasetGenerator` 自动生成的数据集可能残留未解析干净的条目（整段生成输出被当成一道题）。当前脚本的 `is_valid_query()` 已做过滤。
- 自动生成的参考答案本身也可能有误——这正是不能无条件依赖 `CorrectnessEvaluator` 的原因之一。
- 实测：当前 `c6_response_eval_dataset.json` 共 13 条，过滤后 **12 条有效**。

### 10.2 对比实验存在混杂变量

两种方案的上下文注入量并不相同：

| 方案 | 单个 Node 体量 | top_k |
|---|---|---|
| 句子窗口 | 目标句 + 前后各 3 句 | 2 |
| 常规分块 | `chunk_size=512` token | 2 |

而 Context Relevance 与 Faithfulness **都对上下文长度敏感**——文本越多越容易覆盖问题，也越容易混入无关内容。因此两项分数的差异**不能全部归因于"检索粒度更优"**。

**建议**：记录两种方案每次实际注入的 token 数；或固定总 token 预算后再比较。

### 10.3 样本量偏小

当前有效样本 **n=12**，比例类指标（通过率、平均分）的置信区间很宽，几道题的差距可能只是噪声。输出中的"领先/落后"应理解为趋势，而非统计意义上的显著差异。

**建议**：至少 50 条以上再讨论"优于"；或对同一批题重复评估取均值。

### 10.4 评分稳定性

LLM 裁判不是确定性的，相同输入重复评估分数也可能不同（即使 `temperature=0.1`）。两个方案的对比应在**同一批次**内完成，避免分批跑引入系统性偏差。

### 10.5 结论措辞

本笔记与脚本统一使用"高于 / 领先"，不使用"显著优于"——当前样本量不支持显著性判断。

---

## 附：想加「正确性」评估时的接法

```python
from llama_index.core.evaluation import CorrectnessEvaluator

correctness_evaluator = CorrectnessEvaluator(llm=Settings.llm)

# 在现有三元组基础上追加 correctness（key 名与脚本中 evaluators 的写法保持一致）
evaluators = {
    "context_relevancy": context_relevancy_evaluator,
    "faithfulness": faithfulness_evaluator,
    "answer_relevancy": answer_relevancy_evaluator,
    "correctness": correctness_evaluator,
}

# 多个评估器时：{评估器名: {参数名: [逐条的值]}}
# BatchEvalRunner 检测到值是 dict 就进入 multi_kwargs 分支，
# 按评估器名分派参数，没被点名的评估器拿到空 kwargs。
results = await runner.aevaluate_queries(
    queries=queries,
    query_engine=query_engine,
    correctness={"reference": references},
)
```

⚠️ **`CorrectnessEvaluator` 的分数不是 0~1**。源码（`correctness.py`）直接返回裁判给的原始分，
`score` 范围是 **1~5**，`passing` 判定为 `score >= 4.0`，全程没有除以阈值。
所以不能直接丢给 `calc_avg_score` 再用 `:.1%` 打印（会得到 450% 这种结果），需要自己归一化：

```python
raw_scores = [r.score for r in results["correctness"] if r.score is not None]
avg_1_to_5 = sum(raw_scores) / len(raw_scores)
print(f"正确性: {avg_1_to_5 / 5.0:.1%}")   # 除以 5 换算成百分比
```

> 另：`CorrectnessEvaluator` 的 `reference` 允许缺省，缺省时提示词会填
> `(NO REFERENCE ANSWER SUPPLIED)`，但此时它退化成"相关性 + 大致正确性"的混合判断，
> 不建议作为验收依据。
