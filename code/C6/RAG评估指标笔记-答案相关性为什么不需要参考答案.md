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
10. [附：想加「正确性」评估时的接法](#附想加正确性评估时的接法)

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

四个指标各自吃不同的输入，看清楚谁用谁：

```text
                    ┌─→ contexts ──┬─→ ContextRelevancy  (query + contexts)
                    │              │
query ─→ 检索器 ────┘              │
   │                               └─→ Faithfulness      (query + response + contexts)
   │
   └─→ 生成 ─→ response ─┬─→ AnswerRelevancy  (query + response)            ← 不碰 reference
                         │
                         └─→ Correctness      (query + response + reference) ← 唯一需要 reference 的
```

`response` 是**你的系统现场生成的**，每道题都不一样；
`reference` 是**数据集里写死的**。

评估「`response` 好不好」不一定要拿 `reference` 来比——可以先从几个不需要标准答案的角度看。

---

## 六、指标分类总表

| 类型 | 需要参考答案？ | LlamaIndex 中的类 | 输出字段 | 用来干什么 |
|---|---|---|---|---|
| 无参考（reference-free） | ❌ | `FaithfulnessEvaluator` | `passing` (True/False) | **查幻觉**：回答的每句话是否都能在检索到的原文里找到依据。回答"听着很对但文档里没说"时，这里会挂。 |
| 无参考（reference-free） | ❌ | `RelevancyEvaluator` | `passing` (YES/NO) | **查串味**：回答是否与检索到的上下文一致。注意名字有误导——它对比的是「回答 vs 上下文」，不评价「问题有没有被回答」，和忠实度高度重叠。 |
| 无参考（reference-free） | ❌ | `AnswerRelevancyEvaluator` | `score` (0~1) | **查答非所问**：回答是否切中问题本身。完全不看上下文、也不判断对错，只判断"这段文字像不像在回答这个问题"。 |
| 无参考（reference-free） | ❌ | `ContextRelevancyEvaluator` | `score` (0~1) | **查检索质量**：检索回来的片段是否切题、是否足以回答问题。检索坏了，后面模型再强也救不回来，所以这项要单独盯。 |
| **有参考（reference-based）** | ✅ | **`CorrectnessEvaluator`** | `score` (0~1) | **查对不对**：把回答与参考答案对比打分。用于版本验收——上线前确认"答案确实是对的"，适合离线回归测试。 |
| **有参考（reference-based）** | ✅ | **`SemanticSimilarityEvaluator`** | `score` + `passing` | **查像不像**：用向量算回答与参考答案的语义相似度。便宜、快，适合大批量初筛；但只测"意思接不接近"，推理错了但措辞接近照样高分。 |

> 注意：`passing` 型指标算**通过率**，`score` 型指标算**平均分**。
> `AnswerRelevancyEvaluator` 不设置 `passing`（值为 `None`），直接混用会抛 `TypeError`。

### 怎么组合用

- **只查"哪里坏了"**（线上监控）→ 挑三个无参考指标就够：`ContextRelevancy`（检索）+ `Faithfulness`（生成）+ `AnswerRelevancy`（端到端），正好是第 9 节讲的 RAG 三元组。
- **要确认"到底对不对"**（发版验收）→ 必须有 golden set，加 `Correctness` 兜底。
- **只想廉价大批量筛**→ `SemanticSimilarity` 先过一遍，可疑的再人工或 LLM 复核。

> 一句话记法：**无参考指标查"过程有没有跑偏"，有参考指标查"结果有没有跑错"。**

---

## 七、那 DatasetGenerator 为什么还要生成参考答案？

因为 `agenerate_dataset_from_nodes()` 内部调用时传了 `generate_response=True`，它顺手把参考答案也生成了，存进 `QueryResponseDataset.responses` 字段。

**这份数据是有备无患的**：你以后想加 `CorrectnessEvaluator` 就直接能用。
但当前用的三个指标（忠实度、一致性、答案相关性）确实一个都用不上它。

```python
# 问题：Dict[str, str]  → 必须 .values() 取文本
queries = list(response_eval_dataset.queries.values())

# 参考答案：同样是 Dict，.values() 顺序与 queries 一致，可按下标对齐
references = list(response_eval_dataset.responses.values())
```

> ⚠️ 坑点：`QueryResponseDataset.queries` 是 `Dict[str, str]`（"题目ID" -> "问题文本"），不是 `list`。
> `BatchEvalRunner.aevaluate_queries` 内部是 `for query in queries`，
> 直接传 dict 遍历到的是 **key（题目ID）**，会拿 UUID 字符串去检索提问，分数全是垃圾。

---

## 八、对应代码位置速查

`01_llamaindex_evaluation_example.py`：

| 行号 | 内容 |
|---|---|
| 206 | `evaluators = {...}` 三个评估器的注册 |
| 218 | `queries = list(...queries.values())` 取问题文本 |
| 223 | `references = list(...responses.values())` 取参考答案（当前未使用，为 `Correctness` 预留） |
| 262 | `calc_pass_rate()` 统计 passing 型指标 |
| 273 | `calc_avg_score()` 统计 score 型指标 |
| 287–294 | 两个方案的三项分数计算 |
| 314 | 三维度对比 |

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
| **TruLens RAG Triad** | 在线监控 / 归因诊断 | ❌ |
| **RAGAS** | 离线评测套件 | ✅ `answer_correctness`、`answer_semantic_similarity` |
| **LlamaIndex** | 两者都提供 | ✅ `CorrectnessEvaluator`、`SemanticSimilarityEvaluator` |

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

## 附：想加「正确性」评估时的接法

```python
from llama_index.core.evaluation import CorrectnessEvaluator

correctness_evaluator = CorrectnessEvaluator(llm=Settings.llm)
evaluators = {
    "faithfulness": faithfulness_evaluator,
    "relevancy": relevancy_evaluator,
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

`CorrectnessEvaluator` 默认满分 5，`score` 已除以阈值归一化到 0~1，用 `calc_avg_score` 统计即可。
