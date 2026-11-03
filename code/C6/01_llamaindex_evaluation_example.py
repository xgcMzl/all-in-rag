# =============================================================================
# 文件名：01_llamaindex_evaluation_example.py
# 作用：用 LlamaIndex 对 RAG（检索增强生成）系统的「回答质量」做自动评估，
#       并对比两种不同的文档切分/检索策略：
#         ① 句子窗口检索（Sentence Window Retrieval）
#         ② 常规分块检索（SentenceSplitter + 固定 chunk_size，作为基准 baseline）
#
# 什么是 RAG？
#   RAG = Retrieval Augmented Generation（检索增强生成）。
#   大模型本身不知道你的私有文档，所以流程是：
#     用户提问 → 先去文档库里检索最相关的片段 → 把片段塞进提示词 → 让大模型基于片段回答。
#   这样回答就有据可依，能减少「胡编乱造」（幻觉）。
#
# 为什么需要评估？
#   改了切分方式、检索数量、提示词之后，回答到底变好了还是变差了？
#   靠肉眼看几条例子是不靠谱的，所以需要「用大模型给大模型的回答打分」（LLM-as-a-judge）。
#
# 本脚本采用业界标准的「RAG 三元组」（RAG Triad），三个指标各管一段流水线：
#   ① Context Relevance（上下文相关性）—— 检索环节：检索回来的片段切题吗？够用来回答吗？
#   ② Groundedness     （扎根性/忠实度）—— 生成环节：整段回答是否被检索内容支撑？
#                                          （衡量有没有幻觉）
#                                          注意 LlamaIndex 的口径比较宽松：只要上下文里
#                                          有任意内容支持回答，就判通过，不会逐句核查。
#   ③ Answer Relevance （答案相关性）  —— 端到端  ：回答本身切中用户的问题吗？
#                                          （衡量有没有答非所问）
#   三者都不需要参考答案，所以每一条 query 都能算，适合在线监控。
# =============================================================================

# os：用来读环境变量（API Key）、判断文件是否存在（os.path.exists）。
import os
# asyncio：Python 的异步编程库。LlamaIndex 的批量评估是异步的（同时发多个请求更快），
#          所以要在异步函数里写，最后用 asyncio.run() 启动。
import asyncio

# VectorStoreIndex：向量索引。文档被切成小块（Node）后，每块会被嵌入模型转成向量（一堆数字），
#                   存进索引里。查询时把问题也转成向量，找最相似的块 —— 这就是「向量检索」。
# SimpleDirectoryReader：读文件的小工具，能把 PDF / txt 等变成 LlamaIndex 的 Document 对象。
# Settings：LlamaIndex 的「全局配置单例」。在这里设置一次 llm 和 embed_model，
#           后面所有组件没显式传参时都会默认用它俩。
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# SentenceWindowNodeParser：句子窗口切分器。每个 Node 只存「一句话」用于检索（精准），
#                           但同时在 metadata 里存上它前后各 window_size 句（用于生成时提供上下文）。
# SentenceSplitter：最常规的切分器，按字符数（chunk_size）把文本切成一段段固定大小的块。
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
# DeepSeek：DeepSeek 大模型的 LlamaIndex 封装（OpenAI 兼容接口），这里既用来生成回答，也用来打分。
from llama_index.llms.deepseek import DeepSeek
# HuggingFaceEmbedding：本地运行的嵌入模型封装，把文本变成向量，不需要联网调 API。
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# MetadataReplacementPostProcessor：后处理器。句子窗口检索的关键一步：
#   检索时用句子向量匹配，但真正喂给大模型时，把句子替换成它前后的「窗口」文本，给出更完整的上下文。
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# 评估相关组件（正好对应 RAG 三元组）：
#   ContextRelevancyEvaluator：① 上下文相关性。只把 query 和 contexts 交给裁判，
#                              问「检索到的片段是否切题、是否足以回答问题」，输出 0~1。
#   FaithfulnessEvaluator    ：② 扎根性/忠实度。把 contexts 和 response 交给裁判，
#                              问「回答是否被上下文支撑」，输出 YES/NO（passing）。
#   AnswerRelevancyEvaluator ：③ 答案相关性。只把 query 和 response 交给裁判
#                              （源码里显式 del contexts，不看上下文），输出 0~1。
#   BatchEvalRunner          ：批量评估器，能并发跑多条问题 + 多个评估器（workers 控制并发数）。
#
# 注：LlamaIndex 里还有一个 RelevancyEvaluator，名字像「相关性」，实际测的是
#     「回答与上下文是否一致」（YES/NO），和忠实度高度重叠，不属于三元组，本脚本不再使用。
from llama_index.core.evaluation import (
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    BatchEvalRunner,
)
# DatasetGenerator  ：自动生成评估数据集（它会读你的文档，自动出题 + 给标准答案参考答案）。
# QueryResponseDataset：评估数据集对象，包含 queries（问题列表）等字段，可以存成 JSON 复用。
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

# -----------------------------------------------------------------------------
# 全局配置：设置默认使用的大模型和嵌入模型
# -----------------------------------------------------------------------------
# Settings.llm：默认 LLM。
#   model="deepseek-chat"：模型名（DeepSeek 的对话模型）。
#   temperature=0.1：温度，0~2，越低输出越稳定/保守，越高越有创造性。评估场景要稳定，所以给 0.1。
#   api_key=...：从系统环境变量 DEEPSEEK_API_KEY 读取密钥，避免把密钥硬编码进代码。
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
# Settings.embed_model：默认嵌入模型。
#   BAAI/bge-small-en 是 BAAI（北京智源）出的英文小模型（small = 体积小、速度快）。
#   注意：这里用的是英文模型，所以文档最好也是英文（本脚本读的是英文 IPCC 报告，没问题）。
#   第一次运行会自动从 HuggingFace 下载模型权重，需要联网，之后再运行就是读本地缓存。
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")


# async def：定义异步函数。函数里可以用 await 等待耗时操作（网络请求）而不阻塞程序。
# 注意：异步函数不能直接调用，必须用 asyncio.run(main()) 启动（见文件末尾）。
async def main():
    # =========================================================================
    # 第 1 步：加载文档
    # =========================================================================
    # SimpleDirectoryReader：输入一个文件路径列表，把文件内容读成 Document 对象列表。
    #   input_files=[...]：指定具体文件（而不是 input_dir=某个目录）。
    #   路径是「相对路径」，相对于你运行脚本时所在的目录（通常是 code/C6），
    #   所以 cd 到别的目录跑会报「文件找不到」。
    reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
    # load_data()：真正执行读取。返回 list[Document]，一个 PDF 可能对应 1 个或多个 Document。
    documents = reader.load_data()

    # =========================================================================
    # 第 1.1 步：准备「评估数据集」—— 也就是一批「问题 + 参考答案」
    # =========================================================================
    # 评估必须有题目。这里采用「缓存」策略：第一次跑自动生成并保存成 JSON，
    # 之后再跑直接读 JSON。好处是每次对比用的是同一批题，结果才有可比性；
    # 而且省掉了重复出题的钱和时间。
    if os.path.exists("./c6_response_eval_dataset.json"):
        # 情况 A：本地已有数据集文件 → 直接加载
        print("加载响应评估数据集...")
        # from_json：从 JSON 文件反序列化回 QueryResponseDataset 对象。
        response_eval_dataset = QueryResponseDataset.from_json("./c6_response_eval_dataset.json")
    else:
        # 情况 B：文件不存在 → 现场生成一套
        print("生成响应评估数据集...")
        # DatasetGenerator.from_documents(...)：根据文档内容自动生成问题。
        #   documents[:10]：只取前 10 个 Document（代码注释说「减少文档数量」）。
        #   原因：自动出题要调很多次大模型，文档越多越慢越费钱，学习示例里减量跑得快。
        dataset_generator = DatasetGenerator.from_documents(documents[:10])  # 减少文档数量
        # agenerate_dataset_from_nodes(num=15)：异步生成评估数据集。
        #   num=15：生成 15 道题。
        #   await：等待这个异步任务跑完。生成的每题通常还带参考答案（reference answer）。
        response_eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=15)  # 减少问题数量
        # save_json：把生成好的数据集存到磁盘，下次直接读，不用再花钱生成。
        response_eval_dataset.save_json("./c6_response_eval_dataset.json")

    # =========================================================================
    # 第 2 步：构建两种 RAG 检索方案，后面做对比
    # =========================================================================

    # -------------------------------------------------------------------------
    # 2.1 方案一：句子窗口检索（Sentence Window Retrieval）
    # -------------------------------------------------------------------------
    # 原理一句话：用「一个小句子」去检索（精准），但用「这个小句子 + 前后文窗口」去回答（上下文完整）。
    #
    # 为什么要这么设计？
    #   - 块切大了：检索时噪音多，向量会被无关内容稀释，匹配不准。
    #   - 块切小了：检索准，但信息不完整，大模型看到半句话没法回答。
    #   句子窗口把「检索粒度」和「阅读粒度」解耦，同时拿到两边的好处。
    sentence_parser = SentenceWindowNodeParser.from_defaults(
        # window_size=3：每个句子左右各取 3 句，组成一个「窗口」。
        #   所以一个 Node 的 window 文本 ≈ 目标句子 + 前 3 句 + 后 3 句，共约 7 句。
        window_size=3,
        # window_metadata_key="window"：窗口文本存到 metadata 的 "window" 键下。
        #   后面 MetadataReplacementPostProcessor(target_metadata_key="window") 就靠这个键取值。
        window_metadata_key="window",
        # original_text_metadata_key="original_text"：把原始的单句也存一份到 metadata 的 "original_text" 键下，
        #   方便你自己查看/调试「检索命中的到底是哪一句」。
        original_text_metadata_key="original_text",
    )
    # get_nodes_from_documents：把 Document 按上面的规则切成 Node 列表（Node 是 LlamaIndex 的检索基本单元）。
    sentence_nodes = sentence_parser.get_nodes_from_documents(documents)
    # VectorStoreIndex：把这些 Node 向量化并建索引（默认存在内存里）。
    #   注意：这里只传 nodes，没传 embed_model，所以会自动用 Settings.embed_model（bge-small-en）。
    sentence_index = VectorStoreIndex(sentence_nodes)

    # as_query_engine：把索引包装成「查询引擎」—— 输入一个问题，自动完成
    #   「向量检索 → 拼提示词 → 调 LLM 生成回答」这一整条链路。
    sentence_query_engine = sentence_index.as_query_engine(
        # similarity_top_k=2：检索时取相似度最高的 2 个 Node。
        similarity_top_k=2,
        node_postprocessors=[
            # 关键！检索拿回来的是「句子 Node」，这里把它替换成 metadata["window"] 里的窗口文本，
            # 再交给 LLM 生成回答。如果不加这一步，LLM 就只能看到孤立的一句话。
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    # 补充说明：如果只想拿「检索结果」而不让 LLM 生成回答，可以用
    #   sentence_index.as_retriever(similarity_top_k=2)
    # 本脚本评估的是完整回答，所以用 query_engine，不创建 retriever。

    # -------------------------------------------------------------------------
    # 2.2 方案二：常规分块检索（基准 baseline）
    # -------------------------------------------------------------------------
    # 最朴素的 RAG 做法：按固定长度切块，检索到哪块就把整块丢给 LLM。
    # 拿它当基准，才能判断「句子窗口」到底是真提升了，还是只是随机波动。
    base_parser = SentenceSplitter(chunk_size=512)
    #   chunk_size=512：每块最多 512 个 token（注意不是字符，是 token 数）。
    #   其余参数用默认值，例如 chunk_overlap 默认 200，即相邻块重叠 200 token，避免把一句话拦腰切断。
    base_nodes = base_parser.get_nodes_from_documents(documents)
    base_index = VectorStoreIndex(base_nodes)

    # 同样创建一个查询引擎，但没有 postprocessor（因为整块内容本来就是完整的）。
    base_query_engine = base_index.as_query_engine(similarity_top_k=2)

    # =========================================================================
    # 第 3 步：初始化评估器（LLM-as-a-judge：让大模型当裁判）
    # =========================================================================
    # 三个评估器内部都会自己构造提示词去问 LLM，输入输出各不同：
    #   ① ContextRelevancyEvaluator：吃 query + contexts，问检索片段是否切题。
    #      裁判问两个子问题，每问 2 分（允许半分），满分 4 分，
    #      最终 score = 原始分 / score_threshold(默认 4.0) → 0~1。
    #      不设置 passing 字段 → 统计时算平均分。
    #   ② FaithfulnessEvaluator：吃 contexts + response，问回答是否被上下文支撑。
    #      整段回答问一次 YES/NO（注意：是整体判断，不会逐句拆分）。
    #      返回 passing=True/False → 统计时算通过率。
    #   ③ AnswerRelevancyEvaluator：吃 query + response，问回答是否切题。
    #      裁判问两个子问题，每问 1 分，满分 2 分，
    #      最终 score = 原始分 / score_threshold(默认 2.0) → 0~1。
    #      不设置 passing 字段 → 统计时算平均分。
    # llm=Settings.llm：显式指定裁判模型（这里也是 deepseek-chat）。
    #   想省钱/更快可以换成更小的模型，但判断质量会下降。
    context_relevancy_evaluator = ContextRelevancyEvaluator(llm=Settings.llm)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=Settings.llm)

    # =========================================================================
    # 第 4 步：批量执行评估（两个方案各跑一遍同样的问题）
    # =========================================================================
    print("开始执行响应评估对比...")
    # 把多个评估器打包成 dict，key 会成为结果的键名。
    # 顺序对应 RAG 三元组的三个阶段：检索 → 生成 → 端到端。
    evaluators = {
        "context_relevancy": context_relevancy_evaluator,  # ① 检索是否切题（score 型：0~1）
        "faithfulness": faithfulness_evaluator,            # ② 回答是否有依据（passing 型：True/False）
        "answer_relevancy": answer_relevancy_evaluator,    # ③ 回答是否切题（score 型：0~1）
    }
    # queries：评估数据集里的问题列表。
    #   ⚠️ 重要：QueryResponseDataset.queries 的类型是 Dict[str, str]
    #      （"题目ID" -> "问题文本"），不是 list！
    #   而 aevaluate_queries 内部是 `for query in queries`，
    #   直接传 dict 遍历到的是「key（题目ID）」，不是问题本身，
    #   结果会拿 UUID 字符串去检索和提问，分数全是垃圾。
    #   所以必须取出问题文本，再转成 list。
    #
    # 另外要过滤脏数据：DatasetGenerator 偶尔会把「整段生成输出」当成一道题存进来
    # （例如 "Based on the provided context..., here are N questions..."），
    # 这种条目会同时拉低三个指标，评估前必须剔除。
    def is_valid_query(text: str) -> bool:
        text = text.strip()
        if not text or "\n" in text:
            return False  # 空文本或多行文本，基本是没解析干净的生成输出
        low = text.lower()
        if "here are" in low and "questions" in low:
            return False  # 典型脏数据特征
        return True

    # 用「题目ID」把问题和参考答案配对后一起过滤。
    # 这比假设两个 dict 的 .values() 顺序一致更安全——顺序一致只是当前文件的巧合，不是 API 保证。
    valid_items = [
        (qid, question, response_eval_dataset.responses.get(qid, ""))
        for qid, question in response_eval_dataset.queries.items()
        if is_valid_query(question)
    ]
    queries = [question for _, question, _ in valid_items]
    # 对应的参考答案。注意：三元组的三个指标全部是无参考指标，用不到它；
    # 只有 CorrectnessEvaluator、SemanticSimilarityEvaluator 这类有参考指标才需要。
    references = [reference for _, _, reference in valid_items]
    print(f"有效问题数：{len(queries)} / {len(response_eval_dataset.queries)}")

    # ---- 方案一的评估 ----
    print("\n=== 评估句子窗口检索 ===")
    # BatchEvalRunner(评估器字典, workers=并发数, show_progress=是否显示进度条)
    #   workers=2：同时最多跑 2 个评估任务。调大会更快，但容易触发 API 限流（429）。
    sentence_runner = BatchEvalRunner(evaluators, workers=3, show_progress=True)
    # aevaluate_queries：异步批量评估。
    #   对 queries 里的每个问题，先让 query_engine 生成回答，再用每个评估器打分。
    #   返回值是嵌套 dict，结构大致是：
    #     {"context_relevancy": [...], "faithfulness": [...], "answer_relevancy": [...]}
    #   每个 EvaluationResult 有 .passing（是否通过）、.score、.feedback（裁判给的理由）等字段。
    sentence_response_results = await sentence_runner.aevaluate_queries(
        queries=queries, query_engine=sentence_query_engine
    )

    # ---- 方案二的评估 ----
    print("\n=== 评估常规分块检索 ===")
    # 用同样的评估器、同样的并发数、同样的问题，只换 query_engine —— 保证唯一变量是检索方案。
    base_runner = BatchEvalRunner(evaluators, workers=3, show_progress=True)
    base_response_results = await base_runner.aevaluate_queries(
        queries=queries, query_engine=base_query_engine
    )

    # =========================================================================
    # 第 5 步：汇总分数并打印对比结果
    # =========================================================================
    print("\n" + "="*60)
    print("响应评估结果对比")
    print("="*60)

    # 注意：不同评估器的返回字段不一样，所以要两个统计函数。
    #   - FaithfulnessEvaluator                      → 给的是 passing（True/False）→ 算通过率
    #   - ContextRelevancyEvaluator / AnswerRelevancyEvaluator
    #                                                → 只给 score（0~1），passing 是 None → 算平均分
    # 如果混用，sum(r.passing ...) 遇到 None 会抛 TypeError。

    # 函数一：算「通过率」，适用于 passing 型指标
    #   results：BatchEvalRunner 返回的嵌套 dict
    #   metric ：指标名，比如 "faithfulness"
    def calc_pass_rate(results, metric):
        if results and results.get(metric):
            # 只保留真正给出了判定结果的条目（passing 不是 None 的）
            items = [r for r in results[metric] if r.passing is not None]
            if not items:
                return 0.0
            # True 按 1 算、False 按 0 算，求和除以条数 = 通过率（0~1）
            return sum(1 for r in items if r.passing) / len(items)
        return 0.0

    # 函数二：算「平均分」，适用于 score 型指标
    def calc_avg_score(results, metric):
        if results and results.get(metric):
            # 过滤掉解析失败（invalid_result=True）和没有分数的条目，避免拉低/污染均值
            scores = [
                r.score
                for r in results[metric]
                if r.score is not None and not r.invalid_result
            ]
            if not scores:
                return 0.0
            return sum(scores) / len(scores)
        return 0.0

    # 句子窗口检索的三个指标（按三元组顺序：检索 → 生成 → 端到端）
    sentence_ctx_rel = calc_avg_score(sentence_response_results, "context_relevancy")
    sentence_faith = calc_pass_rate(sentence_response_results, "faithfulness")
    sentence_ans_rel = calc_avg_score(sentence_response_results, "answer_relevancy")

    # 常规分块检索的三个指标
    base_ctx_rel = calc_avg_score(base_response_results, "context_relevancy")
    base_faith = calc_pass_rate(base_response_results, "faithfulness")
    base_ans_rel = calc_avg_score(base_response_results, "answer_relevancy")

    # f-string 里的 {sentence_faith:.1%} 表示：把 0.8 这样的数格式化成 "80.0%"。
    print(f"\n句子窗口检索:")
    print(f"  ① 上下文相关性: {sentence_ctx_rel:.1%}")
    print(f"  ② 忠实度:       {sentence_faith:.1%}")
    print(f"  ③ 答案相关性:   {sentence_ans_rel:.1%}")

    print(f"\n常规分块检索:")
    print(f"  ① 上下文相关性: {base_ctx_rel:.1%}")
    print(f"  ② 忠实度:       {base_faith:.1%}")
    print(f"  ③ 答案相关性:   {base_ans_rel:.1%}")

    # -------------------------------------------------------------------------
    # 简单对比结论（逐维度比较，统计赢了几项）
    # -------------------------------------------------------------------------
    # 三个维度：(句子窗口分数, 常规分块分数)
    dimensions = {
        "上下文相关性": (sentence_ctx_rel, base_ctx_rel),
        "忠实度": (sentence_faith, base_faith),
        "答案相关性": (sentence_ans_rel, base_ans_rel),
    }
    # 逐维度打印谁赢，方便看出差距出在检索段还是生成段
    for name, (s_score, b_score) in dimensions.items():
        if s_score > b_score:
            flag = "句子窗口领先"
        elif s_score < b_score:
            flag = "常规分块领先"
        else:
            flag = "持平"
        print(f"  {name}: {s_score:.1%} vs {b_score:.1%} → {flag}")

    # 统计「句子窗口赢」的维度数量
    wins = sum(1 for s, b in dimensions.values() if s > b)
    print(f"\n句子窗口检索在 {wins}/{len(dimensions)} 个维度上领先")

    # 样本量提醒：比例类指标在小样本下波动很大（十几条样本的差异可能只是噪声），
    # 所以这里只说「高/低」，不说「显著优于」，结论只能当趋势参考。
    print(f"（样本量 n={len(queries)}，比例类指标在小样本下波动较大，结论仅供参考）")

    if wins == len(dimensions):
        print(f"\n✅ 句子窗口检索在所有维度上都高于常规分块检索")
    elif wins > 0:
        print(f"\n⚖️  句子窗口检索在某些维度上更高")
    else:
        print(f"\n❌ 句子窗口检索未显示明显优势")


# Python 的入口判断：只有「直接运行这个文件」时才执行下面的代码；
# 被别的文件 import 时不会执行。
# asyncio.run(main())：启动事件循环，运行异步函数 main()，直到它执行完毕。
if __name__ == "__main__":
    asyncio.run(main())
