"""
基于 LLM 的路由（LLM-based Routing）入门示例

【要解决的问题】
    用户的问题五花八门："麻婆豆腐怎么做" 和 "白切鸡的正宗做法" 需要不同的
    专家来回答。如果所有问题都丢给同一个通用提示词，回答质量会打折。
    传统做法是写一堆 if/else 规则去判断，但规则永远写不完。

【LLM 路由的思路】
    先用一次 LLM 调用做"分类器"，判断问题属于哪个领域，再把它分发到
    对应的处理链。分类这一步交给 LLM，就不用维护规则表了。
        question ──> 分类器 LLM ──> "川菜" / "粤菜" / "其他"
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                      川菜链         粤菜链        通用链

【本文件用到的 LangChain 核心概念：LCEL】
    LCEL 就是 "用 | 把组件连起来" 的写法，类似 shell 的管道：
        prompt | llm | parser
    数据从左往右流：prompt 收到参数 -> 生成 PromptValue ->
    llm 收到 PromptValue -> 返回 AIMessage ->
    parser 收到 AIMessage -> 返回普通字符串。
    中间任意一步都可以换成别的组件。
"""

import os
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser：把 LLM 返回的 AIMessage 对象取成纯字符串，
# 不加这一步，下游拿到的是消息对象而不是文本
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
# RunnableBranch：按条件把输入分发给不同处理链的组件，也就是"路由"
from langchain_core.runnables import RunnableBranch, RunnableLambda

# ---------------------------------------------------------------------------
# 0. 初始化大模型
# ---------------------------------------------------------------------------
# temperature=0：让输出尽量确定。
# 路由场景必须要确定性——同一个问题这次分到"川菜"、下次分到"粤菜"就乱套了。
# api_key 从环境变量读，避免把密钥写死在代码里提交到仓库
llm = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

# ---------------------------------------------------------------------------
# 1. 设置不同菜系的处理链
# ---------------------------------------------------------------------------
# ChatPromptTemplate.from_template(...)：把一段带占位符的字符串变成"提示词模板"。
# 模板里的 {question} 在调用时会被实际参数替换掉。
# 注意：模板最终会被发给"聊天模型"，所以它其实被包装成了一条 user 消息
sichuan_prompt = ChatPromptTemplate.from_template(
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
)
# 用 | 把三步串成一条链：
#   模板(填参) -> 大模型(生成) -> 解析器(转成字符串)
# sichuan_chain.invoke({"question": "..."}) 就能拿到字符串答案
sichuan_chain = sichuan_prompt | llm | StrOutputParser()

cantonese_prompt = ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
)
cantonese_chain = cantonese_prompt | llm | StrOutputParser()

# 定义备用通用链
# 当分类结果既不是川菜也不是粤菜时，走这条"兜底"链，
# 保证任何输入都有回答，不会报错
general_prompt = ChatPromptTemplate.from_template(
    "你是一个美食助手。请回答关于「{question}」的问题。"
)
# general_chain = general_prompt | llm | StrOutputParser()
def classifier(info):
    chain = general_prompt | llm | StrOutputParser()
    invoke = chain.invoke(info)
    print(f"我是普通链")
    return invoke
general_chain = RunnableLambda(classifier)

# ---------------------------------------------------------------------------
# 2. 创建路由链
# ---------------------------------------------------------------------------
# 第一步：分类器。它本身也是一条完整的 LCEL 链（模板 -> LLM -> 字符串）
classifier_prompt = ChatPromptTemplate.from_template(
    """你是菜品菜系分类器。根据用户问题中提到的菜品，将问题分类为以下三个标签之一：川菜、粤菜、其他。
分类规则：若问题中明确提到的菜品属于川菜，返回“川菜”；若属于粤菜，返回“粤菜”；若未提到菜品、无法判断、包含多个不同菜系或不属于川菜/粤菜，返回“其他”。
只输出一个标签词，不要输出标点、空格、引号、解释或额外文字。
用户问题：{question}"""
)
# "不要解释理由，只返回一个单词" 这句很关键：
# 后面路由分支用的是简单的字符串包含判断（"川菜" in topic），
# 如果模型回一句"这个问题属于川菜类，因为……"，虽然也含"川菜"能命中，
# 但如果它回英文或者换个说法，判断就会落空 -> 全部走兜底链
# classifier_chain = classifier_prompt | llm | StrOutputParser()
def classifier(info):
    chain = classifier_prompt | llm | StrOutputParser()
    invoke = chain.invoke(info)
    return invoke
classifier_chain = RunnableLambda(classifier)

# 第二步：分支路由
# RunnableBranch 的构造方式是：
#     RunnableBranch(
#         (条件函数1, 条件1成立时执行的链),
#         (条件函数2, 条件2成立时执行的链),
#         ...
#         默认链          <- 最后一个位置参数（不是元组）就是兜底分支
#     )
# 每个"条件函数"接收当前的输入，返回 True/False；按顺序匹配，命中第一个就执行
# 并返回，后面的不再检查——所以顺序是有意义的
router_branch = RunnableBranch(
    # x 是上游传来的字典，例如 {"topic": "川菜", "question": "麻婆豆腐怎么做？"}
    # 用 in 做子串判断，能容忍模型输出里夹带标点或多余空格
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], cantonese_chain),
    general_chain  # 默认选项：上面都没命中时执行
)

# 第三步：把"分类"和"路由"拼起来
# 左边这个字典是 LangChain 的"并行执行"写法（RunnableParallel）：
# 字典的每个 value 都会用【同一份输入】被调用，最后把各自结果组装成一个新字典。
#   - key "topic"    : 交给 classifier_chain，得到一个分类字符串
#                      （它内部直接取输入字典里的 "question" 字段）
#   - key "question" : 用 lambda 原样取出用户问题（lambda 在这里也是合法的 Runnable）
# 所以整条链的输入是 {"question": "..."}，
# 经过左边之后变成 {"topic": "川菜", "question": "..."}，
# 再交给 router_branch 做路由判断，并由被选中的链使用其中的 "question"
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch
print("完整的路由链创建成功。\n")


# ---------------------------------------------------------------------------
# 3. 运行演示查询
# ---------------------------------------------------------------------------
demo_questions = [
    {"question": "麻婆豆腐怎么做？"},      # 应该路由到川菜
    {"question": "白切鸡的正宗做法是什么？"}, # 应该路由到粤菜
    {"question": "番茄炒蛋需要放糖吗？"}      # 应该路由到其他
]

for i, item in enumerate(demo_questions, 1):
    question = item["question"]
    print(f"\n--- 问题 {i}: {question} ---")

    try:
        # 获取路由决策
        # 注意：这里单独又跑了一次分类器，只是为了把"路由到了哪里"打印出来。
        # 下面的 full_router_chain 内部还会再跑一次分类器，
        # 也就是说每个问题实际调用了 2 次分类 + 1 次回答 = 3 次 LLM。
        # 生产环境想省成本，可以改成自己组装链、复用同一个分类结果
        # topic = classifier_chain.invoke({"question": question})
        # print(f"路由决策: {topic}")

        # 执行完整链（分类 + 路由 + 回答，一步到位）
        result = full_router_chain.invoke(item)
        print(f"回答: {result}")
    except Exception as e:
        # 网络、鉴权、额度等问题都会落到这里。
        # 用 try/except 包住的好处是：一个问题失败不会中断整批演示
        print(f"执行错误: {e}")
