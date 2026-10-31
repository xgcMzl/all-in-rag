# ============================================================
# 这个文件用 LlamaIndex 实现了一个最小的 RAG（检索增强生成）程序。
#
# RAG 的思路很简单：大模型本身不知道你本地的私有文档，
# 所以可以先把文档切碎、转成"向量"存进索引；
# 用户提问时，先从索引里找出和问题最相关的几段文字，
# 再把这些文字和问题一起交给大模型，让它"看着资料"回答。
#
# 本文件整体流程：
#   1. 加载环境变量（API Key 等）
#   2. 配置全局的「大模型」和「向量模型」
#   3. 读取 md 文档
#   4. 用文档建立向量索引（切块 + 向量化 + 存储）
#   5. 把索引包装成"查询引擎"
#   6. 提问，打印答案
# ============================================================

import os  # 用于读取环境变量（比如 API Key）

# 下面这行被注释掉了。HuggingFace 网站在国内访问可能很慢，
# 取消注释后会把下载模型的地址换成国内镜像站 hf-mirror.com。
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'

# load_dotenv 会读取项目根目录下的 .env 文件，
# 把里面写的 KEY=VALUE 加载进 os.environ，这样就能用 os.getenv 取到密钥。
from dotenv import load_dotenv

# LlamaIndex 的三个核心类：
#   VectorStoreIndex      —— 向量索引：把文档切块、向量化，并支持相似度检索
#   SimpleDirectoryReader —— 文档读取器：从文件夹或指定文件里把文字读出来
#   Settings              —— 全局配置：在这里设一次，LlamaIndex 各处都会用它
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# OpenAILike 是一个"兼容 OpenAI 接口"的大模型适配器。
# 只要服务商提供的是 OpenAI 风格的 API（deepseek、通义、智谱等大多如此），
# 就可以用它来连接，不需要专门的 SDK。
from llama_index.llms.openai_like import OpenAILike

# HuggingFaceEmbedding 是"向量模型"适配器：
# 它把一段文字变成一串数字（向量），意思相近的文字，向量也相近，
# 这样程序才能用"数学距离"判断两段文字是否相关。
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 从 .env 文件加载环境变量。
# 注意：必须在读取 os.getenv("DEEPSEEK_API_KEY") 之前调用，否则取到的是 None。
load_dotenv()


# ------------------------------------------------------------
# 1) 配置大模型（LLM）：负责最后"组织语言、回答问题"
# ------------------------------------------------------------
# Settings 是 LlamaIndex 的全局配置对象。给 Settings.llm 赋值后，
# 后面创建索引、查询引擎时如果不额外指定，都会默认使用这个模型。
Settings.llm = OpenAILike(
    model="deepseek-flash",                     # 要调用的模型名称
    api_key=os.getenv("DEEPSEEK_API_KEY"),      # 从 .env 里读到的 DeepSeek 密钥
    api_base="https://api.deepseek.com",        # 服务商的接口地址（OpenAI 兼容格式）
    is_chat_model=True                          # 表示这是"对话模型"（走 /chat/completions 接口）
                                                # 如果填 False，就会走旧的补全接口
)

# ------------------------------------------------------------
# 2) 配置向量模型（Embedding Model）：负责"把文字变成向量"
# ------------------------------------------------------------
# BAAI/bge-small-zh-v1.5 是北京智源开源的轻量中文向量模型（small 版，速度快、体积小）。
# 第一次运行时，它会自动从 HuggingFace 下载模型文件并缓存到本地，
# 之后再用就从缓存加载，不会重复下载。
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")


# ------------------------------------------------------------
# 3) 读取文档
# ------------------------------------------------------------
# SimpleDirectoryReader 可以读整个文件夹，也可以用 input_files 精确指定某几个文件。
# .load_data() 返回一个 Document 列表，每个元素是一份读进来的文档对象。
# 这里的相对路径 "../../data/..." 是相对于「运行脚本时的当前工作目录」，
# 所以最好在 code/C1 目录下执行本脚本，否则会找不到文件。
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]
).load_data()


# ------------------------------------------------------------
# 4) 建立向量索引
# ------------------------------------------------------------
# from_documents 会做三件事（都在这一步内部完成）：
#   ① 把长文档切成许多小片段（chunk），因为后面要按片段检索；
#   ② 对每个片段调用上面的 embed_model 算出向量；
#   ③ 默认在内存里建一个向量库，把"片段 + 向量"存起来。
# 注意：这一步会比较慢，因为要给每段文字都算一次向量；
# 而且是内存索引，脚本一结束索引就没了，下次运行需要重新建。
index = VectorStoreIndex.from_documents(docs)


# ------------------------------------------------------------
# 5) 把索引包装成"查询引擎"
# ------------------------------------------------------------
# as_query_engine() 生成的 query_engine 把 RAG 的全流程封装好了：
#   你的问题 → 变成向量 → 在索引里检索最相关的几个片段
#            → 拼成提示词（prompt） → 交给大模型 → 返回答案
query_engine = index.as_query_engine()


# ------------------------------------------------------------
# 5.5) 解析提示词模板，只打印正文内容
# ------------------------------------------------------------
# query_engine.get_prompts() 返回的对象字段特别多（metadata、
# template_vars、kwargs、output_parser、函数内存地址……），
# 直接 print 出来是一大坨，根本看不清。
# 下面两个小函数把结构"剥"一遍，只保留真正发给大模型的文字。


def _template_to_text(template):
    """把单个模板对象转成纯文本，其他字段统统丢掉。"""
    # 情况一：ChatPromptTemplate（对话模板）→ 里面是多条消息
    messages = getattr(template, "message_templates", None)
    if messages:
        lines = []
        for msg in messages:
            # role 形如 MessageRole.SYSTEM，取 .value 得到 'system'
            role = getattr(msg, "role", None)
            role = getattr(role, "value", role)
            # content 是消息正文；老的写法里正文放在 blocks 中
            text = getattr(msg, "content", None)
            if text is None:
                text = "".join(
                    getattr(block, "text", "")
                    for block in (getattr(msg, "blocks", None) or [])
                )
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)

    # 情况二：普通 PromptTemplate → 正文本身就是字符串
    return getattr(template, "template", str(template))


def parse_prompts(prompts):
    """把 get_prompts() 的返回值拆成 {模板名: [(分支名, 正文), ...]}。"""
    result = {}
    for name, selector in prompts.items():
        branches = []

        # 兜底模板：当下面的条件都不满足时使用
        default_template = getattr(selector, "default_template", None)
        if default_template is not None:
            branches.append(("默认模板", _template_to_text(default_template)))

        # 条件分支：条件函数返回 True 时，就用它配的模板
        for condition, template in getattr(selector, "conditionals", None) or []:
            condition_name = getattr(condition, "__name__", str(condition))
            branches.append(
                (f"条件分支（{condition_name}）", _template_to_text(template))
            )

        # 如果它不是"选择器"而本身就是个模板，就直接取正文
        if not branches:
            branches.append(("模板", _template_to_text(selector)))

        result[name] = branches
    return result


def print_prompts(prompts):
    """按"模板名 → 分支名 → 正文"的层级打印，不显示任何其他字段。"""
    for name, branches in parse_prompts(prompts).items():
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        for branch_name, text in branches:
            print(f"\n--- {branch_name} ---")
            print(text)


# 提示词模板就是"喂给大模型的那段话的格式"，
# 里面会有 {context_str}（检索到的资料）和 {query_str}（你的问题）这样的占位符。
# 想定制回答风格时，就是改这些模板。
# 如果想看原始对象（很吵），把下面这行换成 print(query_engine.get_prompts())
print_prompts(query_engine.get_prompts())


# ------------------------------------------------------------
# 6) 提问
# ------------------------------------------------------------
# .query() 内部会先检索、再调用大模型，返回一个 Response 对象。
# 直接 print 它，会打印出答案文本。
print(query_engine.query("文中举了哪些例子?"))
