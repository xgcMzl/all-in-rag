import os

from dotenv import load_dotenv
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 1. 配置嵌入模型
# 关键：加载索引前必须设置成与建索引时相同的嵌入模型，
# 否则 LlamaIndex 会退回默认的 OpenAI embedding（无 Key 直接报错），
# 即使能跑，查询向量和索引向量也不在同一语义空间，检索结果毫无意义。
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 2. 从本地加载已持久化的索引
persist_path = "./llamaindex_index_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
index = load_index_from_storage(storage_context)
print(f"索引已从 {persist_path} 加载完成")

# 3. 只用检索器做相似度检索（不调用 LLM，无需 API Key）
query = "LlamaIndex 是做什么的？"
retriever = index.as_retriever(similarity_top_k=2)
nodes = retriever.retrieve(query)

print(f"\n查询: '{query}'")
print("相似度最高的节点:")
for node in nodes:
    print(f"- [score={node.score:.4f}] {node.get_content()}")

# 4. 需要生成自然语言答案时，再配置 LLM 走 query_engine
#    query_engine = index.as_query_engine()
#    response = query_engine.query(query)
#    print(response)
# if os.getenv("DEEPSEEK_API_KEY"):
#     from llama_index.llms.deepseek import DeepSeek
#
#     Settings.llm = DeepSeek(
#         model="deepseek-chat",
#         temperature=0.1,
#         api_key=os.getenv("DEEPSEEK_API_KEY"),
#     )
#     query_engine = index.as_query_engine(similarity_top_k=2)
#     print("\nLLM 生成的答案:")
#     print(query_engine.query(query))
# else:
#     print("\n（未检测到 DEEPSEEK_API_KEY，跳过 LLM 生成答案的部分）")
