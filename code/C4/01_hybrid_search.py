"""
混合检索（Hybrid Search）入门示例：Milvus + BGE-M3

先理解两个概念，后面所有代码都是围绕它们展开的：

【密集向量 dense vector】
    把一段文字交给神经网络模型，输出一个定长浮点数组（这里是 1024 维）。
    它表达的是"语义"：说法完全不同但意思相近的两句话，向量会很接近。
    优点：能理解同义词、换一种说法。
    缺点：对专有名词、编号、精确关键词不敏感（比如"编号 A-1024"）。

【稀疏向量 sparse vector】
    一个维度极高、但绝大多数位置都是 0 的向量（这里 25 万维左右，
    非零元素通常只有几十到几百个）。非零位置的"下标"代表某个词，
    "值"代表这个词在这段文字里的权重。
    优点：关键词精确命中能力强（类似传统倒排索引 / BM25）。
    缺点：不懂同义词，"龙"和"dragon"是两回事。

【混合检索 hybrid search】
    同时用上面两种向量各搜一遍，再把两份结果"融合"成一个排名。
    这样既有语义理解，又不会漏掉精确关键词。融合算法本文件用的是 RRF。

整体流程：
    1. 读配置
    2. 连接 Milvus + 初始化 BGE-M3（一个同时能产出 dense 和 sparse 的模型）
    3. 建 Collection（相当于关系数据库里的"表"）
    4. 给两个向量字段各建索引
    5. 读 json 数据 -> 拼成待检索文本 -> 编码成向量 -> 插入
    6. 分别做 dense 检索、sparse 检索、混合检索，对比三者结果
    7. 释放并删除 Collection（Demo 收尾）

注意：所有路径都是相对路径，必须在 code/C4 目录下运行本脚本。
"""

import json
import os
import numpy as np
# connections: 老式的连接管理接口，用来建立全局连接
# MilvusClient: 新式高层客户端，适合做"建表/删表/加载/释放"这类管理操作
# FieldSchema / CollectionSchema / DataType: 定义"表结构"的三个类
# Collection: 面向集合的 ORM 风格对象，用来插入数据和执行检索
# AnnSearchRequest: 描述"一路检索请求"（用哪个字段、什么参数、取前几条）
# RRFRanker: 融合多路结果用的 Reciprocal Rank Fusion 排名器
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker
# BGE-M3 是智源(BAAI)出的多语言嵌入模型，最大特点是一次编码同时给出
# dense（密集）和 sparse（稀疏）两种向量 —— 正是混合检索需要的
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# ---------------------------------------------------------------------------
# 1. 初始化设置
# ---------------------------------------------------------------------------
# Collection 名，相当于表名
COLLECTION_NAME = "dragon_hybrid_demo"
# Milvus 服务地址。格式固定是 "<协议>://<主机>:<端口>"，
# 这里连的是远程服务器（不是本地 Docker），所以要求网络能通
MILVUS_URI = "http://106.55.165.169:19530"  # 服务器模式
# 原始数据文件，每条形如 {"img_id":..., "title":..., "description":..., ...}
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径
# 批量插入大小。本 Demo 数据量很小，这个变量实际没被用到，
# 数据量上万时要真的按它分批 insert，否则一次请求会过大
BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# 2. 连接 Milvus 并初始化嵌入模型
# ---------------------------------------------------------------------------
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
# 建立"全局默认连接"。后面的 Collection(...) 不带连接参数时会自动用它
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
# use_fp16=False：不用半精度，精度更高但更慢（CPU 上建议 False）
# device="cpu"：没有 GPU 就在 CPU 上跑，慢但能用
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
# 模型的输出是一个 dict，键 "dense" 和 "sparse" 分别对应两种向量；
# 密集向量维度固定（BGE-M3 是 1024），稀疏向量维度取决于词表大小
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# ---------------------------------------------------------------------------
# 3. 创建 Collection
# ---------------------------------------------------------------------------
# MilvusClient 用来做集合级别的管理操作（是否存在 / 删除）
milvus_client = MilvusClient(uri=MILVUS_URI)
if milvus_client.has_collection(COLLECTION_NAME):
    # Demo 每次都想要一份干净的数据，所以同名集合存在就先删掉。
    # drop_collection 是物理删除，里面的数据会一并消失
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

# 字段定义 = 表结构。Milvus 是强 schema 的，字段名、类型、长度都要先声明
fields = [
    # 主键（primary key），类型是变长字符串，auto_id=True 表示
    # 主键由 Milvus 自动生成，插入数据时不需要（也不能）自己填
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    # 业务上的图片编号，方便结果里对回原始数据
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    # 图片文件路径
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    # 标题 / 描述 / 分类 / 地点 / 环境，都是普通的"标量字段"（非向量），
    # 作用有两个：一是作为检索结果回带给调用方，二是可以参与过滤（expr）
    # 提醒：VARCHAR 的 max_length 按【字节】算，中文一个字占 3 字节
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    # 稀疏向量字段：只声明类型，不写维度（维度由数据本身决定）
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    # 密集向量字段：必须写死维度，且和模型输出一致，否则插入会报错
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# 如果集合不存在，则创建它及索引
# （上面刚删过，所以这里几乎一定成立；这样写是为了让脚本也能在
#   "集合已存在" 的极端情况下继续往下跑）
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    # 创建集合。consistency_level="Strong" 表示强一致：
    # 刚插入的数据立刻就能被搜到，方便 Demo 观察；代价是性能较低
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # -----------------------------------------------------------------------
    # 4. 创建索引
    # -----------------------------------------------------------------------
    # 向量字段必须建索引才能高效检索，否则只能暴力全表扫描。
    # 注意：两个字段要用【不同】的索引类型，因为稀疏和密集的底层结构不一样
    print("--> 正在为新集合创建索引...")
    # 稀疏向量用倒排索引：把"词 -> 包含该词的文档列表"存起来，精确匹配关键词。
    # metric_type="IP" 即内积（Inner Product）：稀疏向量通常已归一化，
    # 内积越大表示匹配度越高
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 密集向量用 AUTOINDEX：交给 Milvus 自己按数据规模和部署形态挑一个合适的
    # 索引（云服务上通常是 HNSW 一类）。本地想自己控制可以换成 HNSW/IVF_FLAT
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

# 用集合名重新拿到一个 Collection 句柄（不管上面走的是"新建"分支还是"已存在"分支，
# 到这里都能得到一个可用的对象）
collection = Collection(COLLECTION_NAME)

# ---------------------------------------------------------------------------
# 5. 加载数据并插入
# ---------------------------------------------------------------------------
# load() 把集合的索引从磁盘加载进内存，之后才能执行检索。
# Milvus 是存算分离的，"建表"和"能用它查"是两件事，这一步容易漏
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
    # 以 utf-8 读 json，避免中文乱码
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    docs, metadata = [], []
    for item in dataset:
        # 把一条记录里"适合用来检索"的文字拼成一段话。
        # 混合检索要先经过一次文本编码，所以结构化字段必须先拍平成文本
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
            # 以下几行被注释掉了，说明作者试过把战斗细节也拼进来，
            # 如果想让检索更细，可以打开这些字段：
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        # filter(None, parts) 会过滤掉空字符串等"假值"，
        # 避免拼出 "标题  描述" 这种多余空格
        docs.append(' '.join(filter(None, parts)))
        # 原始记录整体保留一份，插入时还要用里面的标量字段
        metadata.append(item)
    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print("--> 正在生成向量嵌入...")
    # 一次调用批量编码所有文本。返回值形如：
    #   {"dense":  ndarray(形状 [n, 1024]),
    #    "sparse": scipy 的 CSR 稀疏矩阵(形状 [n, 词表大小])}
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")
    # 为每个字段准备批量数据
    # 注意：Milvus 的 collection.insert() 接收的是"按字段排列的多个列表"，
    # 顺序必须和 schema 里字段定义的顺序一致，并且【跳过 auto_id 的主键 pk】
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # 获取向量
    # dense：普通的二维数组，第 i 行就是第 i 条文本的密集向量
    # sparse：CSR 稀疏矩阵，第 i 行是第 i 条文本的稀疏向量
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]

    # 插入数据
    # 9 个列表对应 schema 里的 9 个非主键字段，顺序不能乱
    collection.insert([
        img_ids,
        paths,
        titles,
        descriptions,
        categories,
        locations,
        environments,
        sparse_vectors,
        dense_vectors
    ])

    # flush 把内存里还在缓冲区中的数据强制落盘，并让它们可以立刻被检索到。
    # 不调用也能查（强一致），但立即查询的数据量统计可能滞后
    collection.flush()
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    # 集合里已经有数据就不重复插入，直接进入检索环节
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# ---------------------------------------------------------------------------
# 6. 执行搜索
# ---------------------------------------------------------------------------
search_query = "悬崖上的巨龙"
# 过滤表达式（标量过滤 + 向量检索 一起做，叫"过滤式检索"）。
# 语法类似 SQL 的 WHERE，这里只保留三个分类的记录。
# 先过滤再检索，既减少计算量，又能保证结果符合业务约束
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
# 每种检索各取前 5 条
top_k = 5

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# 查询文本同样要编码成两种向量：密集的用来做语义搜索，
# 稀疏的用来做关键词搜索。注意返回的是一个"只有 1 条"的批量结果
query_embeddings = ef([search_query])
# 取出第 0 条密集向量。[0] 是因为批量维度，二维数组取第一行就是一维向量
dense_vec = query_embeddings["dense"][0]
# 稀疏矩阵不能像数组那样取下标，要用 _getrow(0) 把第 0 行取出来，
# 得到的是一个 1 x 词表大小 的 CSR 子矩阵（所以下面取字段时还要用 [0]）
sparse_vec = query_embeddings["sparse"]._getrow(0)

# 打印向量信息
# 这一段纯粹是"教学观察"，让你直观看到两种向量的差异
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
# 范数 = 向量的长度。BGE-M3 的密集向量已被归一化，范数应该接近 1.0，
# 这也是内积(IP)和余弦相似度(COS)在此场景下等价的原因
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")

# 稀疏向量的"维度"是模型的词表大小，通常几十万维
print(f"\n稀疏向量维度: {sparse_vec.shape[1]}")
# nnz = number of non-zeros，非零元素个数，一般是几十到几百
print(f"稀疏向量非零元素数量: {sparse_vec.nnz}")
print("稀疏向量前5个非零元素:")
for i in range(min(5, sparse_vec.nnz)):
    # indices: 非零元素所在的维度下标（可理解为"哪个词"）
    # data:    该位置上的权重值（可理解为"这个词有多重要"）
    print(f"  - 索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")
# 密度 = 非零元素占比。稀疏向量密度极低（远小于 1%），
# 这正是"稀疏"这个名字的由来
density = (sparse_vec.nnz / sparse_vec.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")

# 定义搜索参数
# 检索时也要声明度量方式，且必须和建索引时的 metric_type 一致
search_params = {"metric_type": "IP", "params": {}}

# 先执行单独的搜索
# 第 1 种：只用密集向量（纯语义检索）
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],
    # anns_field: 用哪个向量字段来比较
    anns_field="dense_vector",
    param=search_params,
    # limit: 取回多少条
    limit=top_k,
    # expr: 标量过滤条件
    expr=search_filter,
    # output_fields: 除了主键和分数，还要把哪些字段带回来
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]  # 返回的是"每个查询一个结果列表"，我们只有一个查询，所以取 [0]

for i, hit in enumerate(dense_results):
    # hit 里有 id / distance(分数) / entity(标量字段)。
    # IP 度量下分数越大越相似
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    # [:100] 只截前 100 个字符，避免控制台刷屏
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 第 2 种：只用稀疏向量（纯关键词检索）
print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 第 3 种：混合检索 = 上面两种一起搜，再融合排名
print("\n--- [混合] 稀疏+密集向量搜索结果 ---")
# 创建 RRF 融合器
# RRF（Reciprocal Rank Fusion）不关心两路分数的绝对值（它们量纲完全不同，
# 没法直接相加），只用【排名】来融合：
#     某条结果的最终分 = Σ 1 / (k + 它在某一路中的名次)
# 所以一条结果在两路里都排得靠前，最终分就高。
# k=60 是论文推荐的经典值，作用是"压平"头几名的差距，避免某一路的一家独大
rerank = RRFRanker(k=60)

# 创建搜索请求
# 每个 AnnSearchRequest 描述"一路检索"：用哪个向量、查哪个字段、什么参数、取几条。
# 注意：混合检索里过滤条件要写在各自的请求里（本 Demo 没加 expr，
# 所以这里的混合检索结果是【全库】的，范围比前面两次更大）
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k,expr=search_filter)
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k,expr=search_filter)

# 执行混合搜索
# 把两路请求一起交给 Milvus，由它在服务端并行执行并用 rerank 融合，
# 返回的仍然"每个查询一个结果列表"，所以取 [0]。
# limit=top_k 是融合后最终返回的条数
results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印最终结果
# 这里的 Score 是 RRF 融合后的分数，数值很小（通常在 0.01~0.03 量级），
# 只能用来排序，不能和前两种检索的 IP 分数横向比较
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---------------------------------------------------------------------------
# 7. 清理资源
# ---------------------------------------------------------------------------
# release：把集合从查询节点的内存里卸载，内存紧张时很重要
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
# drop：连同 schema、索引、全部数据一起物理删除。
# 只在 Demo 里这么写，真实项目千万别放在脚本末尾
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")
