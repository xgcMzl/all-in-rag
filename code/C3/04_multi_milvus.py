"""
多模态图文检索 Demo（Milvus + Visualized-BGE）

整体流程：
    1. 加载 Visualized-BGE 编码器，把"图像"和"文本"映射到同一个向量空间
    2. 连接 Milvus 并建一个 Collection（相当于关系库里的"表"）
    3. 把一批图像逐张编码成向量后插入 Collection
    4. 给向量字段建 HNSW 索引，再把 Collection 加载进内存
    5. 用「一张查询图 + 一句查询文本」融合出的向量做近似最近邻检索
    6. 把检索结果拼成一张全景图保存并弹窗展示
    7. 释放并删除 Collection（Demo 收尾，连数据一起删掉）

说明：本文件所有路径都是相对路径，必须在 code/C3 目录下运行，
      否则会找不到 ../../models 和 ../../data。
"""

import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# 1. 初始化设置
# ---------------------------------------------------------------------------
# 基础模型：Visualized-BGE 是在 bge-base-en-v1.5 之上扩展出来的多模态版本，
# 它本身不训练新的文本编码器，而是复用 BGE 主干，只在图像侧加了一个
# "visual token" 分支，因此图像向量和文本向量天然处在同一语义空间 ——
# 这正是可以用一句话去搜图的前提。
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# 微调权重（Visualized-BGE 为图像分支训练出来的参数），必须和 MODEL_NAME 配套，
# 维度不匹配会直接报 shape 错误。
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"

# 数据目录，下面应有 dragon/ 子目录，里面放待入库的 .png 图像和 query.png
DATA_DIR = "../../data/C3"

# 集合名。Milvus 的 Collection 类似关系数据库中的表，
# 同一 Collection 内所有向量的维度必须完全一致。
COLLECTION_NAME = "multimodal_demo"

# Milvus 服务地址。这里连的是远程实例，不是本地 Docker，
# 因此需要注意网络可达性和该实例的鉴权配置。
MILVUS_URI = "http://106.55.165.169:19530"


# ---------------------------------------------------------------------------
# 2. 定义工具（编码器 + 可视化函数）
# ---------------------------------------------------------------------------
class Encoder:
    """编码器类，用于将图像和文本编码为向量。

    封装 Visualized_BGE，对外只暴露"图/文 -> list[float]"这一种能力，
    这样后面调用方完全不需要关心 torch 和模型内部结构。
    """

    def __init__(self, model_name: str, model_path: str):
        # 加载 BGE 主干 + 图像分支的微调权重
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        # eval() 切换到推理模式：关闭 Dropout / BatchNorm 的训练期行为，
        # 保证同一个输入每次编码得到的向量一致（检索场景这点非常关键）
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        """编码「图像 + 文本」的联合查询。

        模型会把图像分支和文本分支的输出融合成一个向量，
        所以这一个向量同时携带了"图中的样子"和"文字描述"两种意图，
        用它在库里检索会比单模态查询更准。
        """
        # no_grad 关闭自动求导，省内存也更快（推理不需要梯度）
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        # encode 返回的是 [batch, dim] 的 2D tensor，这里只取了第 0 条，
        # 所以是 .tolist()[0] -> 一维 list，正好是 Milvus 要求的插入格式
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        """只编码图像（纯图搜图时用）—— 入库阶段用的就是这个方法。"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300,
                      img_width: int = 300, row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。

    布局说明（最终返回 np.hstack 左右拼接的两块）：
        ┌───────────┬───────────────────────┐
        │           │  0   │  1   │  2        │
        │  Query    │  3   │  4   │  5        │  <- 检索结果按 row_count 列排成网格
        │ (左下角)  │  6   │  7   │  8        │
        └───────────┴───────────────────────┘
    左列是一个与右图等高的空白画布，查询图被贴在它的最后一行（底部）。
    """
    # 右侧结果网格的尺寸：row_count 列 -> 宽 = 单元格宽 * 列数，
    # 高也按 row_count 估算，因此最多能容纳 row_count * row_count 张图
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count

    # 右侧结果画布，初值 255(白)，dtype 必须 uint8 才能直接写 PNG
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    # 左侧查询图画布，宽度只占 1 格
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # ------- 处理查询图像 -------
    # 统一转 RGB，避免带透明通道（RGBA）或灰度图导致后续 resize 出问题
    query_pil = Image.open(query_image_path).convert("RGB")
    # PIL 是 RGB 顺序，OpenCV 是 BGR 顺序，[:, :, ::-1] 就是通道反转，做转换
    query_cv = np.array(query_pil)[:, :, ::-1]
    # 缩放到单元格尺寸，保证和右侧结果图大小一致
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    # 加一圈 10px 边框。注意 OpenCV 用的是 BGR，所以 (255, 0, 0) 是【蓝色】而不是红色，
    # 这是给查询图做的视觉强调
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(255, 0, 0))
    # 只把左画布的最后一行换成查询图（bordered_query 已经带边框，这里再 resize 一次
    # 是为了把它压回单元格尺寸，避免边框把画布撑大）
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(
        bordered_query, (img_width, img_height))
    # 在左下角写上 "Query" 标签。(10, panoramic_height-20) 是文字基线的像素坐标
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ------- 处理检索到的图像 -------
    for i, img_path in enumerate(retrieved_images):
        # 一维序号 i 换算成二维网格的行列：第 i 张图在第 i//row_count 行、第 i%row_count 列
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        # 比单元格各小 4px，再补 2px 黑色边框，正好填满单元格——
        # 这样做出来的 2px 黑边就是图与图之间的分隔线
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # 用切片把这张图"盖章"到结果画布的对应格子里
        panoramic_image[start_row:start_row + img_height,
                        start_col:start_col + img_width] = bordered_retrieved

        # 添加索引号（红色，(0,0,255) 在 BGR 下才是红），方便对照控制台打印的 Top N
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 左右两块等高，hstack 横向拼接后就是最终的全景图
    return np.hstack([query_display_area, panoramic_image])


# ---------------------------------------------------------------------------
# 3. 初始化客户端
# ---------------------------------------------------------------------------
print("--> 正在初始化编码器和Milvus客户端...")
# 先造编码器（要加载 .pth，比较慢），再连 Milvus
encoder = Encoder(MODEL_NAME, MODEL_PATH)
# MilvusClient 是 pymilvus 提供的高层 ORM 风格客户端，
# 比老式的 connections/Collection 写法简洁，适合 Demo
milvus_client = MilvusClient(uri=MILVUS_URI)


# ---------------------------------------------------------------------------
# 4. 创建 Milvus Collection
# ---------------------------------------------------------------------------
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
# 幂等处理：这个 Demo 每次都要重新灌数据，若同名 Collection 已存在就先删掉。
# 注意 drop_collection 是【物理删除】，里面的向量和标量数据会一起没了。
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 收集待入库的图片。glob 不支持 Windows 反斜杠，所以统一用 os.path.join 拼路径
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")

# 向量维度由模型决定（bge-base 是 768），这里不写死而是真实编码一张图去取长度，
# 这样以后换模型、换维度都不用改代码，也不会出现"维度对不上"的建表错误。
dim = len(encoder.encode_image(image_list[0]))

# 字段定义。Milvus 是强 schema 的，建表时字段、类型、维度都必须明确声明
fields = [
    # 主键字段，设置自增 (auto_id=True)：插入时不需要（也不允许）由我们指定 id，
    # Milvus 会自动分配一个全局唯一的 INT64 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 向量字段，维度与模型的输出向量维度一致；
    # 注意向量字段不建索引就无法高效检索，索引在第 6 步创建
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    # 存储原图像路径的标量字段，最大长度限制 512 字符。
    # 需要留意：Milvus 的 VARCHAR max_length 按【字节】计算，
    # 含中文的路径会更早触顶，实际项目里建议留足余量
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 创建集合 Schema：可以理解为"表结构"，由字段列表 + 描述组成
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

# 创建集合。这里只建了"表结构"，还没设索引，所以还不能建索引前的高效检索
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))


# ---------------------------------------------------------------------------
# 5. 准备并插入数据
# ---------------------------------------------------------------------------
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
# tqdm 给这个循环加进度条，因为逐张编码图像很慢（每张都要过一次 Transformer）
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    # 每张图编码成一个向量；维度必须与 schema 里 vector 字段声明的 dim 一致
    vector = encoder.encode_image(image_path)
    # 用 dict 形式插入，key 必须与字段名对应。
    # id 是 auto_id，所以这里刻意不提供
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    # 批量插入。Milvus 插入后数据先进入可增长的 segment，
    # 此时能查到但还不是索引结构；真正的倒排/图索引在第 6 步创建
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")


# ---------------------------------------------------------------------------
# 6. 创建索引
# ---------------------------------------------------------------------------
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
# prepare_index_params() 返回一个索引参数构造器，可以往里 add 多个索引
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    # HNSW：分层可导航小世界图，召回率高、查询快，代价是占内存多、建索引慢，
    # 是向量检索里最常用的索引类型
    index_type="HNSW",
    # COSINE：按余弦相似度度量。BGE 系列向量做过归一化，用 IP 和 COSINE 等价，
    # 但 COSINE 语义更直观（只看方向不看模长）
    metric_type="COSINE",
    params={
        # M：每个节点在图中的最大邻居数。越大召回越高、内存越大。
        # 16 是官方推荐的通用起点
        "M": 16,
        # efConstruction：建索引时每层的候选队列长度。
        # 越大建图质量越好（召回更高），但建索引更慢。256 属于偏"高质量"的取值
        "efConstruction": 256,
    }
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))

# Milvus 是"存算分离"的：数据在磁盘上，查询前必须把 Collection
# 加载(load)到查询节点的内存里，向量索引才能被使用。
# 不 load 直接 search，低版本会报错或退化成全表暴力扫描
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")


# ---------------------------------------------------------------------------
# 7. 执行多模态检索
# ---------------------------------------------------------------------------
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
# 查询文本用英文 "一条龙" 的英文表述。Visualized-BGE 的文本主干是英文模型
# （bge-base-en），查询文本用英文效果明显更好
query_text = "一条龙"
# 图文联合编码出【一个】查询向量，一次检索同时受图像和文本影响
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    # data 是二维列表：一次可以传多个查询向量做批量检索。
    # 返回值同样按查询分组，所以下面用 [0] 取第一个（也是唯一一个）查询的结果
    data=[query_vector],
    # output_fields 指定除了距离之外还要回带哪些标量字段。
    # 不回带的字段不会出现在结果里，这里要的是图片路径，方便后面读图
    output_fields=["image_path"],
    # limit 即 Top-K，返回最相似的 5 条
    limit=5,
    search_params={
        "metric_type": "COSINE",
        # ef：HNSW 查询时候选队列长度，必须 >= limit。
        # 越大召回越高但越慢；128 对 5 条结果的场景算是很宽松的设置
        "params": {"ef": 128}
    }
)[0]

retrieved_images = []
print("检索结果:")
for i, hit in enumerate(search_results):
    # hit 结构：{"id": 主键, "distance": 相似度分数, "entity": {标量字段...}}
    # 注意 COSINE 度量下 Milvus 返回的 distance 其实是【余弦相似度】，
    # 越接近 1 越相似（不是"距离越小越好"），打印时不要按误差来理解
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])


# ---------------------------------------------------------------------------
# 8. 可视化与清理
# ---------------------------------------------------------------------------
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    # 拼成"左查询图 + 右结果网格"的全景图
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    # cv2.imwrite 写盘用的是 BGR 顺序，而我们在函数里已经统一成 BGR，所以颜色不会串
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    # 调系统默认看图软件弹窗显示（阻塞式，关掉窗口才继续往下走）
    Image.open(combined_image_path).show()

# 释放 Collection：把索引数据从查询节点内存里卸载，GPU/内存紧张时很重要
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
# 删除 Collection：连同 Schema、索引和全部数据一起物理删除。
# Demo 收尾用，真实业务里千万不要放在脚本末尾这么执行
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")
