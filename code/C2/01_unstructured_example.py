import sys

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# Windows 控制台默认是 GBK，PDF 中的特殊字符（如康熙部首 ⺊）打印时会抛
# UnicodeEncodeError，这里统一用 UTF-8 输出，无法编码的字符用 ? 代替。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
# elements = partition(
#     filename=pdf_path,
#     content_type="application/pdf"
# )


elements = partition_pdf(
    filename=pdf_path,
    content_type="application/pdf",
    strategy='hi_res',
    languages=["chi_sim", "eng"],  # OCR 语言，不指定会默认只用英文，中文会识别成乱码
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)