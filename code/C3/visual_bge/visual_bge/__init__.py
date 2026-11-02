"""Visualized-BGE：在 BGE 文本编码器之上增加视觉分支的多模态编码包。

图像和文本都会被编码到同一个向量空间，因此可以用文本搜图、用图搜图，
也可以把「图 + 文」融合成一个查询向量。
"""

from .modeling import Visualized_BGE

__all__ = ["Visualized_BGE"]
