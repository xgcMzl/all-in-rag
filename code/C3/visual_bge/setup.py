"""兼容用的 setup.py。

包的元数据已经全部搬到 pyproject.toml，这里只保留一个空壳，
让仍然按老方式（`pip install -e .`）调用 setup.py 的工具不会报错。
真正的配置见同目录的 pyproject.toml。
"""

from setuptools import setup

setup()
