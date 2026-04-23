# 文档处理模块
"""
文档处理模块
===========

负责文档的加载、解析和切片处理。

主要组件：
- DocumentLoader: 文档加载器，支持多种格式
- TextSplitter: 文本切片器，将长文本切分成合适大小的片段
- DocumentProcessor: 文档处理器，整合加载和切片功能

支持的文档格式：
- PDF (.pdf)
- Markdown (.md, .markdown)
- 纯文本 (.txt)

依赖：
- pypdf: PDF 文档解析
- pymupdf: 高性能 PDF 解析（备用）
- langchain: LangChain 文档加载器和文本分割器
"""

from src.document.processor import (
    DocumentLoader,
    TextSplitter,
    DocumentProcessor,
    SupportedFormats,
)

__all__ = [
    "DocumentLoader",
    "TextSplitter",
    "DocumentProcessor",
    "SupportedFormats",
]
