# 向量化模块
"""
向量化模块
=========

负责将文本转换为向量表示，
是 RAG 系统的核心组件之一。

主要组件：
- EmbeddingFactory: 嵌入模型工厂，创建嵌入模型实例
- EmbeddingConfig: 嵌入模型配置
- get_embedding_model: 便捷函数获取嵌入模型

支持的嵌入模型：
1. 本地模型 (sentence-transformers)
   - 优点：免费、离线可用、速度快
   - 缺点：效果可能不如商业模型
   - 推荐模型：all-MiniLM-L6-v2（轻量、快速）

2. OpenAI 嵌入模型
   - 优点：效果好、语义理解能力强
   - 缺点：需要 API Key、有费用、需要网络
   - 推荐模型：text-embedding-ada-002

依赖：
- sentence-transformers: 本地嵌入模型
- langchain-openai: OpenAI 嵌入集成
- numpy: 向量计算
"""

from src.embedding.embedder import (
    EmbeddingConfig,
    EmbeddingFactory,
    get_embedding_model,
    EmbeddingModelType,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingFactory",
    "get_embedding_model",
    "EmbeddingModelType",
]
