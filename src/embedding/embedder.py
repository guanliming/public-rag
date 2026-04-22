# 向量化模块
"""
向量化模块
============

实现文本向量化功能，将文本转换为高维向量表示。

什么是嵌入（Embedding）？
嵌入是将文本（或其他离散数据）转换为连续向量空间中的点的过程。
相似的文本在向量空间中距离较近，不相似的文本距离较远。

例如：
- "猫" 和 "狗" 都是动物，向量距离较近
- "猫" 和 "汽车" 向量距离较远

嵌入的作用：
1. 语义相似度计算
2. 文档检索
3. 聚类分析
4. 分类任务

向量维度：
- 本地模型 all-MiniLM-L6-v2: 384 维
- OpenAI text-embedding-ada-002: 1536 维
"""

import os
from enum import Enum
from typing import Optional, List, Any
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class EmbeddingModelType(Enum):
    """
    嵌入模型类型枚举

    定义支持的嵌入模型类型。

    属性:
        LOCAL: 本地模型（sentence-transformers）
        OPENAI: OpenAI 嵌入模型
    """

    LOCAL = "local"
    OPENAI = "openai"


@dataclass
class EmbeddingConfig:
    """
    嵌入模型配置类

    封装嵌入模型的配置参数，
    支持从环境变量读取配置。

    属性:
        model_type: 模型类型（LOCAL 或 OPENAI）
        model_name: 模型名称
        openai_api_key: OpenAI API Key（仅 OpenAI 模型需要）
        device: 运行设备（"cpu" 或 "cuda"，仅本地模型）
    """

    # 模型类型
    model_type: EmbeddingModelType = EmbeddingModelType.LOCAL

    # 模型名称
    model_name: str = "all-MiniLM-L6-v2"

    # OpenAI 相关配置
    openai_api_key: Optional[str] = None

    # 本地模型相关配置
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """
        从环境变量创建配置实例

        读取 .env 文件中的环境变量，
        构建嵌入模型配置对象。

        Returns:
            EmbeddingConfig: 嵌入模型配置实例

        环境变量:
            EMBEDDING_MODEL_TYPE: "local" 或 "openai"
            EMBEDDING_MODEL_NAME: 模型名称
            OPENAI_API_KEY: OpenAI API Key
            EMBEDDING_DEVICE: 运行设备（cpu/cuda）
        """
        # 读取模型类型
        model_type_str = os.getenv("EMBEDDING_MODEL_TYPE", "local").lower()
        if model_type_str == "openai":
            model_type = EmbeddingModelType.OPENAI
        else:
            model_type = EmbeddingModelType.LOCAL

        # 读取模型名称
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

        # 读取 OpenAI API Key
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # 读取设备配置
        device = os.getenv("EMBEDDING_DEVICE", "cpu")

        return cls(
            model_type=model_type,
            model_name=model_name,
            openai_api_key=openai_api_key,
            device=device,
        )


class EmbeddingFactory:
    """
    嵌入模型工厂

    使用工厂模式创建不同类型的嵌入模型实例。

    设计模式：
    - 工厂模式：根据配置创建不同类型的产品（嵌入模型）
    - 单例模式：同一配置只创建一个实例

    优点：
    - 封装模型创建细节
    - 统一接口，便于切换模型
    - 支持延迟加载
    """

    # 缓存已创建的实例
    _instances: dict = {}

    @classmethod
    def create(
        cls,
        config: Optional[EmbeddingConfig] = None,
    ) -> Embeddings:
        """
        创建嵌入模型实例

        根据配置创建对应的嵌入模型，
        支持本地模型和 OpenAI 模型。

        Args:
            config: 嵌入模型配置，如果为 None 则从环境变量读取

        Returns:
            Embeddings: LangChain Embeddings 接口实例

        Raises:
            ValueError: 模型类型不支持或配置错误

        示例:
            >>> config = EmbeddingConfig(model_type=EmbeddingModelType.LOCAL)
            >>> embedding = EmbeddingFactory.create(config)
            >>> vector = embedding.embed_query("你好")
            >>> print(len(vector))
            384
        """
        # 如果没有提供配置，从环境变量读取
        if config is None:
            config = EmbeddingConfig.from_env()

        # 检查缓存
        cache_key = (config.model_type.value, config.model_name)
        if cache_key in cls._instances:
            print(f"使用缓存的嵌入模型: {config.model_type.value} - {config.model_name}")
            return cls._instances[cache_key]

        # 根据模型类型创建实例
        if config.model_type == EmbeddingModelType.LOCAL:
            embedding = cls._create_local_embedding(config)
        elif config.model_type == EmbeddingModelType.OPENAI:
            embedding = cls._create_openai_embedding(config)
        else:
            raise ValueError(f"不支持的嵌入模型类型: {config.model_type}")

        # 缓存实例
        cls._instances[cache_key] = embedding

        print(f"嵌入模型创建成功: {config.model_type.value} - {config.model_name}")
        return embedding

    @classmethod
    def _create_local_embedding(cls, config: EmbeddingConfig) -> Embeddings:
        """
        创建本地嵌入模型

        使用 sentence-transformers 库加载本地模型。

        Args:
            config: 嵌入模型配置

        Returns:
            Embeddings: LangChain HuggingFaceEmbeddings 实例

        注意:
            首次使用时会自动下载模型到本地缓存目录。
            模型下载地址: ~/.cache/huggingface/hub/

        推荐的本地模型:
            - all-MiniLM-L6-v2: 轻量快速，384 维（推荐测试用）
            - all-mpnet-base-v2: 效果更好，768 维
            - paraphrase-multilingual-MiniLM-L12-v2: 多语言支持
        """
        try:
            # 延迟导入，只有在使用本地模型时才需要安装
            from langchain_community.embeddings import HuggingFaceEmbeddings

            print(f"正在加载本地嵌入模型: {config.model_name}...")
            print(f"运行设备: {config.device}")

            # 配置模型参数
            model_kwargs = {"device": config.device}

            # 配置编码参数
            encode_kwargs = {
                "normalize_embeddings": True,  # 归一化向量，便于余弦相似度计算
            }

            # 创建嵌入模型
            embedding = HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

            print(f"本地模型加载完成: {config.model_name}")
            return embedding

        except ImportError as e:
            raise ImportError(
                "使用本地嵌入模型需要安装 sentence-transformers。"
                "请运行: pip install sentence-transformers"
            ) from e

    @classmethod
    def _create_openai_embedding(cls, config: EmbeddingConfig) -> Embeddings:
        """
        创建 OpenAI 嵌入模型

        使用 OpenAI API 进行向量化。

        Args:
            config: 嵌入模型配置

        Returns:
            Embeddings: LangChain OpenAIEmbeddings 实例

        注意:
            - 需要有效的 OpenAI API Key
            - 会产生 API 调用费用
            - 需要网络连接

        推荐的 OpenAI 模型:
            - text-embedding-ada-002: 最新、最经济的嵌入模型，1536 维
            - text-embedding-3-small: 新版小模型，1536 维
            - text-embedding-3-large: 新版大模型，3072 维
        """
        try:
            # 延迟导入
            from langchain_openai import OpenAIEmbeddings

            # 检查 API Key
            api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "使用 OpenAI 嵌入模型需要设置 OPENAI_API_KEY 环境变量。"
                    "请在 .env 文件中添加: OPENAI_API_KEY=your-api-key"
                )

            print(f"正在初始化 OpenAI 嵌入模型: {config.model_name}...")

            # 创建 OpenAI 嵌入模型
            embedding = OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=api_key,
            )

            print(f"OpenAI 嵌入模型初始化完成: {config.model_name}")
            return embedding

        except ImportError as e:
            raise ImportError(
                "使用 OpenAI 嵌入模型需要安装 langchain-openai。"
                "请运行: pip install langchain-openai"
            ) from e

    @classmethod
    def clear_cache(cls) -> None:
        """
        清空模型缓存

        用于强制重新创建模型实例。
        """
        cls._instances.clear()
        print("嵌入模型缓存已清空")


def get_embedding_model(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    获取嵌入模型的便捷函数

    这是对外的主要接口，简化模型创建过程。

    Args:
        model_type: 模型类型，"local" 或 "openai"
        model_name: 模型名称

    Returns:
        Embeddings: LangChain Embeddings 接口实例

    示例:
        >>> # 使用默认配置（从环境变量读取）
        >>> embedding = get_embedding_model()
        >>>
        >>> # 显式指定使用本地模型
        >>> embedding = get_embedding_model(model_type="local")
        >>>
        >>> # 指定特定模型
        >>> embedding = get_embedding_model(
        ...     model_type="local",
        ...     model_name="all-mpnet-base-v2"
        ... )
    """
    # 从环境变量获取默认配置
    config = EmbeddingConfig.from_env()

    # 覆盖配置（如果提供了参数）
    if model_type:
        if model_type.lower() == "openai":
            config.model_type = EmbeddingModelType.OPENAI
        else:
            config.model_type = EmbeddingModelType.LOCAL

    if model_name:
        config.model_name = model_name

    # 创建并返回嵌入模型
    return EmbeddingFactory.create(config)


# 一些常用嵌入模型的维度信息
EMBEDDING_DIMENSIONS = {
    # 本地模型 (sentence-transformers)
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    # OpenAI 模型
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def get_embedding_dimension(model_name: str) -> int:
    """
    获取嵌入模型的维度

    知道向量维度对于配置 pgvector 很重要。

    Args:
        model_name: 模型名称

    Returns:
        int: 向量维度，如果未知返回 384（默认）

    示例:
        >>> get_embedding_dimension("all-MiniLM-L6-v2")
        384
        >>> get_embedding_dimension("text-embedding-ada-002")
        1536
    """
    return EMBEDDING_DIMENSIONS.get(model_name, 384)


class EmbeddingUtils:
    """
    嵌入工具类

    提供一些常用的嵌入相关的工具函数。
    """

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        计算两个向量的余弦相似度

        余弦相似度 = (v1 · v2) / (|v1| * |v2|)

        值范围: [-1, 1]
        - 1: 完全相同
        - 0: 正交（无关）
        - -1: 完全相反

        Args:
            v1: 第一个向量
            v2: 第二个向量

        Returns:
            float: 余弦相似度

        示例:
            >>> v1 = [1, 0, 0]
            >>> v2 = [0, 1, 0]
            >>> EmbeddingUtils.cosine_similarity(v1, v2)
            0.0
        """
        import math

        # 计算点积
        dot_product = sum(a * b for a, b in zip(v1, v2))

        # 计算模长
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(a * a for a in v2))

        # 避免除零
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

    @staticmethod
    def euclidean_distance(v1: List[float], v2: List[float]) -> float:
        """
        计算两个向量的欧氏距离

        欧氏距离 = sqrt(Σ(vi - ui)^2)

        Args:
            v1: 第一个向量
            v2: 第二个向量

        Returns:
            float: 欧氏距离
        """
        import math

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    @staticmethod
    def normalize_vector(v: List[float]) -> List[float]:
        """
        归一化向量

        将向量转换为单位向量（模长为 1），
        这样余弦相似度就等于点积。

        Args:
            v: 原始向量

        Returns:
            List[float]: 归一化后的向量
        """
        import math

        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return v
        return [x / norm for x in v]
