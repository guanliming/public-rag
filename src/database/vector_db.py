# 向量数据库模块
"""
向量数据库模块
=============

使用 PostgreSQL + pgvector 作为向量数据库，
实现文档向量的存储、检索和管理功能。

pgvector 是 PostgreSQL 的一个扩展，支持：
- 向量类型存储
- 相似度检索（L2 距离、内积、余弦相似度）
- 向量索引加速检索

设计模式：
- 使用单例模式管理数据库连接
- 使用配置类封装数据库参数
- 提供统一的接口进行向量操作
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from dotenv import load_dotenv

# 加载环境变量
# 优先加载 .env.local（包含敏感信息，不会提交到 git）
# 然后加载 .env（作为默认值）
_env_local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env.local')
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

if os.path.exists(_env_local_path):
    load_dotenv(_env_local_path, override=True)

if os.path.exists(_env_path):
    load_dotenv(_env_path, override=False)


@dataclass
class DatabaseConfig:
    """
    数据库配置类

    使用数据类封装数据库连接参数，
    支持从环境变量读取配置。

    属性:
        host: 数据库主机地址
        port: 数据库端口
        user: 数据库用户名
        password: 数据库密码
        db_name: 数据库名称
        table_name: 向量存储表名
        embedding_dimension: 嵌入向量维度（默认 1024，适用于阿里百炼 text-embedding-v3）
    """

    # 基础连接参数
    host: str
    port: int
    user: str
    password: str
    db_name: str

    # 向量存储相关
    table_name: str = "rag_documents"
    embedding_dimension: int = 1024

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        从环境变量创建配置实例

        读取 .env 文件中的环境变量，
        构建数据库配置对象。

        Returns:
            DatabaseConfig: 数据库配置实例

        示例:
            >>> config = DatabaseConfig.from_env()
            >>> print(config.host)
            '172.19.249.222'
        """
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            db_name=os.getenv("DB_NAME", "rag_db"),
            table_name=os.getenv("VECTOR_TABLE_NAME", "rag_documents"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        )

    @property
    def connection_string(self) -> str:
        """
        生成 psycopg2 连接字符串

        Returns:
            str: 数据库连接字符串

        示例:
            >>> config = DatabaseConfig(...)
            >>> print(config.connection_string)
            'host=172.19.249.222 port=5432 dbname=rag_db user=postgres password=postgres'
        """
        return (
            f"host={self.host} "
            f"port={self.port} "
            f"dbname={self.db_name} "
            f"user={self.user} "
            f"password={self.password}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        """
        生成 SQLAlchemy 连接 URL

        用于 LangChain PGVector 初始化。

        Returns:
            str: SQLAlchemy 风格的连接 URL

        示例:
            >>> config = DatabaseConfig(...)
            >>> print(config.sqlalchemy_url)
            'postgresql+psycopg2://postgres:postgres@172.19.249.222:5432/rag_db'
        """
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db_name}"
        )


class DatabaseManager:
    """
    数据库管理类

    负责 PostgreSQL 数据库的连接管理、
    扩展安装和表结构创建。

    使用上下文管理器模式确保连接正确关闭。

    属性:
        config: 数据库配置
        _connection: 当前数据库连接（如果有）
    """

    def __init__(self, config: DatabaseConfig):
        """
        初始化数据库管理器

        Args:
            config: DatabaseConfig 配置实例
        """
        self.config = config
        self._connection = None

    @contextmanager
    def get_connection(self):
        """
        获取数据库连接的上下文管理器

        使用 `with` 语句自动管理连接的创建和关闭。

        Yields:
            psycopg2.extensions.connection: 数据库连接对象

        示例:
            >>> manager = DatabaseManager(config)
            >>> with manager.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT 1")
        """
        conn = None
        try:
            conn = psycopg2.connect(self.config.connection_string)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def create_database_if_not_exists(self) -> bool:
        """
        创建数据库（如果不存在）

        由于 PostgreSQL 不支持在事务中创建数据库，
        需要先连接到默认的 'postgres' 数据库，
        然后创建目标数据库。

        Returns:
            bool: True 表示数据库创建成功或已存在

        注意:
            此方法需要具有创建数据库权限的用户
        """
        # 连接到默认的 postgres 数据库
        default_conn_string = (
            f"host={self.config.host} "
            f"port={self.config.port} "
            f"dbname=postgres "
            f"user={self.config.user} "
            f"password={self.config.password}"
        )

        conn = None
        try:
            conn = psycopg2.connect(default_conn_string)
            conn.autocommit = True
            cursor = conn.cursor()

            # 检查数据库是否存在
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config.db_name,)
            )

            if not cursor.fetchone():
                # 创建数据库
                cursor.execute(f'CREATE DATABASE {self.config.db_name}')
                print(f"数据库 {self.config.db_name} 创建成功")
            else:
                print(f"数据库 {self.config.db_name} 已存在")

            return True

        except Exception as e:
            print(f"创建数据库时出错: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def enable_vector_extension(self) -> bool:
        """
        启用 pgvector 扩展

        pgvector 是 PostgreSQL 的向量扩展，
        需要先安装并启用才能使用向量类型。

        Returns:
            bool: True 表示扩展启用成功

        注意:
            需要超级用户权限来创建扩展
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 检查并创建 vector 扩展
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()

                # 验证扩展是否安装
                cursor.execute(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                )
                result = cursor.fetchone()

                if result:
                    print("pgvector 扩展已启用")
                    return True
                else:
                    print("pgvector 扩展安装失败")
                    return False

        except Exception as e:
            print(f"启用 pgvector 扩展时出错: {e}")
            return False

    def test_connection(self) -> bool:
        """
        测试数据库连接

        Returns:
            bool: True 表示连接成功
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                version = cursor.fetchone()
                print(f"数据库连接成功，PostgreSQL 版本: {version[0]}")
                return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False


class VectorStore:
    """
    向量存储类

    封装 LangChain PGVector，提供统一的向量操作接口。

    主要功能:
    - 文档向量化存储
    - 相似度检索
    - 文档管理（增删改查）

    设计模式:
    - 使用组合模式封装 PGVector
    - 提供统一的 API 接口
    - 支持多种相似度检索策略
    """

    def __init__(
        self,
        config: DatabaseConfig,
        embedding_model,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ):
        """
        初始化向量存储

        Args:
            config: 数据库配置
            embedding_model: 嵌入模型（需要实现 embed_documents 和 embed_query 方法）
            distance_strategy: 相似度计算策略，默认使用余弦相似度

        可用的距离策略:
            - COSINE: 余弦相似度（最常用，适合文本相似度）
            - EUCLIDEAN: 欧氏距离
            - MAX_INNER_PRODUCT: 最大内积
        """
        self.config = config
        self.embedding_model = embedding_model
        self.distance_strategy = distance_strategy
        self._vector_store = None

    def _get_vector_store(self) -> PGVector:
        """
        获取或初始化 PGVector 实例

        使用延迟初始化模式，只有在真正需要时才创建连接。

        Returns:
            PGVector: LangChain PGVector 实例

        注意:
            PGVector 会自动创建必要的表结构
        """
        if self._vector_store is None:
            self._vector_store = PGVector(
                connection=self.config.sqlalchemy_url,
                embedding=self.embedding_model,
                collection_name=self.config.table_name,
                distance_strategy=self.distance_strategy,
            )
        return self._vector_store

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加文档到向量存储

        支持批量添加文档，自动进行向量化并存储到数据库。

        Args:
            documents: Document 对象列表，每个 Document 包含 page_content 和 metadata
            ids: 可选的自定义 ID 列表，如果不提供则自动生成 UUID

        Returns:
            List[str]: 生成的文档 ID 列表

        示例:
            >>> from langchain_core.documents import Document
            >>> docs = [Document(page_content="这是测试文本", metadata={"source": "test.txt"})]
            >>> ids = vector_store.add_documents(docs)
            >>> print(ids)
            ['xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx']
        """
        vector_store = self._get_vector_store()
        return vector_store.add_documents(documents=documents, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        相似度检索

        根据查询文本，在向量数据库中检索最相似的文档片段。

        Args:
            query: 查询文本
            k: 返回的最相似文档数量，默认 4
            filter: 可选的元数据过滤条件

        Returns:
            List[Document]: 按相似度排序的 Document 列表

        示例:
            >>> results = vector_store.similarity_search("机器学习是什么", k=3)
            >>> for doc in results:
            ...     print(doc.page_content[:100])
        """
        vector_store = self._get_vector_store()
        return vector_store.similarity_search(query=query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        相似度检索（带分数）

        与 similarity_search 类似，但额外返回相似度分数。

        Args:
            query: 查询文本
            k: 返回的最相似文档数量
            filter: 可选的元数据过滤条件

        Returns:
            List[tuple[Document, float]]: (Document, 相似度分数) 元组列表

        注意:
            分数的含义取决于距离策略:
            - COSINE: 分数范围 [0, 1]，越接近 1 越相似
            - EUCLIDEAN: 分数越小越相似
        """
        vector_store = self._get_vector_store()
        return vector_store.similarity_search_with_score(
            query=query, k=k, filter=filter
        )

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        删除指定 ID 的文档

        Args:
            ids: 要删除的文档 ID 列表

        示例:
            >>> vector_store.delete(["id1", "id2"])
        """
        vector_store = self._get_vector_store()
        if ids:
            vector_store.delete(ids=ids)

    def search(
        self,
        query: str,
        search_type: str = "similarity",
        k: int = 4,
        **kwargs,
    ) -> List[Document]:
        """
        通用搜索接口

        支持多种搜索类型，是 similarity_search 的扩展版本。

        Args:
            query: 查询文本
            search_type: 搜索类型，可选值:
                - "similarity": 相似度搜索（默认）
                - "similarity_score_threshold": 带分数阈值的相似度搜索
                - "mmr": 最大边际相关性搜索
            k: 返回结果数量
            **kwargs: 额外参数，取决于 search_type

        Returns:
            List[Document]: 搜索结果文档列表

        示例:
            >>> # MMR 搜索（多样性搜索）
            >>> results = vector_store.search(
            ...     query="机器学习",
            ...     search_type="mmr",
            ...     k=5,
            ...     fetch_k=20
            ... )
        """
        vector_store = self._get_vector_store()
        return vector_store.search(query=query, search_type=search_type, k=k, **kwargs)

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        获取检索器

        返回 LangChain Retriever 接口，可用于 RAG 链。

        Args:
            search_type: 搜索类型
            search_kwargs: 搜索参数字典

        Returns:
            Retriever: LangChain 检索器对象

        示例:
            >>> retriever = vector_store.get_retriever(search_kwargs={"k": 5})
            >>> from langchain.chains import RetrievalQA
            >>> qa_chain = RetrievalQA.from_chain_type(
            ...     llm=llm,
            ...     chain_type="stuff",
            ...     retriever=retriever
            ... )
        """
        vector_store = self._get_vector_store()
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {},
        )


def init_database(config: DatabaseConfig) -> tuple[DatabaseManager, VectorStore]:
    """
    初始化数据库

    这是一个便捷函数，执行以下步骤:
    1. 创建数据库（如果不存在）
    2. 测试数据库连接
    3. 启用 pgvector 扩展

    Args:
        config: 数据库配置

    Returns:
        tuple[DatabaseManager, VectorStore]: 数据库管理器和向量存储实例

    注意:
        此函数需要在导入 embedding 模块后才能获取 embedding_model，
        实际使用时需要传入 embedding_model 给 VectorStore。
        此函数返回的 VectorStore 不包含 embedding_model，需要手动设置。

    示例:
        >>> config = DatabaseConfig.from_env()
        >>> db_manager, _ = init_database(config)
        >>> # 然后创建带 embedding_model 的 VectorStore
        >>> vector_store = VectorStore(config, embedding_model)
    """
    db_manager = DatabaseManager(config)

    # 创建数据库
    if not db_manager.create_database_if_not_exists():
        raise RuntimeError("数据库创建失败")

    # 测试连接
    if not db_manager.test_connection():
        raise RuntimeError("数据库连接测试失败")

    # 启用 pgvector 扩展
    if not db_manager.enable_vector_extension():
        print("警告: pgvector 扩展可能需要手动启用")

    # 创建一个临时的 VectorStore（无 embedding_model）
    # 实际使用时需要传入 embedding_model
    vector_store = VectorStore(config, embedding_model=None)

    return db_manager, vector_store
