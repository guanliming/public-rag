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
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from src.config import load_config

# 加载配置（支持 settings.yaml 和环境变量）
load_config()


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
        self._actual_embedding_dimension = None
        
        self._detect_embedding_dimension()
        self._validate_vector_table()
    
    def _detect_embedding_dimension(self) -> int:
        """
        检测实际的嵌入维度
        
        生成一个测试嵌入来检测实际的嵌入维度，
        确保与数据库表的维度匹配。
        
        Returns:
            int: 实际的嵌入维度
        """
        if self._actual_embedding_dimension is not None:
            return self._actual_embedding_dimension
        
        try:
            print("检测嵌入模型的实际维度...")
            test_embedding = self.embedding_model.embed_query("测试文本用于检测维度")
            self._actual_embedding_dimension = len(test_embedding)
            
            config_dim = self.config.embedding_dimension
            print(f"  实际嵌入维度: {self._actual_embedding_dimension}")
            print(f"  配置的维度: {config_dim}")
            
            if self._actual_embedding_dimension != config_dim:
                print(f"\n⚠️  警告: 嵌入维度不匹配!")
                print(f"   配置维度 (EMBEDDING_DIMENSION): {config_dim}")
                print(f"   实际维度: {self._actual_embedding_dimension}")
                print(f"\n   这可能导致向量存储失败!")
                print(f"   建议更新 .env 文件: EMBEDDING_DIMENSION={self._actual_embedding_dimension}")
            
            return self._actual_embedding_dimension
            
        except Exception as e:
            print(f"检测嵌入维度时出错: {e}")
            self._actual_embedding_dimension = self.config.embedding_dimension
            return self._actual_embedding_dimension
    
    def _should_auto_recreate(self) -> bool:
        """
        检查是否启用了自动重建表的开关
        
        从环境变量读取 AUTO_RECREATE_TABLE 配置。
        
        Returns:
            bool: True 表示启用自动重建
        """
        auto_recreate = os.getenv("AUTO_RECREATE_TABLE", "false").lower()
        return auto_recreate in ("true", "1", "yes", "on")
    
    def _drop_vector_tables(self) -> bool:
        """
        删除向量相关的表
        
        删除 langchain_pg_embedding 和 langchain_pg_collection 表，
        让 PGVector 自动重新创建以匹配新的嵌入维度。
        
        Returns:
            bool: 是否成功删除
        """
        try:
            import psycopg2
            
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = [t[0] for t in cursor.fetchall()]
                print(f"数据库中的表: {tables}")
                
                dropped = []
                
                if 'langchain_pg_embedding' in tables:
                    try:
                        cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")
                        conn.commit()
                        dropped.append('langchain_pg_embedding')
                        print("已删除表: langchain_pg_embedding")
                    except Exception as e:
                        print(f"删除 langchain_pg_embedding 失败: {e}")
                
                if 'langchain_pg_collection' in tables:
                    try:
                        cursor.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
                        conn.commit()
                        dropped.append('langchain_pg_collection')
                        print("已删除表: langchain_pg_collection")
                    except Exception as e:
                        print(f"删除 langchain_pg_collection 失败: {e}")
                
                if dropped:
                    print(f"\n✅ 已删除向量表，下次使用时将自动重建以匹配当前嵌入模型的维度")
                    return True
                else:
                    print("没有需要删除的向量表")
                    return False
                    
        except Exception as e:
            print(f"删除向量表时出错: {e}")
            return False
    
    def _check_table_schema_compatibility(self, cursor) -> tuple[bool, str]:
        """
        检查表结构是否与当前版本的 PGVector 兼容
        
        检查关键列是否存在，如 custom_id 等。
        
        Args:
            cursor: 数据库游标
            
        Returns:
            tuple[bool, str]: (是否兼容, 不兼容的原因)
        """
        try:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'langchain_pg_embedding' 
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cursor.fetchall()]
            print(f"  表 langchain_pg_embedding 的列: {columns}")
            
            required_columns = ['collection_id', 'embedding', 'document', 'cmetadata', 'custom_id', 'uuid']
            
            missing_columns = []
            for col in required_columns:
                if col not in columns:
                    missing_columns.append(col)
            
            if missing_columns:
                return False, f"缺少关键列: {missing_columns}"
            
            return True, "表结构兼容"
            
        except Exception as e:
            return False, f"检查表结构时出错: {e}"
    
    def _validate_vector_table(self) -> None:
        """
        验证向量表的维度和结构是否匹配
        
        检查数据库中已存在的向量表的维度和结构，
        如果维度不匹配或结构不兼容且启用了自动重建开关，
        则自动删除表让系统重建。
        """
        try:
            import psycopg2
            
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'langchain_pg_embedding'
                """)
                
                if not cursor.fetchone():
                    print("向量表不存在，将在首次使用时创建")
                    return
                
                print("\n" + "=" * 70)
                print("验证现有向量表...")
                print("=" * 70)
                
                print("\n1. 检查表结构兼容性...")
                schema_compatible, schema_message = self._check_table_schema_compatibility(cursor)
                print(f"   结果: {schema_message}")
                
                print("\n2. 检查向量维度...")
                dimension_ok = True
                table_dim = None
                actual_dim = self._actual_embedding_dimension or self.config.embedding_dimension
                
                try:
                    cursor.execute("""
                        SELECT atttypmod 
                        FROM pg_attribute 
                        WHERE attrelid = 'langchain_pg_embedding'::regclass 
                        AND attname = 'embedding'
                    """)
                    result = cursor.fetchone()
                    
                    if result and result[0] > -1:
                        table_dim = result[0]
                        print(f"   表中向量维度: {table_dim}")
                        print(f"   实际嵌入维度: {actual_dim}")
                        
                        if table_dim != actual_dim:
                            dimension_ok = False
                    else:
                        print("   无法检测表中向量维度")
                        
                except Exception as e:
                    print(f"   检查维度时出错: {e}")
                
                need_recreate = False
                reasons = []
                
                if not schema_compatible:
                    need_recreate = True
                    reasons.append(f"表结构不兼容: {schema_message}")
                
                if not dimension_ok and table_dim is not None:
                    need_recreate = True
                    reasons.append(f"维度不匹配: 表中 {table_dim} 维 vs 实际 {actual_dim} 维")
                
                if need_recreate:
                    print("\n" + "!" * 70)
                    print("⚠️  检测到表需要重建!")
                    for reason in reasons:
                        print(f"   - {reason}")
                    print(f"\n   这会导致向量插入失败!")
                    
                    if self._should_auto_recreate():
                        print(f"\n✅ 自动重建表开关已开启，准备删除现有表...")
                        self._drop_vector_tables()
                        print(f"✅ 表已删除，下次操作时将自动重建以匹配当前配置")
                        print(f"!" * 70)
                    else:
                        print(f"\n   解决方案:")
                        print(f"   1. 在 settings.yaml 中设置 database.auto_recreate_table: true")
                        print(f"      或在 .env 中设置 AUTO_RECREATE_TABLE=true")
                        print(f"   2. 然后重启应用")
                        print(f"!" * 70)
                    
                    self._recreate_vector_table = True
                else:
                    print("\n✅ 向量表验证通过:")
                    print(f"   - 表结构兼容")
                    if table_dim:
                        print(f"   - 维度匹配: {table_dim}")
                    print("=" * 70)
                    
        except Exception as e:
            print(f"验证向量表时出错: {e}")
            import traceback
            traceback.print_exc()

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
                connection_string=self.config.sqlalchemy_url,
                embedding_function=self.embedding_model,
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
        result_ids = vector_store.add_documents(documents=documents, ids=ids)
        
        print(f"PGVector.add_documents 返回了 {len(result_ids)} 个 ID")
        print(f"调用 _force_session_flush 确保数据写入...")
        self._force_session_flush()
        
        return result_ids
    
    def _force_session_flush(self) -> None:
        """
        强制刷新 SQLAlchemy 会话，确保数据写入数据库
        
        PGVector 使用 SQLAlchemy 管理会话，
        这个方法确保会话被正确提交或关闭，
        使其他数据库连接能够看到写入的数据。
        """
        try:
            if self._vector_store is not None:
                if hasattr(self._vector_store, '_embedder'):
                    embedder = self._vector_store._embedder
                    if hasattr(embedder, 'client'):
                        client = embedder.client
                        if hasattr(client, 'commit'):
                            client.commit()
                            print("已提交 SQLAlchemy 会话")
                        elif hasattr(client, 'close'):
                            client.close()
                            print("已关闭 SQLAlchemy 会话")
                
                if hasattr(self._vector_store, '_conn'):
                    conn = self._vector_store._conn
                    if hasattr(conn, 'commit'):
                        conn.commit()
                        print("已提交连接")
                    elif hasattr(conn, 'close'):
                        conn.close()
                        print("已关闭连接")
                
                if hasattr(self._vector_store, 'Collection'):
                    try:
                        from sqlalchemy.orm import sessionmaker
                        if hasattr(self._vector_store, '_session'):
                            session = self._vector_store._session
                            if session:
                                session.commit()
                                print("已提交会话")
                    except Exception as e:
                        print(f"会话提交尝试失败（非致命）: {e}")
        
        except Exception as e:
            print(f"强制刷新会话时出错: {e}")

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

    def delete_documents_by_filename(self, filename: str) -> int:
        """
        根据文件名删除所有相关的文档向量

        用于实现同名文件覆盖功能：上传同名文件前，
        先删除该文件之前存储的所有向量数据。

        Args:
            filename: 文件名（如 "document.pdf" 或 "test.txt"）

        Returns:
            int: 删除的文档数量

        示例:
            >>> count = vector_store.delete_documents_by_filename("test.pdf")
            >>> print(f"已删除 {count} 条记录")
        """
        try:
            import psycopg2
            import json
            
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'langchain_pg_embedding'
                """)
                if not cursor.fetchone():
                    print(f"表 langchain_pg_embedding 不存在，无需删除")
                    return 0
                
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'langchain_pg_embedding' AND table_schema = 'public'
                """)
                columns = [row[0] for row in cursor.fetchall()]
                
                if 'cmetadata' not in columns:
                    print(f"表 langchain_pg_embedding 中没有 cmetadata 列")
                    return 0
                
                cursor.execute("""
                    SELECT COUNT(*) FROM langchain_pg_embedding 
                    WHERE cmetadata->>'file_name' = %s
                """, (filename,))
                count_result = cursor.fetchone()
                count = count_result[0] if count_result else 0
                
                if count == 0:
                    print(f"数据库中没有找到文件名 {filename} 的记录")
                    return 0
                
                cursor.execute("""
                    DELETE FROM langchain_pg_embedding 
                    WHERE cmetadata->>'file_name' = %s
                """, (filename,))
                conn.commit()
                
                print(f"✅ 已删除文件名 {filename} 的 {count} 条向量记录")
                return count
                
        except Exception as e:
            print(f"按文件名删除文档时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def get_all_filenames(self) -> List[str]:
        """
        获取向量数据库中所有已存储的文件名列表

        用于知识库管理，查看已上传了哪些文件。

        Returns:
            List[str]: 去重后的文件名列表

        示例:
            >>> filenames = vector_store.get_all_filenames()
            >>> print(filenames)
            ['document1.pdf', 'test.txt', ...]
        """
        try:
            import psycopg2
            
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'langchain_pg_embedding'
                """)
                if not cursor.fetchone():
                    return []
                
                cursor.execute("""
                    SELECT DISTINCT cmetadata->>'file_name' as filename
                    FROM langchain_pg_embedding
                    WHERE cmetadata->>'file_name' IS NOT NULL
                    ORDER BY filename
                """)
                
                filenames = [row[0] for row in cursor.fetchall() if row[0]]
                print(f"数据库中共有 {len(filenames)} 个不同的文件")
                return filenames
                
        except Exception as e:
            print(f"获取文件名列表时出错: {e}")
            return []

    def get_documents_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        """
        根据文件名获取该文件的所有文档信息

        用于验证和调试，查看某个文件在数据库中的存储情况。

        Args:
            filename: 文件名

        Returns:
            List[Dict]: 文档信息列表，包含 id、内容预览、元数据等
        """
        try:
            import psycopg2
            import json
            
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'langchain_pg_embedding'
                """)
                if not cursor.fetchone():
                    return []
                
                cursor.execute("""
                    SELECT uuid, document, cmetadata, custom_id
                    FROM langchain_pg_embedding
                    WHERE cmetadata->>'file_name' = %s
                    ORDER BY (cmetadata->>'chunk_index')::int ASC
                """, (filename,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': str(row['uuid']) if row['uuid'] else row.get('custom_id'),
                        'content_preview': row['document'][:200] if row['document'] else '',
                        'metadata': dict(row['cmetadata']) if row['cmetadata'] else {},
                    })
                
                print(f"找到 {len(results)} 条 {filename} 的记录")
                return results
                
        except Exception as e:
            print(f"按文件名获取文档时出错: {e}")
            return []

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

    def get_document_count(self) -> int:
        """
        获取向量数据库中的文档数量

        直接查询数据库表中的记录数，用于验证数据是否成功存储。

        Returns:
            int: 文档数量

        示例:
            >>> count = vector_store.get_document_count()
            >>> print(f"向量数据库中有 {count} 条记录")
        """
        try:
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor()

                # 首先检查数据库中有哪些表
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = cursor.fetchall()
                print(f"数据库中的表: {[t[0] for t in tables]}")

                # 尝试查找可能的向量表
                possible_tables = ['langchain_pg_embedding', 'rag_documents']
                for table_name in possible_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        result = cursor.fetchone()
                        print(f"表 {table_name} 中的记录数: {result[0]}")
                        if result[0] > 0:
                            return result[0]
                    except Exception as e:
                        print(f"查询表 {table_name} 失败: {e}")

                # 如果都没找到，返回 0
                return 0
        except Exception as e:
            print(f"获取文档数量时出错: {e}")
            return -1

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        返回当前集合的详细统计信息，包括文档数量、集合信息等。

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        try:
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = cursor.fetchall()
                table_names = [t['table_name'] for t in tables]
                print(f"数据库中的表: {table_names}")

                total_count = 0
                collection_count = 0
                collection_id = None

                if 'langchain_pg_embedding' in table_names:
                    try:
                        cursor.execute("SELECT COUNT(*) as cnt FROM langchain_pg_embedding")
                        result = cursor.fetchone()
                        total_count = result['cnt'] if result and 'cnt' in result else 0
                        print(f"langchain_pg_embedding 总记录数: {total_count}")
                    except Exception as e:
                        print(f"查询 langchain_pg_embedding 失败: {e}")
                        import traceback
                        traceback.print_exc()

                if 'langchain_pg_collection' in table_names:
                    try:
                        cursor.execute("""
                            SELECT column_name, data_type FROM information_schema.columns
                            WHERE table_name = 'langchain_pg_collection'
                            ORDER BY ordinal_position
                        """)
                        column_info = cursor.fetchall()
                        columns = [c['column_name'] for c in column_info]
                        print(f"langchain_pg_collection 列: {columns}")

                        if 'name' in columns:
                            id_col = None
                            id_type = None
                            
                            for col in column_info:
                                col_name = col['column_name']
                                if col_name == 'id':
                                    id_col = 'id'
                                    id_type = col['data_type']
                                    break
                                elif col_name == 'uuid':
                                    id_col = 'uuid'
                                    id_type = col['data_type']
                                    break
                            
                            if not id_col:
                                id_col = columns[0] if columns else 'name'
                                id_type = None
                            
                            cursor.execute(
                                f"SELECT {id_col}, name FROM langchain_pg_collection WHERE name = %s",
                                (self.config.table_name,)
                            )
                            collection = cursor.fetchone()
                            print(f"查询集合 {self.config.table_name}: {collection}")

                            if collection:
                                collection_id = collection[id_col] if id_col in collection else None
                                collection_id_str = str(collection_id) if collection_id else None
                                
                                print(f"集合 ID: {collection_id}, 类型: {type(collection_id)}")
                                
                                try:
                                    if id_type == 'uuid' and collection_id_str:
                                        cursor.execute(
                                            "SELECT COUNT(*) as cnt FROM langchain_pg_embedding WHERE collection_id::text = %s",
                                            (collection_id_str,)
                                        )
                                    else:
                                        cursor.execute(
                                            "SELECT COUNT(*) as cnt FROM langchain_pg_embedding WHERE collection_id = %s",
                                            (collection_id,)
                                        )
                                    result = cursor.fetchone()
                                    collection_count = result['cnt'] if result and 'cnt' in result else 0
                                    print(f"集合 {self.config.table_name} 的记录数: {collection_count}")
                                except Exception as e:
                                    print(f"查询集合记录数失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                    except Exception as e:
                        print(f"查询 langchain_pg_collection 失败: {e}")
                        import traceback
                        traceback.print_exc()

                other_count = 0
                if 'rag_documents' in table_names:
                    try:
                        cursor.execute("SELECT COUNT(*) as cnt FROM rag_documents")
                        result = cursor.fetchone()
                        other_count = result['cnt'] if result and 'cnt' in result else 0
                        print(f"rag_documents 表记录数: {other_count}")
                    except Exception as e:
                        print(f"rag_documents 表不存在: {e}")

                return {
                    "total_documents": total_count,
                    "collection_documents": collection_count,
                    "collection_name": self.config.table_name,
                    "collection_id": collection_id,
                    "table_exists": True,
                    "other_count": other_count,
                    "tables": table_names,
                }
        except Exception as e:
            print(f"获取集合统计时出错: {e}")
            return {
                "total_documents": -1,
                "collection_documents": -1,
                "collection_name": self.config.table_name,
                "collection_id": None,
                "table_exists": False,
                "error": str(e),
            }

    def verify_documents_stored(self, expected_ids: List[str] = None) -> Dict[str, Any]:
        """
        验证文档是否成功存储到向量数据库

        Args:
            expected_ids: 期望存储的文档 ID 列表（可选）

        Returns:
            Dict[str, Any]: 验证结果，包含：
                - success: 是否验证成功
                - stored_count: 实际存储的数量
                - expected_count: 期望的数量（如果提供了 expected_ids）
                - missing_ids: 缺失的文档 ID（如果提供了 expected_ids）
                - message: 验证消息
        """
        stats = self.get_collection_stats()

        total_count = stats.get("total_documents", 0)
        collection_count = stats.get("collection_documents", 0)
        other_count = stats.get("other_count", 0)

        stored_count = 0
        if collection_count > 0:
            stored_count = collection_count
        elif total_count > 0:
            stored_count = total_count
        elif other_count > 0:
            stored_count = other_count

        if expected_ids and len(expected_ids) > 0:
            expected_count = len(expected_ids)
            
            if stored_count == expected_count:
                result = {
                    "success": True,
                    "stored_count": stored_count,
                    "expected_count": expected_count,
                    "missing_ids": [],
                    "message": f"验证成功：期望存储 {expected_count} 条，集合中实际有 {stored_count} 条",
                }
            elif stored_count > 0:
                result = {
                    "success": False,
                    "stored_count": stored_count,
                    "expected_count": expected_count,
                    "missing_ids": [],
                    "message": f"验证失败：期望 {expected_count} 条，集合中实际有 {stored_count} 条",
                }
            else:
                result = {
                    "success": False,
                    "stored_count": 0,
                    "expected_count": expected_count,
                    "missing_ids": expected_ids,
                    "message": f"验证失败：没有文档存储到向量数据库（期望 {expected_count} 条）",
                }
        else:
            result = {
                "success": stored_count > 0,
                "stored_count": stored_count,
                "expected_count": None,
                "missing_ids": [],
                "message": f"验证成功：向量数据库中有 {stored_count} 条记录" if stored_count > 0 else "验证失败：向量数据库中没有记录",
            }

        return result

    def clear_collection(self) -> bool:
        """
        清空当前集合中的所有文档

        Returns:
            bool: 是否成功清空
        """
        try:
            with psycopg2.connect(self.config.connection_string) as conn:
                cursor = conn.cursor()

                # 首先获取数据库中的表
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = [t[0] for t in cursor.fetchall()]
                print(f"数据库中的表: {tables}")

                # 尝试多种方式清空

                # 方式 1: langchain_pg_collection + langchain_pg_embedding
                if 'langchain_pg_collection' in tables and 'langchain_pg_embedding' in tables:
                    try:
                        cursor.execute("""
                            SELECT column_name FROM information_schema.columns
                            WHERE table_name = 'langchain_pg_collection'
                            ORDER BY ordinal_position
                        """)
                        columns = [c[0] for c in cursor.fetchall()]

                        id_col = None
                        if 'id' in columns:
                            id_col = 'id'
                        elif 'uuid' in columns:
                            id_col = 'uuid'
                        else:
                            id_col = columns[0] if columns else 'name'

                        cursor.execute(
                            f"SELECT {id_col} FROM langchain_pg_collection WHERE name = %s",
                            (self.config.table_name,)
                        )
                        collection = cursor.fetchone()

                        if collection:
                            collection_id = str(collection[0])
                            cursor.execute(
                                "DELETE FROM langchain_pg_embedding WHERE collection_id = %s",
                                (collection_id,)
                            )
                            conn.commit()
                            print(f"已清空集合 {self.config.table_name} 中的所有文档 (方式 1)")
                            return True
                    except Exception as e:
                        print(f"方式 1 清空失败: {e}")

                # 方式 2: 直接清空 langchain_pg_embedding 表
                if 'langchain_pg_embedding' in tables:
                    try:
                        cursor.execute("DELETE FROM langchain_pg_embedding")
                        conn.commit()
                        print("已清空 langchain_pg_embedding 表中的所有文档 (方式 2)")
                        return True
                    except Exception as e:
                        print(f"方式 2 清空失败: {e}")

                # 方式 3: 清空 rag_documents 表
                if 'rag_documents' in tables:
                    try:
                        cursor.execute("DELETE FROM rag_documents")
                        conn.commit()
                        print("已清空 rag_documents 表中的所有文档 (方式 3)")
                        return True
                    except Exception as e:
                        print(f"方式 3 清空失败: {e}")

                print(f"集合 {self.config.table_name} 不存在或无法清空")
                return False
        except Exception as e:
            print(f"清空集合时出错: {e}")
            return False


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
