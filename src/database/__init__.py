# 数据库模块
"""
数据库模块
=========

负责向量数据库的初始化、连接和操作。

主要组件：
- VectorStore: 向量存储类，封装 pgvector 操作
- DatabaseConfig: 数据库配置类
- init_database: 初始化数据库连接

依赖：
- langchain-postgres: LangChain 的 PostgreSQL 向量存储集成
- psycopg2-binary: PostgreSQL 数据库驱动
- pgvector: PostgreSQL 向量扩展支持
"""

from src.database.vector_db import VectorStore, DatabaseConfig, init_database

__all__ = ["VectorStore", "DatabaseConfig", "init_database"]
