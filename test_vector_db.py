#!/usr/bin/env python3
"""
向量数据库测试脚本
用于诊断向量存储问题
"""

import os
import sys
from dotenv import load_dotenv

env_local_path = os.path.join(os.path.dirname(__file__), '.env.local')
env_path = os.path.join(os.path.dirname(__file__), '.env')

if os.path.exists(env_local_path):
    load_dotenv(env_local_path, override=True)
if os.path.exists(env_path):
    load_dotenv(env_path, override=False)

from langchain_core.documents import Document
from src.database.vector_db import DatabaseConfig, VectorStore
from src.embedding.embedder import get_embedding_model

def test_direct_connection():
    """
    测试直接数据库连接，查看表结构
    """
    print("=" * 60)
    print("测试1: 直接数据库连接测试")
    print("=" * 60)
    
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    config = DatabaseConfig.from_env()
    print(f"数据库配置:")
    print(f"  主机: {config.host}")
    print(f"  端口: {config.port}")
    print(f"  数据库: {config.db_name}")
    print(f"  表名: {config.table_name}")
    print(f"  嵌入维度: {config.embedding_dimension}")
    
    try:
        conn = psycopg2.connect(config.connection_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n检查数据库中的所有表:")
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        table_names = [t['table_name'] for t in tables]
        print(f"  表列表: {table_names}")
        
        for table in table_names:
            print(f"\n  --- 表 {table} 的结构:")
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, (table,))
            columns = cursor.fetchall()
            for col in columns:
                print(f"    {col['column_name']}: {col['data_type']}")
            
            try:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                count = cursor.fetchone()['cnt']
                print(f"  记录数: {count}")
            except Exception as e:
                print(f"  查询记录数失败: {e}")
        
        if 'langchain_pg_collection' in table_names:
            print("\n检查 langchain_pg_collection 表内容:")
            cursor.execute("SELECT * FROM langchain_pg_collection")
            collections = cursor.fetchall()
            for col in collections:
                print(f"  {dict(col)}")
        
        conn.close()
        print("\n直接连接测试完成")
        return table_names
        
    except Exception as e:
        print(f"直接连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_pgvector_add_documents():
    """
    测试 PGVector 添加文档
    """
    print("\n" + "=" * 60)
    print("测试2: PGVector 添加文档测试")
    print("=" * 60)
    
    config = DatabaseConfig.from_env()
    
    print("加载嵌入模型...")
    embedding_model = get_embedding_model()
    
    print("创建 VectorStore...")
    vector_store = VectorStore(config, embedding_model)
    
    test_docs = [
        Document(
            page_content=f"这是测试文档第{i}。包含一些中文内容用于测试向量存储功能。",
            metadata={"source": f"test_{i}.txt", "test": True, "index": i}
        )
        for i in range(3)
    ]
    
    print(f"\n准备添加 {len(test_docs)} 个测试文档...")
    
    try:
        print("调用 add_documents...")
        ids = vector_store.add_documents(test_docs)
        print(f"add_documents 返回了 {len(ids)} 个 ID: {ids}")
        
        print("\n立即验证存储结果...")
        print("方式1: 使用 verify_documents_stored")
        verification = vector_store.verify_documents_stored(ids)
        print(f"  验证结果: {verification}")
        
        print("\n方式2: 直接查询数据库")
        import psycopg2
        conn = psycopg2.connect(config.connection_string)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        count = cursor.fetchone()[0]
        print(f"  langchain_pg_embedding 表记录数: {count}")
        
        cursor.execute("SELECT * FROM langchain_pg_embedding LIMIT 5")
        rows = cursor.fetchall()
        print(f"  前5条记录:")
        for row in rows:
            print(f"    ID: {row[0]}, collection_id: {row[1]}")
        
        cursor.execute("SELECT * FROM langchain_pg_collection")
        collections = cursor.fetchall()
        print(f"\n  langchain_pg_collection 表内容:")
        for col in collections:
            print(f"    {col}")
        
        conn.close()
        
        print("\n方式3: 测试相似度搜索验证")
        results = vector_store.similarity_search("测试文档", k=5)
        print(f"  搜索到 {len(results)} 个结果")
        for i, doc in enumerate(results):
            print(f"    结果{i}: {doc.page_content[:50]}...")
        
        return ids
        
    except Exception as e:
        print(f"PGVector 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_transaction_commit():
    """
    测试事务提交问题
    """
    print("\n" + "=" * 60)
    print("测试3: 事务提交测试")
    print("=" * 60)
    
    config = DatabaseConfig.from_env()
    
    print("加载嵌入模型...")
    embedding_model = get_embedding_model()
    
    print("创建 VectorStore...")
    vector_store = VectorStore(config, embedding_model)
    
    test_doc = Document(
        page_content="事务提交测试文档。这是用于测试事务是否正确提交的内容。",
        metadata={"source": "transaction_test.txt", "test_transaction": True}
    )
    
    try:
        print("添加文档...")
        ids = vector_store.add_documents([test_doc])
        print(f"返回 ID: {ids}")
        
        print("\n在另一个连接中查询...")
        import psycopg2
        
        conn1 = psycopg2.connect(config.connection_string)
        cursor1 = conn1.cursor()
        cursor1.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        count1 = cursor1.fetchone()[0]
        print(f"  连接1查询到的记录数: {count1}")
        conn1.close()
        
        print("\n等待2秒后再次查询...")
        import time
        time.sleep(2)
        
        conn2 = psycopg2.connect(config.connection_string)
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        count2 = cursor2.fetchone()[0]
        print(f"  连接2查询到的记录数: {count2}")
        
        cursor2.execute("SELECT id, document FROM langchain_pg_embedding WHERE document LIKE '%事务提交%'")
        rows = cursor2.fetchall()
        print(f"  包含'事务提交'的记录数: {len(rows)}")
        for row in rows:
            print(f"    ID: {row[0]}, 内容: {row[1][:50]}...")
        
        conn2.close()
        
        return True
        
    except Exception as e:
        print(f"事务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_dimension():
    """
    测试嵌入维度
    """
    print("\n" + "=" * 60)
    print("测试4: 嵌入维度测试")
    print("=" * 60)
    
    print("加载嵌入模型...")
    embedding_model = get_embedding_model()
    
    test_text = "这是一个测试文本，用于检查嵌入维度。"
    
    try:
        print("获取查询嵌入...")
        query_embedding = embedding_model.embed_query(test_text)
        print(f"  查询嵌入维度: {len(query_embedding)}")
        
        print("获取文档嵌入...")
        doc_embeddings = embedding_model.embed_documents([test_text, "另一个测试文本"])
        print(f"  文档嵌入数量: {len(doc_embeddings)}")
        if doc_embeddings:
            print(f"  文档嵌入维度: {len(doc_embeddings[0])}")
        
        config = DatabaseConfig.from_env()
        print(f"\n配置的嵌入维度: {config.embedding_dimension}")
        
        if len(query_embedding) != config.embedding_dimension:
            print(f"\n警告: 实际嵌入维度 ({len(query_embedding)}) 与配置 ({config.embedding_dimension}) 不一致!")
            print("这可能导致向量存储失败!")
        
        return len(query_embedding)
        
    except Exception as e:
        print(f"嵌入维度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("向量数据库问题诊断工具")
    print("=" * 60)
    
    tables = test_direct_connection()
    
    if not tables:
        print("\n警告: 无法连接数据库或没有表，先测试嵌入维度...")
        test_embedding_dimension()
        return
    
    dimension = test_embedding_dimension()
    
    ids = test_pgvector_add_documents()
    
    if ids:
        test_transaction_commit()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
