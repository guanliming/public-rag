#!/usr/bin/env python3
"""
简化的向量数据库诊断脚本
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

import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_core.documents import Document

from src.database.vector_db import DatabaseConfig, VectorStore
from src.embedding.embedder import get_embedding_model

def check_db_tables():
    """检查数据库表结构"""
    print("=" * 60)
    print("步骤1: 检查数据库表结构")
    print("=" * 60)
    
    config = DatabaseConfig.from_env()
    
    conn = psycopg2.connect(config.connection_string)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print(f"\n数据库连接信息:")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Database: {config.db_name}")
    print(f"  Table name (config): {config.table_name}")
    
    print("\n检查所有表:")
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = cursor.fetchall()
    table_names = [t['table_name'] for t in tables]
    print(f"  表列表: {table_names}")
    
    for table in table_names:
        print(f"\n  --- 表 {table} ---")
        cursor.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s 
            ORDER BY ordinal_position
        """, (table,))
        cols = cursor.fetchall()
        for c in cols:
            print(f"    {c['column_name']}: {c['data_type']}")
        
        try:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            cnt = cursor.fetchone()['cnt']
            print(f"    记录数: {cnt}")
        except Exception as e:
            print(f"    记录数: 查询失败 - {e}")
    
    if 'langchain_pg_collection' in table_names:
        print("\n检查集合表内容:")
        cursor.execute("SELECT * FROM langchain_pg_collection")
        collections = cursor.fetchall()
        for col in collections:
            print(f"  {dict(col)}")
    
    conn.close()
    return table_names

def check_embedding_dimension():
    """检查嵌入维度"""
    print("\n" + "=" * 60)
    print("步骤2: 检查嵌入模型和维度")
    print("=" * 60)
    
    config = DatabaseConfig.from_env()
    print(f"\n配置的嵌入维度: {config.embedding_dimension}")
    
    print("\n加载嵌入模型...")
    embedding_model = get_embedding_model()
    
    test_text = "测试文本用于检查嵌入维度"
    
    print("生成测试嵌入...")
    query_emb = embedding_model.embed_query(test_text)
    doc_emb = embedding_model.embed_documents([test_text])
    
    actual_dim = len(query_emb)
    print(f"\n实际嵌入维度: {actual_dim}")
    print(f"文档嵌入维度: {len(doc_emb[0]) if doc_emb else 'N/A'}")
    
    if actual_dim != config.embedding_dimension:
        print(f"\n⚠️  警告: 嵌入维度不匹配!")
        print(f"   配置维度: {config.embedding_dimension}")
        print(f"   实际维度: {actual_dim}")
        print(f"\n这可能导致向量存储失败，因为 pgvector 表的向量维度是固定的!")
    
    return actual_dim

def test_vector_store_direct():
    """直接测试向量存储"""
    print("\n" + "=" * 60)
    print("步骤3: 直接测试向量存储")
    print("=" * 60)
    
    config = DatabaseConfig.from_env()
    embedding_model = get_embedding_model()
    
    print("\n创建 VectorStore...")
    vector_store = VectorStore(config, embedding_model)
    
    test_docs = [
        Document(
            page_content=f"诊断测试文档{i}。这是用于测试向量存储是否正常工作的内容。",
            metadata={"source": "diagnostic_test.txt", "test_run": True, "index": i}
        )
        for i in range(2)
    ]
    
    print(f"\n准备存储 {len(test_docs)} 个文档...")
    
    print("\n调用 add_documents...")
    ids = vector_store.add_documents(test_docs)
    print(f"返回的 ID 数量: {len(ids)}")
    print(f"返回的 IDs: {ids}")
    
    print("\n立即使用 SQL 检查数据库...")
    conn = psycopg2.connect(config.connection_string)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
    count = cursor.fetchone()[0]
    print(f"langchain_pg_embedding 记录数: {count}")
    
    print(f"\n检查是否有我们刚插入的文档 (ID 匹配)...")
    placeholders = ','.join(['%s'] * len(ids))
    cursor.execute(f"SELECT id, document FROM langchain_pg_embedding WHERE id IN ({placeholders})", ids)
    matching = cursor.fetchall()
    print(f"找到匹配 ID 的记录数: {len(matching)}")
    
    if len(matching) == len(ids):
        print("✅ 所有 ID 都在数据库中找到!")
    else:
        print(f"❌ 缺失 {len(ids) - len(matching)} 条记录!")
        print(f"   期望 ID: {ids}")
        print(f"   找到的 ID: {[m[0] for m in matching]}")
    
    print("\n检查集合关联...")
    cursor.execute("SELECT * FROM langchain_pg_collection")
    collections = cursor.fetchall()
    print(f"集合数量: {len(collections)}")
    for col in collections:
        print(f"  集合: {col}")
        coll_id = col[0]
        cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s", (coll_id,))
        coll_count = cursor.fetchone()[0]
        print(f"  该集合下的记录数: {coll_count}")
    
    conn.close()
    
    print("\n测试相似度搜索...")
    results = vector_store.similarity_search("诊断测试文档", k=5)
    print(f"搜索结果数量: {len(results)}")
    for i, doc in enumerate(results):
        print(f"  结果{i}: {doc.page_content[:60]}...")
    
    return len(ids), len(matching)

def main():
    print("=" * 60)
    print("向量数据库问题诊断")
    print("=" * 60)
    
    tables = check_db_tables()
    
    if not tables:
        print("\n错误: 无法连接到数据库或没有表")
        return
    
    dimension = check_embedding_dimension()
    
    expected_count, actual_count = test_vector_store_direct()
    
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    print(f"\n存储测试结果:")
    print(f"  期望存储数量: {expected_count}")
    print(f"  实际数据库数量: {actual_count}")
    
    if expected_count == actual_count:
        print("✅ 存储正常，数量匹配!")
    else:
        print("❌ 存储异常，数量不匹配!")
        print("\n可能的原因:")
        print("  1. 嵌入维度不匹配导致插入失败")
        print("  2. 事务没有正确提交")
        print("  3. PGVector 内部错误")

if __name__ == "__main__":
    main()
