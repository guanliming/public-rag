#!/usr/bin/env python3
"""
向量数据库问题深度诊断脚本
"""

import os
import sys
import uuid
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

from src.database.vector_db import DatabaseConfig, VectorStore, DatabaseManager
from src.embedding.embedder import get_embedding_model, get_embedding_dimension, EMBEDDING_DIMENSIONS, EmbeddingConfig

def check_embedding_config():
    """检查嵌入模型配置"""
    print("=" * 70)
    print("步骤1: 检查嵌入模型配置")
    print("=" * 70)
    
    config = EmbeddingConfig.from_env()
    print(f"\n模型类型: {config.model_type.value}")
    print(f"模型名称: {config.model_name}")
    print(f"设备: {config.device}")
    
    db_config = DatabaseConfig.from_env()
    print(f"\n数据库配置的嵌入维度: {db_config.embedding_dimension}")
    
    known_dim = EMBEDDING_DIMENSIONS.get(config.model_name)
    if known_dim:
        print(f"已知模型 {config.model_name} 的维度: {known_dim}")
    else:
        print(f"未知模型 {config.model_name}，将实际测试维度")
    
    print("\n加载嵌入模型...")
    embedding_model = get_embedding_model()
    
    test_texts = [
        "测试文本1，用于检查嵌入维度。",
        "测试文本2，这是另一个测试。"
    ]
    
    print("\n测试嵌入生成...")
    query_emb = embedding_model.embed_query(test_texts[0])
    doc_embs = embedding_model.embed_documents(test_texts)
    
    actual_dim = len(query_emb)
    print(f"\n实际嵌入维度: {actual_dim}")
    
    config_dim = db_config.embedding_dimension
    if actual_dim != config_dim:
        print(f"\n⚠️  严重问题: 嵌入维度不匹配!")
        print(f"   配置维度 (EMBEDDING_DIMENSION): {config_dim}")
        print(f"   实际维度: {actual_dim}")
        print(f"\n这会导致向量存储失败，因为 pgvector 表的向量维度是固定的!")
        print(f"需要更新 .env 文件中的 EMBEDDING_DIMENSION={actual_dim}")
    else:
        print(f"\n✅ 嵌入维度匹配: {actual_dim}")
    
    return actual_dim, config_dim

def check_database_tables():
    """检查数据库表结构和维度"""
    print("\n" + "=" * 70)
    print("步骤2: 检查数据库表结构")
    print("=" * 70)
    
    config = DatabaseConfig.from_env()
    
    conn = psycopg2.connect(config.connection_string)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print(f"\n检查所有表...")
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = cursor.fetchall()
    table_names = [t['table_name'] for t in tables]
    print(f"表列表: {table_names}")
    
    if 'langchain_pg_embedding' in table_names:
        print(f"\n--- 表 langchain_pg_embedding 的结构 ---")
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'langchain_pg_embedding'
            ORDER BY ordinal_position
        """)
        cols = cursor.fetchall()
        for c in cols:
            col_info = f"  {c['column_name']}: {c['data_type']}"
            if c['character_maximum_length']:
                col_info += f"({c['character_maximum_length']})"
            col_info += f" [nullable: {c['is_nullable']}]"
            print(col_info)
        
        cursor.execute("SELECT COUNT(*) as cnt FROM langchain_pg_embedding")
        cnt = cursor.fetchone()['cnt']
        print(f"\n当前记录数: {cnt}")
        
        print("\n检查向量列的维度...")
        try:
            cursor.execute("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'langchain_pg_embedding'::regclass 
                AND attname = 'embedding'
            """)
            result = cursor.fetchone()
            if result:
                atttypmod = result['atttypmod']
                if atttypmod > -1:
                    print(f"向量列维度 (从 atttypmod): {atttypmod}")
                else:
                    print("向量列维度: 未指定 (atttypmod = -1)")
                    
                    cursor.execute("""
                        SELECT pg_typeof(embedding) as vector_type 
                        FROM langchain_pg_embedding 
                        LIMIT 1
                    """)
                    type_result = cursor.fetchone()
                    if type_result:
                        print(f"向量列类型: {type_result['vector_type']}")
                    
        except Exception as e:
            print(f"检查向量维度时出错: {e}")
    
    if 'langchain_pg_collection' in table_names:
        print(f"\n--- 表 langchain_pg_collection 的内容 ---")
        cursor.execute("SELECT * FROM langchain_pg_collection")
        collections = cursor.fetchall()
        for col in collections:
            print(f"  {dict(col)}")
    
    conn.close()
    return table_names

def test_direct_vector_insert():
    """直接测试向量插入，绕过 PGVector"""
    print("\n" + "=" * 70)
    print("步骤3: 直接测试向量插入 (绕过 PGVector)")
    print("=" * 70)
    
    config = DatabaseConfig.from_env()
    embedding_model = get_embedding_model()
    
    test_docs = [
        Document(
            page_content=f"直接插入测试文档{i}",
            metadata={"source": "direct_insert_test.txt", "test": True}
        )
        for i in range(2)
    ]
    
    print(f"\n生成测试嵌入...")
    texts = [doc.page_content for doc in test_docs]
    embeddings = embedding_model.embed_documents(texts)
    
    print(f"嵌入维度: {len(embeddings[0])}")
    embedding_dim = len(embeddings[0])
    
    conn = psycopg2.connect(config.connection_string)
    cursor = conn.cursor()
    
    print(f"\n检查集合...")
    collection_name = config.table_name
    
    cursor.execute("SELECT id FROM langchain_pg_collection WHERE name = %s", (collection_name,))
    coll_result = cursor.fetchone()
    
    if coll_result:
        collection_id = coll_result[0]
        print(f"找到集合 '{collection_name}', ID: {collection_id}")
    else:
        print(f"创建集合 '{collection_name}'...")
        collection_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO langchain_pg_collection (id, name, cmetadata) VALUES (%s, %s, %s)",
            (collection_id, collection_name, {})
        )
        conn.commit()
        print(f"集合创建成功, ID: {collection_id}")
    
    print(f"\n直接插入向量数据...")
    inserted_ids = []
    for i, (doc, emb) in enumerate(zip(test_docs, embeddings)):
        doc_id = str(uuid.uuid4())
        inserted_ids.append(doc_id)
        
        cursor.execute("""
            INSERT INTO langchain_pg_embedding 
            (id, collection_id, embedding, document, cmetadata)
            VALUES (%s, %s, %s::vector, %s, %s)
        """, (doc_id, collection_id, emb, doc.page_content, doc.metadata))
        
        print(f"  插入文档 {i+1}: ID={doc_id[:8]}...")
    
    conn.commit()
    print(f"\n提交事务完成")
    
    print(f"\n验证插入的数据...")
    placeholders = ','.join(['%s'] * len(inserted_ids))
    cursor.execute(f"SELECT id, document FROM langchain_pg_embedding WHERE id IN ({placeholders})", inserted_ids)
    results = cursor.fetchall()
    
    print(f"查询到 {len(results)} 条记录")
    for row in results:
        print(f"  ID: {row[0][:8]}..., 内容: {row[1][:50]}...")
    
    if len(results) == len(inserted_ids):
        print(f"\n✅ 直接插入成功: {len(results)}/{len(inserted_ids)} 条记录已验证")
    else:
        print(f"\n❌ 直接插入失败: 期望 {len(inserted_ids)} 条，实际 {len(results)} 条")
    
    cursor.execute("DELETE FROM langchain_pg_embedding WHERE id = ANY(%s)", (inserted_ids,))
    conn.commit()
    print(f"\n清理测试数据完成")
    
    conn.close()
    return len(results) == len(inserted_ids)

def test_pgvector_insert():
    """测试 PGVector 插入"""
    print("\n" + "=" * 70)
    print("步骤4: 测试 PGVector 插入")
    print("=" * 70)
    
    config = DatabaseConfig.from_env()
    embedding_model = get_embedding_model()
    
    print("\n创建 VectorStore...")
    vector_store = VectorStore(config, embedding_model)
    
    test_docs = [
        Document(
            page_content=f"PGVector 测试文档{i}。这是用于测试 PGVector 插入功能的内容。",
            metadata={"source": "pgvector_test.txt", "test_run": True, "index": i}
        )
        for i in range(3)
    ]
    
    print(f"\n准备插入 {len(test_docs)} 个文档...")
    
    print("\n调用 vector_store.add_documents...")
    ids = vector_store.add_documents(test_docs)
    print(f"返回的 ID: {len(ids)} 个")
    print(f"IDs: {[id_[:8] + '...' for id_ in ids]}")
    
    print("\n使用 SQL 验证...")
    conn = psycopg2.connect(config.connection_string)
    cursor = conn.cursor()
    
    placeholders = ','.join(['%s'] * len(ids))
    cursor.execute(f"SELECT id, document FROM langchain_pg_embedding WHERE id IN ({placeholders})", ids)
    results = cursor.fetchall()
    
    print(f"SQL 查询到 {len(results)} 条记录")
    
    found_ids = set(str(row[0]) for row in results)
    expected_ids = set(ids)
    missing_ids = expected_ids - found_ids
    
    if missing_ids:
        print(f"❌ 缺失的 ID: {[mid[:8] + '...' for mid in missing_ids]}")
    else:
        print(f"✅ 所有 ID 都已找到")
    
    cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
    total_count = cursor.fetchone()[0]
    print(f"langchain_pg_embedding 总记录数: {total_count}")
    
    cursor.execute("DELETE FROM langchain_pg_embedding WHERE id = ANY(%s)", (list(ids),))
    conn.commit()
    print(f"\n清理测试数据完成")
    
    conn.close()
    
    return len(results) == len(ids)

def main():
    print("=" * 70)
    print("向量数据库问题深度诊断")
    print("=" * 70)
    
    actual_dim, config_dim = check_embedding_config()
    
    tables = check_database_tables()
    
    if actual_dim != config_dim:
        print(f"\n" + "!" * 70)
        print("⚠️  关键问题: 嵌入维度不匹配!")
        print(f"   配置的 EMBEDDING_DIMENSION={config_dim}")
        print(f"   实际嵌入维度={actual_dim}")
        print(f"\n   这会导致向量存储失败!")
        print(f"   请更新 .env 文件: EMBEDDING_DIMENSION={actual_dim}")
        print("!" * 70)
        
        user_input = input("\n是否继续测试其他功能? (y/n): ")
        if user_input.lower() != 'y':
            print("\n诊断结束，请先修复嵌入维度配置。")
            return
    
    direct_success = test_direct_vector_insert()
    
    if direct_success:
        print(f"\n✅ 直接插入测试通过，数据库连接和维度正常")
    else:
        print(f"\n❌ 直接插入测试失败，请检查数据库配置")
        return
    
    pgvector_success = test_pgvector_insert()
    
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    
    print(f"\n嵌入维度:")
    print(f"  配置维度: {config_dim}")
    print(f"  实际维度: {actual_dim}")
    if actual_dim == config_dim:
        print(f"  ✅ 匹配")
    else:
        print(f"  ❌ 不匹配 - 需要更新 .env 文件")
    
    print(f"\n直接插入测试: {'✅ 通过' if direct_success else '❌ 失败'}")
    print(f"PGVector 插入测试: {'✅ 通过' if pgvector_success else '❌ 失败'}")
    
    if actual_dim != config_dim:
        print(f"\n⚠️  主要问题: 嵌入维度不匹配")
        print(f"   请在 .env 文件中设置: EMBEDDING_DIMENSION={actual_dim}")
    elif not pgvector_success:
        print(f"\n⚠️  PGVector 插入失败，可能是:")
        print(f"   1. 事务没有正确提交")
        print(f"   2. PGVector 内部问题")

if __name__ == "__main__":
    main()
