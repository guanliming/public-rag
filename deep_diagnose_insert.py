#!/usr/bin/env python3
"""
深度诊断脚本 - 检查向量插入失败的真正原因
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

def main():
    print("=" * 70)
    print("深度诊断：向量插入失败分析")
    print("=" * 70)
    
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")
    dimension = os.getenv("EMBEDDING_DIMENSION", "1024")
    print(f"\n当前配置:")
    print(f"  模型: {model_name}")
    print(f"  配置维度: {dimension}")
    
    config = DatabaseConfig.from_env()
    
    print("\n" + "-" * 70)
    print("步骤1: 检查数据库表结构和实际维度")
    print("-" * 70)
    
    try:
        conn = psycopg2.connect(config.connection_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [t['table_name'] for t in cursor.fetchall()]
        print(f"\n数据库中的表: {tables}")
        
        if 'langchain_pg_embedding' in tables:
            print("\n--- langchain_pg_embedding 表结构 ---")
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'langchain_pg_embedding'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col['column_name']}: {col['data_type']} [nullable: {col['is_nullable']}]")
            
            print("\n--- 检查向量列的实际维度 ---")
            cursor.execute("""
                SELECT 
                    attname, 
                    atttypmod,
                    format_type(atttypid, atttypmod) as type_full
                FROM pg_attribute
                WHERE attrelid = 'langchain_pg_embedding'::regclass
                AND attname = 'embedding'
                AND attnum > 0
            """)
            embedding_col = cursor.fetchone()
            if embedding_col:
                print(f"  列名: {embedding_col['attname']}")
                print(f"  atttypmod: {embedding_col['atttypmod']}")
                print(f"  完整类型: {embedding_col['type_full']}")
                
                if embedding_col['atttypmod'] > 0:
                    actual_dim = embedding_col['atttypmod'] - 4
                    print(f"  ⚠️  实际向量维度: {actual_dim}")
                    if str(actual_dim) != dimension:
                        print(f"  ❌ 维度不匹配! 配置: {dimension}, 实际: {actual_dim}")
                        print(f"     这是插入失败的主要原因!")
                    else:
                        print(f"  ✅ 维度匹配")
                else:
                    print(f"  ⚠️  维度未指定 (atttypmod = -1)，pgvector 会接受任何维度")
            else:
                print("  ❌ 未找到 embedding 列")
            
            print("\n--- 表中的现有数据 ---")
            cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
            count = cursor.fetchone()[0]
            print(f"  总记录数: {count}")
            
            if count > 0:
                cursor.execute("SELECT * FROM langchain_pg_embedding LIMIT 1")
                sample = cursor.fetchone()
                if sample:
                    print(f"  示例数据:")
                    for key, value in sample.items():
                        if key != 'embedding':
                            print(f"    {key}: {value}")
                        else:
                            print(f"    embedding: <vector>")
        
        else:
            print("\n❌ langchain_pg_embedding 表不存在")
            print("  系统应该会在首次使用时创建新表")
        
        if 'langchain_pg_collection' in tables:
            print("\n--- langchain_pg_collection 表内容 ---")
            cursor.execute("SELECT * FROM langchain_pg_collection")
            collections = cursor.fetchall()
            for col in collections:
                print(f"  {dict(col)}")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ 检查数据库时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 70)
    print("步骤2: 测试直接插入（捕获所有错误）")
    print("-" * 70)
    
    print("\n加载嵌入模型...")
    try:
        embedding_model = get_embedding_model()
        print("✅ 嵌入模型加载成功")
    except Exception as e:
        print(f"❌ 嵌入模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    test_texts = [
        "这是测试文档1。向量存储诊断测试。",
        "这是测试文档2。检查插入是否成功。"
    ]
    
    print(f"\n生成测试嵌入...")
    try:
        embeddings = embedding_model.embed_documents(test_texts)
        actual_dim = len(embeddings[0])
        print(f"✅ 嵌入生成成功")
        print(f"  实际嵌入维度: {actual_dim}")
        
        if str(actual_dim) != dimension:
            print(f"  ⚠️  与配置维度不匹配! 配置: {dimension}, 实际: {actual_dim}")
            print(f"     这会导致插入失败!")
    except Exception as e:
        print(f"❌ 嵌入生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-" * 70)
    print("步骤3: 使用 VectorStore 测试插入（详细日志）")
    print("-" * 70)
    
    try:
        vector_store = VectorStore(config, embedding_model)
        print("✅ VectorStore 创建成功")
        
        test_docs = [
            Document(
                page_content=f"深度诊断测试文档{i}。这是用于诊断插入问题的测试内容。",
                metadata={"source": "deep_diagnose.txt", "diagnose_run": True, "index": i}
            )
            for i in range(2)
        ]
        
        print(f"\n准备插入 {len(test_docs)} 个测试文档...")
        print(f"\n调用 vector_store.add_documents...")
        
        try:
            ids = vector_store.add_documents(test_docs)
            print(f"\n✅ add_documents 返回结果:")
            print(f"  ID 数量: {len(ids)}")
            print(f"  IDs: {ids}")
            
            print(f"\n" + "=" * 70)
            print("关键检查：验证数据是否真的写入数据库")
            print("=" * 70)
            
            verification = vector_store.verify_documents_stored(ids)
            print(f"\n验证结果:")
            print(f"  成功: {verification['success']}")
            print(f"  期望数量: {verification['expected_count']}")
            print(f"  实际存储: {verification['stored_count']}")
            print(f"  消息: {verification['message']}")
            
            if verification['success']:
                print("\n✅ 插入成功! 问题已修复。")
            else:
                print("\n❌ 插入仍然失败!")
                print("\n可能的原因:")
                print("  1. 维度不匹配（配置维度 != 实际嵌入维度 != 数据库表维度）")
                print("  2. 事务没有正确提交")
                print("  3. 其他数据库错误被抑制了")
                
        except Exception as e:
            print(f"\n❌ add_documents 抛出异常: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"\n❌ VectorStore 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
