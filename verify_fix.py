#!/usr/bin/env python3
"""
最终验证脚本 - 测试向量存储修复
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
from langchain_core.documents import Document

from src.database.vector_db import DatabaseConfig, VectorStore
from src.embedding.embedder import get_embedding_model

def main():
    print("=" * 70)
    print("向量存储修复验证")
    print("=" * 70)
    
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")
    dimension = os.getenv("EMBEDDING_DIMENSION", "1024")
    print(f"\n当前配置:")
    print(f"  模型: {model_name}")
    print(f"  维度: {dimension}")
    print(f"  ⚠️  确保模型和维度匹配！")
    print(f"     - text-embedding-v3: 1024")
    print(f"     - qwen3-vl-embedding: 2560")
    print(f"     - tongyi-embedding-vision-plus-2026-03-06: 1152")
    print(f"     - tongyi-embedding-vision-flash-2026-03-06: 768")
    
    print("\n" + "-" * 70)
    print("步骤1: 初始化嵌入模型和向量存储")
    print("-" * 70)
    
    config = DatabaseConfig.from_env()
    print(f"\n数据库配置:")
    print(f"  主机: {config.host}")
    print(f"  数据库: {config.db_name}")
    print(f"  配置维度: {config.embedding_dimension}")
    
    print("\n加载嵌入模型...")
    embedding_model = get_embedding_model()
    
    print("\n创建 VectorStore (自动检测维度)...")
    vector_store = VectorStore(config, embedding_model)
    
    print("\n" + "-" * 70)
    print("步骤2: 测试文档插入和验证")
    print("-" * 70)
    
    test_docs = [
        Document(
            page_content=f"验证测试文档{i}。这是用于验证修复是否成功的测试内容。向量存储应该能够正确保存和检索这些文档。",
            metadata={"source": "verify_test.txt", "verify_run": True, "index": i}
        )
        for i in range(3)
    ]
    
    print(f"\n准备插入 {len(test_docs)} 个测试文档...")
    
    print("\n调用 vector_store.add_documents...")
    ids = vector_store.add_documents(test_docs)
    print(f"✅ 返回的 ID 数量: {len(ids)}")
    print(f"   IDs: {[id_[:12] + '...' for id_ in ids]}")
    
    print("\n" + "-" * 70)
    print("步骤3: 验证数据是否正确存储")
    print("-" * 70)
    
    print("\n调用 verify_documents_stored...")
    verification = vector_store.verify_documents_stored(ids)
    print(f"\n验证结果:")
    print(f"  成功: {verification['success']}")
    print(f"  期望数量: {verification['expected_count']}")
    print(f"  实际存储: {verification['stored_count']}")
    print(f"  缺失 ID: {verification['missing_ids']}")
    print(f"  消息: {verification['message']}")
    
    print("\n" + "-" * 70)
    print("步骤4: 测试相似度搜索")
    print("-" * 70)
    
    print("\n搜索 '验证测试文档'...")
    results = vector_store.similarity_search("验证测试文档", k=5)
    print(f"✅ 搜索到 {len(results)} 个结果")
    
    for i, doc in enumerate(results):
        print(f"\n  结果 {i+1}:")
        print(f"    内容: {doc.page_content[:80]}...")
        print(f"    元数据: {doc.metadata}")
    
    print("\n" + "-" * 70)
    print("步骤5: 清理测试数据")
    print("-" * 70)
    
    print(f"\n删除测试文档 (ID 数量: {len(ids)})...")
    vector_store.delete(ids)
    print("✅ 已删除测试文档")
    
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    
    if verification['success'] and verification['stored_count'] == verification['expected_count']:
        print("\n✅ 验证成功!")
        print(f"   插入数量: {verification['expected_count']}")
        print(f"   实际存储: {verification['stored_count']}")
        print(f"   搜索结果: {len(results)} 条")
        print("\n   向量存储修复已完成!")
        return True
    else:
        print("\n❌ 验证失败!")
        print(f"   插入数量: {verification['expected_count']}")
        print(f"   实际存储: {verification['stored_count']}")
        print(f"   消息: {verification['message']}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
