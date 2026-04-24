#!/usr/bin/env python3
"""
清理数据库表脚本
删除旧的向量表，让系统用新的维度重新创建
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
from src.database.vector_db import DatabaseConfig

def main():
    print("=" * 70)
    print("清理数据库表")
    print("=" * 70)
    
    config = DatabaseConfig.from_env()
    
    print(f"\n数据库配置:")
    print(f"  主机: {config.host}")
    print(f"  端口: {config.port}")
    print(f"  数据库: {config.db_name}")
    print(f"  配置维度: {config.embedding_dimension}")
    
    print(f"\n当前配置:")
    print(f"  USE_FREE_EMBEDDING_MODELS: {os.getenv('USE_FREE_EMBEDDING_MODELS', 'true')}")
    print(f"  EMBEDDING_DIMENSION: {config.embedding_dimension}")
    
    print("\n" + "!" * 70)
    print("⚠️  警告: 此操作将删除以下表中的所有数据:")
    print("   - langchain_pg_embedding")
    print("   - langchain_pg_collection")
    print("!" * 70)
    
    user_input = input("\n是否继续? (yes/no): ")
    if user_input.lower() not in ['yes', 'y']:
        print("操作已取消")
        return False
    
    print("\n连接数据库...")
    try:
        conn = psycopg2.connect(config.connection_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [t[0] for t in cursor.fetchall()]
        print(f"数据库中的表: {tables}")
        
        if 'langchain_pg_embedding' in tables:
            print("\n删除 langchain_pg_embedding 表...")
            cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")
            print("✅ 已删除 langchain_pg_embedding 表")
        
        if 'langchain_pg_collection' in tables:
            print("删除 langchain_pg_collection 表...")
            cursor.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
            print("✅ 已删除 langchain_pg_collection 表")
        
        conn.commit()
        conn.close()
        
        print("\n" + "=" * 70)
        print("清理完成")
        print("=" * 70)
        print(f"\n下一步:")
        print(f"  1. 系统会在首次使用时自动创建新表")
        print(f"  2. 新表将使用维度: {config.embedding_dimension}")
        print(f"  3. 请运行 verify_fix.py 验证修复")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 清理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
