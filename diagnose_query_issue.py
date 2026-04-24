#!/usr/bin/env python3
"""
诊断查询问题 - 检查 UUID 类型和列结构
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

def main():
    print("=" * 70)
    print("诊断查询问题 - 检查 UUID 类型和列结构")
    print("=" * 70)
    
    from src.database.vector_db import DatabaseConfig
    config = DatabaseConfig.from_env()
    
    print(f"\n连接数据库...")
    
    try:
        conn = psycopg2.connect(config.connection_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n" + "-" * 70)
        print("步骤1: 检查 langchain_pg_embedding 表结构")
        print("-" * 70)
        
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'langchain_pg_embedding'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        
        print("\n表结构:")
        for col in columns:
            print(f"  {col['column_name']}: {col['data_type']} [nullable: {col['is_nullable']}]")
        
        print("\n" + "-" * 70)
        print("步骤2: 检查表中的数据量")
        print("-" * 70)
        
        cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        total_count = cursor.fetchone()[0]
        print(f"\n总记录数: {total_count}")
        
        if total_count > 0:
            cursor.execute("SELECT * FROM langchain_pg_embedding LIMIT 1")
            sample = cursor.fetchone()
            print(f"\n示例数据:")
            for key, value in sample.items():
                if key == 'embedding':
                    if value:
                        print(f"    embedding: <vector, 维度: {len(value)}>")
                    else:
                        print(f"    embedding: None")
                else:
                    print(f"    {key}: {value}")
        
        print("\n" + "-" * 70)
        print("步骤3: 检查 UUID 列的类型和值")
        print("-" * 70)
        
        cursor.execute("""
            SELECT pg_typeof(uuid) as uuid_type,
                   pg_typeof(collection_id) as collection_id_type
            FROM langchain_pg_embedding
            LIMIT 1
        """)
        types = cursor.fetchone()
        if types:
            print(f"\n列类型:")
            print(f"  uuid 列类型: {types['uuid_type']}")
            print(f"  collection_id 列类型: {types['collection_id_type']}")
        
        if total_count > 0:
            cursor.execute("SELECT uuid::text as uuid_str, uuid FROM langchain_pg_embedding LIMIT 3")
            uuid_samples = cursor.fetchall()
            print(f"\nUUID 示例:")
            for i, row in enumerate(uuid_samples):
                print(f"  示例 {i+1}:")
                print(f"    uuid::text: {row['uuid_str']}")
                print(f"    uuid (原始): {row['uuid']}")
                print(f"    类型: {type(row['uuid'])}")
        
        print("\n" + "-" * 70)
        print("步骤4: 测试不同的查询方式")
        print("-" * 70)
        
        if total_count > 0:
            cursor.execute("SELECT uuid::text FROM langchain_pg_embedding LIMIT 1")
            test_uuid = cursor.fetchone()[0]
            print(f"\n测试 UUID: {test_uuid}")
            
            print("\n方式1: 直接用字符串查询 (WHERE uuid = %s)")
            try:
                cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE uuid = %s", (test_uuid,))
                count = cursor.fetchone()[0]
                print(f"  结果: {count} 条")
            except Exception as e:
                print(f"  错误: {e}")
            
            print("\n方式2: 用 ::uuid 转换字符串 (WHERE uuid = %s::uuid)")
            try:
                cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE uuid = %s::uuid", (test_uuid,))
                count = cursor.fetchone()[0]
                print(f"  结果: {count} 条")
            except Exception as e:
                print(f"  错误: {e}")
            
            print("\n方式3: 将列转换为文本 (WHERE uuid::text = %s)")
            try:
                cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE uuid::text = %s", (test_uuid,))
                count = cursor.fetchone()[0]
                print(f"  结果: {count} 条")
            except Exception as e:
                print(f"  错误: {e}")
            
            print("\n方式4: 用 IN 查询 (WHERE uuid IN (%s))")
            try:
                cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE uuid IN (%s)", (test_uuid,))
                count = cursor.fetchone()[0]
                print(f"  结果: {count} 条")
            except Exception as e:
                print(f"  错误: {e}")
            
            print("\n方式5: 用 IN 查询，转换为 uuid (WHERE uuid IN (%s::uuid))")
            try:
                cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE uuid IN (%s::uuid)", (test_uuid,))
                count = cursor.fetchone()[0]
                print(f"  结果: {count} 条")
            except Exception as e:
                print(f"  错误: {e}")
        
        print("\n" + "=" * 70)
        print("诊断完成")
        print("=" * 70)
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ 诊断时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
