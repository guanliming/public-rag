#!/usr/bin/env python3
"""
诊断查询问题 - 详细检查异常和连接
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
    print("诊断查询问题 - 详细检查异常和连接")
    print("=" * 70)
    
    from src.database.vector_db import DatabaseConfig
    config = DatabaseConfig.from_env()
    
    print(f"\n连接数据库...")
    print(f"连接字符串: {config.connection_string}")
    
    try:
        conn = psycopg2.connect(config.connection_string)
        print("✅ 连接成功")
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n" + "-" * 70)
        print("步骤1: 检查表是否存在")
        print("-" * 70)
        
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        table_names = [t['table_name'] for t in tables]
        print(f"\n数据库中的表: {table_names}")
        
        print("\n" + "-" * 70)
        print("步骤2: 直接查询 langchain_pg_embedding")
        print("-" * 70)
        
        if 'langchain_pg_embedding' in table_names:
            print("\n尝试查询 langchain_pg_embedding 表...")
            
            try:
                cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
                total_count = cursor.fetchone()[0]
                print(f"✅ 查询成功，总记录数: {total_count}")
            except Exception as e:
                print(f"❌ 查询失败")
                print(f"  异常类型: {type(e)}")
                print(f"  异常值: {e}")
                print(f"  异常字符串: {str(e)}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "-" * 70)
            print("步骤3: 检查表结构和权限")
            print("-" * 70)
            
            try:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'langchain_pg_embedding'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                print(f"\nlangchain_pg_embedding 表结构:")
                for col in columns:
                    print(f"  {col['column_name']}: {col['data_type']} [nullable: {col['is_nullable']}]")
            except Exception as e:
                print(f"❌ 获取表结构失败: {e}")
            
            print("\n" + "-" * 70)
            print("步骤4: 测试不同的查询方式")
            print("-" * 70)
            
            try:
                cursor.execute("SELECT * FROM langchain_pg_embedding LIMIT 1")
                sample = cursor.fetchone()
                if sample:
                    print(f"\n✅ 成功获取示例数据")
                    print(f"  列名: {sample.keys()}")
                else:
                    print(f"\n表为空")
            except Exception as e:
                print(f"❌ 获取示例数据失败: {e}")
            
            print("\n" + "-" * 70)
            print("步骤5: 检查 langchain_pg_collection 表")
            print("-" * 70)
            
            if 'langchain_pg_collection' in table_names:
                try:
                    cursor.execute("SELECT * FROM langchain_pg_collection")
                    collections = cursor.fetchall()
                    print(f"\nlangchain_pg_collection 表内容:")
                    for col in collections:
                        print(f"  {dict(col)}")
                except Exception as e:
                    print(f"❌ 查询 langchain_pg_collection 失败: {e}")
            
            print("\n" + "-" * 70)
            print("步骤6: 检查 current_schema 和权限")
            print("-" * 70)
            
            try:
                cursor.execute("SELECT current_schema()")
                schema = cursor.fetchone()[0]
                print(f"\n当前 schema: {schema}")
            except Exception as e:
                print(f"❌ 获取 schema 失败: {e}")
            
            try:
                cursor.execute("SELECT current_user")
                user = cursor.fetchone()[0]
                print(f"当前用户: {user}")
            except Exception as e:
                print(f"❌ 获取用户失败: {e}")
            
            try:
                cursor.execute("""
                    SELECT table_name, has_table_privilege(table_name, 'SELECT') as can_select
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                privileges = cursor.fetchall()
                print(f"\n表权限:")
                for p in privileges:
                    print(f"  {p['table_name']}: SELECT = {p['can_select']}")
            except Exception as e:
                print(f"❌ 检查权限失败: {e}")
        
        else:
            print("\n❌ langchain_pg_embedding 表不存在")
        
        print("\n" + "=" * 70)
        print("诊断完成")
        print("=" * 70)
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ 连接或诊断时出错: {e}")
        print(f"  异常类型: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
