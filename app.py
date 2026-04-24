# RAG 测试项目 - 应用入口
"""
RAG (Retrieval-Augmented Generation) 测试项目
===============================================

这是一个基于 LangChain 和 PostgreSQL (pgvector) 的 RAG 测试项目。

启动方式:
    python app.py

或者使用 Flask 命令:
    flask run --host=0.0.0.0 --port=5000

访问地址:
    http://localhost:5000

功能特性:
    - 支持 PDF 和 Markdown 文档上传
    - 文档自动解析、切片和向量化
    - 使用 pgvector 进行向量存储和相似度检索
    - Web 前端界面用于文档管理和检索

依赖:
    - Python 3.8+
    - PostgreSQL 11+ (带 pgvector 扩展)
    - 详见 requirements.txt

项目结构:
    app.py                 # Flask 应用入口
    requirements.txt       # Python 依赖列表
    .env                   # 环境变量配置
    src/
        __init__.py
        database/          # 数据库模块
            __init__.py
            vector_db.py   # 向量数据库实现
        document/          # 文档处理模块
            __init__.py
            processor.py   # 文档加载和切片
        embedding/         # 向量化模块
            __init__.py
            embedder.py    # 嵌入模型实现
        api/               # Flask API 模块
            __init__.py
            routes.py      # API 路由定义
    templates/             # HTML 模板
        index.html         # 首页
        upload.html        # 上传页面
        search.html        # 搜索页面
    static/                # 静态文件
        css/
        js/
    uploads/               # 上传文件存储目录
"""

import os
from src.config import load_config

# 加载配置（支持 settings.yaml 和环境变量）
project_root = os.path.dirname(__file__)
load_config(project_root, verbose=True)

# 从环境变量读取 Flask 配置
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"


def create_app():
    """
    创建 Flask 应用实例

    这是应用的工厂函数，供 Flask 命令行工具使用。

    Returns:
        Flask: Flask 应用实例

    使用示例:
        flask --app app:create_app run
    """
    from src.api.routes import create_app as _create_app
    return _create_app()


def main():
    """
    主函数

    启动 Flask 开发服务器。
    """
    print("=" * 60)
    print("RAG 测试项目启动中...")
    print("=" * 60)

    # 创建 Flask 应用
    app = create_app()

    print(f"Flask 服务器配置:")
    print(f"  - Host: {FLASK_HOST}")
    print(f"  - Port: {FLASK_PORT}")
    print(f"  - Debug: {FLASK_DEBUG}")
    print()
    print(f"访问地址: http://localhost:{FLASK_PORT}")
    print("=" * 60)

    # 启动服务器
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
    )


if __name__ == "__main__":
    main()
