# Flask API 路由模块
"""
Flask API 路由模块
==================

实现 RESTful API 接口，处理客户端请求。

API 端点：
1. GET  /api/health          - 健康检查
2. POST /api/upload          - 上传文档
3. GET  /api/documents       - 获取文档列表
4. DELETE /api/documents/<id> - 删除文档
5. POST /api/search          - 相似度检索
6. GET  /api/stats           - 获取统计信息

设计模式：
- 使用蓝图 (Blueprint) 组织路由
- 使用全局对象管理共享资源（VectorStore 等）
- 统一的 JSON 响应格式
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, Blueprint, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain_core.documents import Document

from src.database.vector_db import DatabaseConfig, DatabaseManager, VectorStore
from src.document.processor import DocumentProcessor, SupportedFormats
from src.embedding.embedder import get_embedding_model, EmbeddingConfig
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 全局对象，用于存储 VectorStore 等共享资源
# 注意：在生产环境中应该使用依赖注入或单例模式
global_store: Dict[str, Any] = {}


def create_app() -> Flask:
    """
    创建 Flask 应用实例

    配置应用、注册蓝图、初始化全局资源。

    Returns:
        Flask: Flask 应用实例

    配置项（从环境变量读取）：
        UPLOAD_FOLDER: 上传文件存储目录
        MAX_CONTENT_LENGTH: 最大文件大小（字节）
        SECRET_KEY: Flask 密钥
    """
    app = Flask(__name__, template_folder="../../templates", static_folder="../../static")

    # 配置跨域
    CORS(app)

    # 配置文件上传
    app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", "104857600"))
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "rag-test-secret-key")

    # 确保上传目录存在
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # 注册蓝图
    register_blueprints(app)

    # 初始化全局资源
    with app.app_context():
        init_global_resources()

    return app


def register_blueprints(app: Flask) -> None:
    """
    注册 Flask 蓝图

    将 API 路由和页面路由分别注册到不同的蓝图，
    便于代码组织和维护。

    Args:
        app: Flask 应用实例
    """
    # API 蓝图（JSON 接口）
    api_bp = Blueprint("api", __name__, url_prefix="/api")

    # 页面蓝图（HTML 页面）
    pages_bp = Blueprint("pages", __name__)

    # 注册路由到蓝图
    register_api_routes(api_bp)
    register_page_routes(pages_bp)

    # 将蓝图注册到应用
    app.register_blueprint(api_bp)
    app.register_blueprint(pages_bp)


def init_global_resources() -> None:
    """
    初始化全局资源

    在应用启动时创建：
    1. 数据库配置
    2. 数据库管理器
    3. 嵌入模型
    4. 向量存储

    注意:
        这些资源是全局单例，避免重复创建连接。
    """
    print("正在初始化全局资源...")

    # 1. 创建数据库配置
    db_config = DatabaseConfig.from_env()
    global_store["db_config"] = db_config

    # 2. 创建数据库管理器
    db_manager = DatabaseManager(db_config)
    global_store["db_manager"] = db_manager

    # 尝试创建数据库和启用扩展
    try:
        print("检查数据库...")
        db_manager.create_database_if_not_exists()
        db_manager.test_connection()
        db_manager.enable_vector_extension()
        print("数据库初始化完成")
    except Exception as e:
        print(f"数据库初始化警告: {e}")
        print("请确保 PostgreSQL 服务运行并且 pgvector 扩展已安装")

    # 3. 创建嵌入模型
    try:
        print("加载嵌入模型...")
        embedding_model = get_embedding_model()
        global_store["embedding_model"] = embedding_model
        print("嵌入模型加载完成")
    except Exception as e:
        print(f"嵌入模型加载失败: {e}")
        print("请检查依赖是否正确安装")

    # 4. 创建向量存储（需要嵌入模型）
    if "embedding_model" in global_store:
        try:
            print("初始化向量存储...")
            vector_store = VectorStore(
                config=db_config,
                embedding_model=global_store["embedding_model"],
            )
            global_store["vector_store"] = vector_store
            print("向量存储初始化完成")
        except Exception as e:
            print(f"向量存储初始化失败: {e}")

    # 5. 创建文档处理器
    global_store["document_processor"] = DocumentProcessor()

    print("全局资源初始化完成")


def get_vector_store() -> Optional[VectorStore]:
    """
    获取向量存储实例

    便捷函数，从全局存储中获取 VectorStore。

    Returns:
        Optional[VectorStore]: 向量存储实例，如果未初始化则返回 None
    """
    return global_store.get("vector_store")


def get_document_processor() -> Optional[DocumentProcessor]:
    """
    获取文档处理器实例

    Returns:
        Optional[DocumentProcessor]: 文档处理器实例
    """
    return global_store.get("document_processor")


def register_api_routes(bp: Blueprint) -> None:
    """
    注册 API 路由

    定义所有 JSON API 端点。

    Args:
        bp: API 蓝图
    """

    @bp.route("/health", methods=["GET"])
    def health_check():
        """
        健康检查接口

        用于检查服务是否正常运行。

        Returns:
            JSON: {
                "status": "healthy",
                "timestamp": "2026-04-23T...",
                "version": "1.0.0"
            }
        """
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        })

    @bp.route("/upload", methods=["POST"])
    def upload_document():
        """
        文档上传接口

        处理用户上传的文档文件，执行：
        1. 验证文件格式
        2. 保存到上传目录
        3. 解析文档内容
        4. 文本切片
        5. 向量化存储

        Request:
            Content-Type: multipart/form-data
            Body: file=<文档文件>

        Returns:
            JSON: {
                "success": true,
                "message": "文档上传成功",
                "data": {
                    "file_name": "example.pdf",
                    "file_path": "uploads/xxx.pdf",
                    "total_chunks": 10,
                    "document_ids": ["uuid1", "uuid2", ...]
                }
            }

        错误响应:
            - 400: 没有文件或文件格式不支持
            - 500: 处理过程中出错
        """
        # 检查是否有文件
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "message": "没有上传文件",
            }), 400

        file = request.files["file"]

        # 检查文件名
        if file.filename == "":
            return jsonify({
                "success": False,
                "message": "没有选择文件",
            }), 400

        # 检查文件格式
        filename = secure_filename(file.filename)
        _, ext = os.path.splitext(filename)
        if SupportedFormats.from_extension(ext) is None:
            return jsonify({
                "success": False,
                "message": f"不支持的文件格式: {ext}",
                "supported_formats": SupportedFormats.get_supported_extensions(),
            }), 400

        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(
            request.app.config["UPLOAD_FOLDER"],
            unique_filename
        )

        try:
            # 保存文件
            file.save(file_path)
            print(f"文件已保存: {file_path}")

            # 获取文档处理器
            processor = get_document_processor()
            if processor is None:
                return jsonify({
                    "success": False,
                    "message": "文档处理器未初始化",
                }), 500

            # 处理文档
            documents = processor.process_file(file_path, split=True)

            if not documents:
                return jsonify({
                    "success": False,
                    "message": "文档内容为空或解析失败",
                }), 500

            # 获取统计信息
            stats = DocumentProcessor.get_document_stats(documents)

            # 获取向量存储
            vector_store = get_vector_store()
            if vector_store is None:
                return jsonify({
                    "success": False,
                    "message": "向量存储未初始化",
                    "stats": stats,
                }), 500

            # 存储到向量数据库
            document_ids = vector_store.add_documents(documents)

            return jsonify({
                "success": True,
                "message": "文档上传并处理成功",
                "data": {
                    "original_name": filename,
                    "stored_name": unique_filename,
                    "file_path": file_path,
                    "total_chunks": len(documents),
                    "document_ids": document_ids,
                    "stats": stats,
                },
            })

        except Exception as e:
            # 清理保存的文件（如果有）
            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "success": False,
                "message": f"处理文档时出错: {str(e)}",
            }), 500

    @bp.route("/search", methods=["POST"])
    def search_documents():
        """
        相似度检索接口

        根据用户查询，在向量数据库中检索相似的文档片段。

        Request:
            Content-Type: application/json
            Body: {
                "query": "要搜索的文本",
                "k": 5,  # 可选，返回结果数量，默认 4
                "filter": {}  # 可选，元数据过滤条件
            }

        Returns:
            JSON: {
                "success": true,
                "data": {
                    "query": "搜索文本",
                    "results": [
                        {
                            "content": "文档片段内容",
                            "score": 0.85,  # 相似度分数
                            "metadata": {
                                "source": "文件路径",
                                "file_name": "文件名",
                                ...
                            }
                        },
                        ...
                    ]
                }
            }

        相似度分数说明:
            - 使用余弦相似度时，分数范围 [0, 1]
            - 越接近 1 表示越相似
        """
        # 获取请求数据
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({
                "success": False,
                "message": "缺少查询参数",
            }), 400

        query = data["query"]
        k = data.get("k", 4)
        filter_dict = data.get("filter")

        # 获取向量存储
        vector_store = get_vector_store()
        if vector_store is None:
            return jsonify({
                "success": False,
                "message": "向量存储未初始化",
            }), 500

        try:
            # 执行相似度检索（带分数）
            results = vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict,
            )

            # 格式化结果
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                })

            return jsonify({
                "success": True,
                "data": {
                    "query": query,
                    "result_count": len(formatted_results),
                    "results": formatted_results,
                },
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"检索时出错: {str(e)}",
            }), 500

    @bp.route("/stats", methods=["GET"])
    def get_stats():
        """
        获取系统统计信息

        返回当前系统的状态和统计数据。

        Returns:
            JSON: {
                "success": true,
                "data": {
                    "vector_store_initialized": true,
                    "database_config": {
                        "host": "172.19.249.222",
                        "port": 5432,
                        "db_name": "rag_db"
                    },
                    "embedding_config": {
                        "model_type": "local",
                        "model_name": "all-MiniLM-L6-v2"
                    }
                }
            }
        """
        db_config = global_store.get("db_config")
        embedding_config = EmbeddingConfig.from_env()

        return jsonify({
            "success": True,
            "data": {
                "vector_store_initialized": "vector_store" in global_store,
                "database_config": {
                    "host": db_config.host if db_config else None,
                    "port": db_config.port if db_config else None,
                    "db_name": db_config.db_name if db_config else None,
                    "table_name": db_config.table_name if db_config else None,
                },
                "embedding_config": {
                    "model_type": embedding_config.model_type.value,
                    "model_name": embedding_config.model_name,
                },
            },
        })


def register_page_routes(bp: Blueprint) -> None:
    """
    注册页面路由

    定义 HTML 页面的路由。

    Args:
        bp: 页面蓝图
    """

    @bp.route("/", methods=["GET"])
    def index():
        """
        首页路由

        渲染主页面，包含文档上传和搜索功能。

        Returns:
            HTML: 渲染后的 index.html 页面
        """
        return render_template("index.html")

    @bp.route("/upload", methods=["GET"])
    def upload_page():
        """
        上传页面路由

        渲染文档上传页面。

        Returns:
            HTML: 渲染后的 upload.html 页面
        """
        return render_template("upload.html")

    @bp.route("/search", methods=["GET"])
    def search_page():
        """
        搜索页面路由

        渲染搜索测试页面。

        Returns:
            HTML: 渲染后的 search.html 页面
        """
        return render_template("search.html")
