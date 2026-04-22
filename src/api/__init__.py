# Flask API 模块
"""
Flask API 模块
===============

提供 Web API 接口，支持：
- 文档上传
- 文档列表查询
- 文档删除
- 相似度检索
- 系统状态查询

API 设计遵循 RESTful 原则：
- GET: 获取资源
- POST: 创建资源
- DELETE: 删除资源

依赖：
- flask: Web 框架
- flask-cors: 跨域支持
- werkzeug: 文件上传处理
"""

from src.api.routes import create_app, register_blueprints

__all__ = ["create_app", "register_blueprints"]
