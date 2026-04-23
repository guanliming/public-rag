# RAG 测试项目 - 源码包
"""
RAG (Retrieval-Augmented Generation) 测试项目
===============================================

这是一个基于 LangChain 和 PostgreSQL (pgvector) 的 RAG 测试项目。

功能特性：
- 支持 PDF 和 Markdown 文档上传
- 文档自动解析、切片和向量化
- 使用 pgvector 进行向量存储和相似度检索
- Web 前端界面用于文档管理

项目结构：
- src/
  - database/     数据库模块
  - document/     文档处理模块
  - embedding/    向量化模块
  - api/          Flask API 路由
- templates/      HTML 模板
- static/         静态文件
- uploads/        上传文件存储目录
"""

__version__ = "1.0.0"
