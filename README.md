# RAG 测试项目

基于 LangChain 和 PostgreSQL (pgvector) 的 RAG (Retrieval-Augmented Generation) 测试项目。

## 项目简介

这是一个完整的 RAG 系统原型，实现了以下核心功能：

- 📄 **文档上传**：支持 PDF、Markdown、纯文本格式的文档上传
- 🔍 **智能检索**：基于向量相似度的语义检索
- ⚡ **向量化存储**：使用 pgvector 进行高性能向量存储和检索
- 🌐 **Web 界面**：提供友好的前端界面用于文档管理和搜索

### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 核心开发语言 |
| LangChain | 0.1+ | RAG 框架 |
| PostgreSQL | 11+ | 关系数据库 |
| pgvector | 0.2+ | 向量扩展 |
| Flask | 3.0+ | Web 框架 |
| sentence-transformers | 2.2+ | 本地嵌入模型 |

## 项目结构

```
public-rag/
├── app.py                      # Flask 应用入口
├── requirements.txt            # Python 依赖列表
├── .env                        # 环境变量配置（实际使用）
├── .env.example                # 环境变量示例
├── README.md                   # 项目说明文档
├── src/                        # 源码目录
│   ├── __init__.py
│   ├── database/               # 数据库模块
│   │   ├── __init__.py
│   │   └── vector_db.py        # 向量数据库实现
│   ├── document/               # 文档处理模块
│   │   ├── __init__.py
│   │   └── processor.py        # 文档加载和切片
│   ├── embedding/              # 向量化模块
│   │   ├── __init__.py
│   │   └── embedder.py         # 嵌入模型实现
│   └── api/                    # Flask API 模块
│       ├── __init__.py
│       └── routes.py           # API 路由定义
├── templates/                  # HTML 模板
│   ├── index.html              # 首页
│   ├── upload.html             # 上传页面
│   └── search.html             # 搜索页面
├── scripts/                    # 辅助脚本
│   ├── init_db.sql             # 数据库初始化脚本
│   ├── install_dependencies.sh # 依赖安装脚本
│   └── start_app.sh            # 应用启动脚本
└── uploads/                    # 上传文件存储目录（运行时创建）
```

## 快速开始

### 1. 环境准备

#### 1.1 Python 环境

项目使用 WSL 中的 Python 环境：

- 路径：`/home/dawn/miniconda3/envs/temp312/bin/python`
- 版本：Python 3.12

#### 1.2 数据库准备

确保 PostgreSQL 服务运行在 `172.19.249.222:5432`，并且已安装 pgvector 扩展。

**安装 pgvector（如果尚未安装）：**

```sql
-- 以超级用户连接到 PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. 安装依赖

#### 方式一：使用脚本（推荐）

```bash
# 在 WSL 终端中执行
bash scripts/install_dependencies.sh
```

#### 方式二：手动安装

```bash
# 激活 Conda 环境
source /home/dawn/miniconda3/etc/profile.d/conda.sh
conda activate temp312

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

查看并修改 `.env` 文件：

```bash
# 数据库配置
DB_HOST=172.19.249.222
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=rag_db

# 嵌入模型配置（使用本地模型，无需 API Key）
EMBEDDING_MODEL_TYPE=local
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# 文档切片配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Flask 配置
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=true
```

### 4. 初始化数据库

首次运行时，系统会自动尝试创建数据库和启用 pgvector 扩展。如果需要手动初始化：

```bash
# 连接到 PostgreSQL
psql -U postgres -h 172.19.249.222

# 执行初始化脚本
\i scripts/init_db.sql
```

### 5. 启动应用

#### 方式一：使用脚本（推荐）

```bash
# 在 WSL 终端中执行
bash scripts/start_app.sh
```

#### 方式二：手动启动

```bash
# 激活 Conda 环境
conda activate temp312

# 启动应用
python app.py
```

### 6. 访问应用

应用启动后，访问以下地址：

- **首页**：http://localhost:5000
- **文档上传**：http://localhost:5000/upload
- **搜索测试**：http://localhost:5000/search

## 核心模块说明

### 1. 数据库模块 (`src/database/`)

#### VectorStore 类

向量存储类，封装了 LangChain PGVector 的操作。

**主要功能：**
- `add_documents()`：添加文档到向量数据库
- `similarity_search()`：相似度检索
- `similarity_search_with_score()`：带分数的相似度检索
- `delete()`：删除文档
- `get_retriever()`：获取 LangChain Retriever

**使用示例：**

```python
from src.database.vector_db import VectorStore, DatabaseConfig
from src.embedding.embedder import get_embedding_model

# 创建配置
config = DatabaseConfig.from_env()

# 获取嵌入模型
embedding = get_embedding_model()

# 创建向量存储
vector_store = VectorStore(config, embedding)

# 添加文档
vector_store.add_documents(documents)

# 检索
results = vector_store.similarity_search("查询文本", k=5)
```

#### DatabaseManager 类

数据库管理类，负责数据库连接管理和初始化。

**主要功能：**
- `create_database_if_not_exists()`：创建数据库（如果不存在）
- `enable_vector_extension()`：启用 pgvector 扩展
- `test_connection()`：测试数据库连接

### 2. 文档处理模块 (`src/document/`)

#### DocumentProcessor 类

文档处理器，整合了文档加载和文本切片功能。

**支持的格式：**
- PDF (.pdf)
- Markdown (.md, .markdown)
- 纯文本 (.txt)

**使用示例：**

```python
from src.document.processor import DocumentProcessor

# 创建处理器（自定义切片大小）
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# 处理单个文件
documents = processor.process_file("document.pdf")

# 处理目录
documents = processor.process_directory("./docs/", recursive=True)

# 获取统计信息
stats = DocumentProcessor.get_document_stats(documents)
print(f"总片段数: {stats['total_count']}")
print(f"总字符数: {stats['total_chars']}")
```

#### 切片策略

项目使用两种切片策略：

1. **递归字符分割（默认）**
   - 按段落、句子、单词的优先级分割
   - 保持语义完整性
   - 适用于通用文本

2. **Markdown 标题分割**
   - 按 Markdown 标题层级分割
   - 保留文档结构
   - 适用于 Markdown 文档

### 3. 向量化模块 (`src/embedding/`)

#### 支持的嵌入模型

**1. 本地模型（推荐测试使用）**

基于 `sentence-transformers` 库，无需 API Key，离线可用。

| 模型名称 | 维度 | 特点 |
|----------|------|------|
| all-MiniLM-L6-v2 | 384 | 轻量快速，推荐测试 |
| all-mpnet-base-v2 | 768 | 效果更好 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 多语言支持 |

**2. OpenAI 嵌入模型**

需要 OpenAI API Key，效果更好但有费用。

| 模型名称 | 维度 | 特点 |
|----------|------|------|
| text-embedding-ada-002 | 1536 | 经济实惠 |
| text-embedding-3-small | 1536 | 新版小模型 |
| text-embedding-3-large | 3072 | 新版大模型 |

**使用示例：**

```python
from src.embedding.embedder import get_embedding_model

# 使用本地模型（默认）
embedding = get_embedding_model()

# 使用 OpenAI 模型
# 需要设置 OPENAI_API_KEY 环境变量
embedding = get_embedding_model(model_type="openai", model_name="text-embedding-ada-002")

# 向量化查询
query_vector = embedding.embed_query("什么是机器学习？")

# 向量化文档
doc_vectors = embedding.embed_documents(["文档内容1", "文档内容2"])
```

### 4. API 模块 (`src/api/`)

#### RESTful API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 健康检查 |
| POST | /api/upload | 上传文档 |
| POST | /api/search | 相似度检索 |
| GET | /api/stats | 系统统计 |

**API 详情：**

**1. 健康检查**
```
GET /api/health

响应：
{
    "status": "healthy",
    "timestamp": "2026-04-23T...",
    "version": "1.0.0"
}
```

**2. 上传文档**
```
POST /api/upload
Content-Type: multipart/form-data

请求体：
- file: 文档文件

响应：
{
    "success": true,
    "message": "文档上传成功",
    "data": {
        "original_name": "document.pdf",
        "stored_name": "uuid_document.pdf",
        "total_chunks": 10,
        "document_ids": ["..."],
        "stats": {
            "total_count": 10,
            "total_chars": 15000,
            ...
        }
    }
}
```

**3. 相似度检索**
```
POST /api/search
Content-Type: application/json

请求体：
{
    "query": "查询文本",
    "k": 5,
    "filter": {}  // 可选，元数据过滤
}

响应：
{
    "success": true,
    "data": {
        "query": "查询文本",
        "result_count": 5,
        "results": [
            {
                "content": "文档片段内容",
                "score": 0.85,
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
```

## 工作原理

### RAG 流程图

```
用户查询
    │
    ▼
┌─────────────────┐
│  向量化查询     │ ← 使用嵌入模型
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  向量相似度检索  │ ← PostgreSQL + pgvector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  返回相关文档片段 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  (可选) 生成回答 │ ← 结合 LLM
└─────────────────┘
```

### 文档处理流程

```
用户上传文档
    │
    ▼
┌──────────────┐
│  文档加载    │ ← PyPDFLoader / TextLoader 等
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  文本提取    │ ← 从文档中提取纯文本
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  文本切片    │ ← RecursiveCharacterTextSplitter
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  向量化      │ ← 嵌入模型
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  存储到数据库 │ ← PostgreSQL + pgvector
└──────────────┘
```

## 配置说明

### 环境变量详解

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| DB_HOST | localhost | 数据库主机地址 |
| DB_PORT | 5432 | 数据库端口 |
| DB_USER | postgres | 数据库用户名 |
| DB_PASSWORD | postgres | 数据库密码 |
| DB_NAME | rag_db | 数据库名称 |
| VECTOR_TABLE_NAME | rag_documents | 向量存储表名 |
| EMBEDDING_MODEL_TYPE | local | 嵌入模型类型 (local/openai) |
| EMBEDDING_MODEL_NAME | all-MiniLM-L6-v2 | 嵌入模型名称 |
| OPENAI_API_KEY | 无 | OpenAI API Key |
| CHUNK_SIZE | 1000 | 文档切片大小（字符数） |
| CHUNK_OVERLAP | 200 | 切片重叠大小 |
| FLASK_HOST | 0.0.0.0 | Flask 绑定地址 |
| FLASK_PORT | 5000 | Flask 监听端口 |
| FLASK_DEBUG | true | 调试模式 |
| MAX_CONTENT_LENGTH | 104857600 | 最大上传文件大小（100MB） |
| UPLOAD_FOLDER | uploads | 上传文件存储目录 |

### 嵌入模型维度

| 模型 | 维度 |
|------|------|
| all-MiniLM-L6-v2 | 384 |
| all-mpnet-base-v2 | 768 |
| text-embedding-ada-002 | 1536 |
| text-embedding-3-large | 3072 |

**注意：** 更换嵌入模型时，需要确保向量数据库中存储的向量维度与新模型一致。

## 常见问题

### Q1: 如何安装 pgvector 扩展？

**方法一：使用 PostgreSQL 包管理器**

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-16-pgvector

# 或使用源码安装
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

**方法二：验证安装**

```sql
-- 连接到 PostgreSQL
psql -U postgres

-- 检查是否已安装
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- 安装扩展
CREATE EXTENSION vector;

-- 验证
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

### Q2: 首次运行时数据库连接失败？

可能的原因：

1. **PostgreSQL 服务未运行**
   ```bash
   # 检查服务状态
   sudo systemctl status postgresql
   
   # 启动服务
   sudo systemctl start postgresql
   ```

2. **网络访问限制**
   - 检查 PostgreSQL 配置 `pg_hba.conf`
   - 确保允许来自 WSL 的连接

3. **数据库不存在**
   - 系统会自动尝试创建数据库
   - 如失败，手动创建：`CREATE DATABASE rag_db;`

### Q3: 嵌入模型下载慢？

本地模型首次使用时需要从 HuggingFace 下载。可以设置镜像加速：

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在 Python 代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q4: 如何切换到 OpenAI 嵌入模型？

1. 在 `.env` 文件中设置：
   ```env
   EMBEDDING_MODEL_TYPE=openai
   EMBEDDING_MODEL_NAME=text-embedding-ada-002
   OPENAI_API_KEY=sk-your-api-key-here
   ```

2. 注意：
   - 切换模型后，之前存储的向量可能无法使用
   - 建议切换前清空数据库或使用新的表名

### Q5: 支持哪些文档格式？

当前支持：

| 格式 | 扩展名 | 加载器 |
|------|--------|--------|
| PDF | .pdf | PyPDFLoader |
| Markdown | .md, .markdown | UnstructuredMarkdownLoader |
| 纯文本 | .txt | TextLoader |

**扩展支持其他格式：**

可以在 `src/document/processor.py` 中添加新的加载器：

```python
# 例如添加 Word 文档支持
from langchain_community.document_loaders import Docx2txtLoader

# 在 SupportedFormats 枚举中添加
DOCX = "docx"

# 在 DocumentLoader.loaders 中映射
self.loaders = {
    # ...
    SupportedFormats.DOCX: Docx2txtLoader,
}
```

## 性能优化建议

### 1. 向量索引优化

pgvector 支持多种索引类型：

```sql
-- IVFFlat 索引（适用于精确搜索）
CREATE INDEX ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- HNSW 索引（适用于近似最近邻搜索，性能更好）
CREATE INDEX ON langchain_pg_embedding 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**索引选择建议：**

| 索引类型 | 构建速度 | 搜索速度 | 内存占用 | 适用场景 |
|----------|----------|----------|----------|----------|
| IVFFlat | 快 | 一般 | 低 | 数据量较小、需要精确搜索 |
| HNSW | 慢 | 快 | 高 | 数据量大、追求性能 |

### 2. 嵌入模型选择

| 需求 | 推荐模型 |
|------|----------|
| 快速测试 | all-MiniLM-L6-v2 (384维) |
| 更好效果 | all-mpnet-base-v2 (768维) |
| 生产环境 | OpenAI text-embedding-ada-002 |

### 3. 切片参数调优

根据文档类型调整切片参数：

```python
# 技术文档：较小的切片，更多重叠
processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

# 普通文章：适中的切片
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# 长文本：较大的切片
processor = DocumentProcessor(chunk_size=2000, chunk_overlap=300)
```

## 扩展开发

### 1. 添加新的嵌入模型

在 `src/embedding/embedder.py` 中扩展 `EmbeddingFactory`：

```python
@classmethod
def _create_custom_embedding(cls, config: EmbeddingConfig):
    # 实现自定义嵌入模型
    from langchain_core.embeddings import Embeddings
    
    class CustomEmbeddings(Embeddings):
        def embed_documents(self, texts):
            # 实现文档向量化
            pass
        
        def embed_query(self, text):
            # 实现查询向量化
            pass
    
    return CustomEmbeddings()
```

### 2. 添加新的文档加载器

在 `src/document/processor.py` 中：

```python
# 1. 在 SupportedFormats 枚举中添加新格式
EPUB = "epub"

# 2. 导入新的加载器
from langchain_community.document_loaders import UnstructuredEPubLoader

# 3. 在 DocumentLoader.__init__ 中映射
self.loaders = {
    # ...
    SupportedFormats.EPUB: UnstructuredEPubLoader,
}

# 4. 在 SupportedFormats.from_extension 中添加扩展名映射
elif ext in ["epub"]:
    return cls.EPUB
```

### 3. 扩展 API 接口

在 `src/api/routes.py` 中添加新路由：

```python
@bp.route("/documents", methods=["GET"])
def list_documents():
    """获取已上传的文档列表"""
    # 实现逻辑
    return jsonify({"success": True, "data": []})

@bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """删除指定文档"""
    # 实现逻辑
    return jsonify({"success": True})
```

## 部署建议

### 开发环境

```bash
# 使用 Flask 内置服务器（调试模式）
python app.py
```

### 生产环境

推荐使用 Gunicorn + Nginx：

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动（4 个工作进程）
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app
```

**Nginx 配置示例：**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /path/to/public-rag/static;
        expires 30d;
    }
}
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0 (2026-04-23)

- 初始版本发布
- 实现文档上传功能
- 实现向量相似度检索
- 实现 Web 前端界面
- 支持 PDF、Markdown、纯文本格式
- 支持本地嵌入模型和 OpenAI 嵌入模型
- 集成 PostgreSQL + pgvector 向量数据库
