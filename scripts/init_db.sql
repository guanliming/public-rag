-- PostgreSQL 数据库初始化脚本
-- =================================
-- 此脚本用于初始化 RAG 项目所需的数据库和扩展
-- 执行方式: psql -U postgres -h 172.19.249.222 -f init_db.sql

-- 注意: 此脚本需要以具有创建数据库权限的用户执行（如 postgres）

-- 1. 创建数据库（如果不存在）
-- PostgreSQL 不支持在事务块内创建数据库，所以需要单独执行
-- CREATE DATABASE rag_db;

-- 连接到刚创建的数据库后，执行以下命令：

-- 2. 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. 验证扩展是否安装成功
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- 4. 创建向量存储表（LangChain PGVector 会自动创建，这里仅作参考）
-- 实际使用时，LangChain 会自动创建以下结构：
--
-- CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
--     id UUID PRIMARY KEY,
--     collection_id UUID,
--     embedding vector(384),  -- 维度取决于嵌入模型
--     document VARCHAR,
--     cmetadata JSONB
-- );
--
-- CREATE TABLE IF NOT EXISTS langchain_pg_collection (
--     id UUID PRIMARY KEY,
--     name VARCHAR,
--     cmetadata JSONB,
--     UNIQUE(name)
-- );

-- 5. 创建测试用的文档管理表（可选，用于追踪已上传的文档）
CREATE TABLE IF NOT EXISTS uploaded_documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    stored_name VARCHAR(255) NOT NULL,
    file_path TEXT,
    file_size BIGINT,
    file_type VARCHAR(50),
    total_chunks INTEGER DEFAULT 0,
    document_ids TEXT[],  -- 存储向量数据库中的 ID 列表
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending'  -- pending, processing, completed, failed
);

-- 6. 添加注释
COMMENT ON TABLE uploaded_documents IS '已上传的文档记录表';
COMMENT ON COLUMN uploaded_documents.file_name IS '原始文件名';
COMMENT ON COLUMN uploaded_documents.stored_name IS '存储的唯一文件名';
COMMENT ON COLUMN uploaded_documents.document_ids IS '向量数据库中的文档ID列表';
COMMENT ON COLUMN uploaded_documents.status IS '处理状态';

-- 7. 创建索引以加速查询
CREATE INDEX IF NOT EXISTS idx_uploaded_documents_file_name 
ON uploaded_documents(file_name);

CREATE INDEX IF NOT EXISTS idx_uploaded_documents_uploaded_at 
ON uploaded_documents(uploaded_at);

-- 完成提示
SELECT '数据库初始化完成!' AS message;
SELECT '请确保 pgvector 扩展已正确安装' AS note;
