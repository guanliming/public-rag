#!/bin/bash
# ============================================
# RAG 测试项目 - 依赖安装脚本
# ============================================
# 此脚本用于在 WSL 环境中安装项目所需的所有 Python 依赖
# 
# 使用方法:
#   1. 确保 WSL 环境已配置好
#   2. 在 WSL 终端中运行: bash scripts/install_dependencies.sh
#   3. 或者直接使用指定的 Python 环境执行
#
# Python 环境:
#   - 路径: /home/dawn/miniconda3/envs/temp312/bin
#   - 版本: Python 3.12
# ============================================

set -e  # 遇到错误立即退出

echo "============================================"
echo "RAG 测试项目 - 依赖安装脚本"
echo "============================================"
echo ""

# 定义 Python 路径
PYTHON_PATH="/home/dawn/miniconda3/envs/temp312/bin/python"
PIP_PATH="/home/dawn/miniconda3/envs/temp312/bin/pip"

# 检查 Python 环境
echo "[1/4] 检查 Python 环境..."
if [ -f "$PYTHON_PATH" ]; then
    echo "✅ 找到 Python 环境: $PYTHON_PATH"
    $PYTHON_PATH --version
else
    echo "❌ 未找到 Python 环境: $PYTHON_PATH"
    echo "请检查 Conda 环境是否正确配置"
    exit 1
fi

# 检查 pip
echo ""
echo "[2/4] 检查 pip..."
if [ -f "$PIP_PATH" ]; then
    echo "✅ 找到 pip: $PIP_PATH"
    $PIP_PATH --version
else
    echo "❌ 未找到 pip"
    exit 1
fi

# 升级 pip
echo ""
echo "[3/4] 升级 pip..."
$PIP_PATH install --upgrade pip
echo "✅ pip 升级完成"

# 安装项目依赖
echo ""
echo "[4/4] 安装项目依赖..."
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "从 $REQUIREMENTS_FILE 安装依赖..."
    $PIP_PATH install -r "$REQUIREMENTS_FILE"
    echo ""
    echo "✅ 依赖安装完成!"
else
    echo "❌ 未找到 requirements.txt 文件: $REQUIREMENTS_FILE"
    exit 1
fi

echo ""
echo "============================================"
echo "依赖安装完成!"
echo "============================================"
echo ""
echo "安装的主要依赖:"
echo "  - LangChain: 用于构建 RAG 系统"
echo "  - langchain-postgres: PostgreSQL 向量存储集成"
echo "  - psycopg2-binary: PostgreSQL 数据库驱动"
echo "  - pgvector: PostgreSQL 向量扩展支持"
echo "  - pypdf / pymupdf: PDF 文档解析"
echo "  - sentence-transformers: 本地嵌入模型"
echo "  - Flask: Web 框架"
echo ""
echo "下一步:"
echo "  1. 确保 PostgreSQL 服务运行在 172.19.249.222:5432"
echo "  2. 确保 pgvector 扩展已安装"
echo "  3. 运行数据库初始化脚本"
echo "  4. 启动应用: python app.py"
echo ""
