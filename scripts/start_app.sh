#!/bin/bash
# ============================================
# RAG 测试项目 - 启动脚本
# ============================================
# 此脚本用于在 WSL 环境中启动 Flask 应用
# 
# 使用方法:
#   bash scripts/start_app.sh
#
# 环境变量说明 (在 .env 文件中配置):
#   - FLASK_HOST: 绑定地址 (默认 0.0.0.0)
#   - FLASK_PORT: 监听端口 (默认 5000)
#   - FLASK_DEBUG: 调试模式 (默认 true)
# ============================================

set -e  # 遇到错误立即退出

echo "============================================"
echo "RAG 测试项目 - 启动脚本"
echo "============================================"
echo ""

# 定义 Python 路径
PYTHON_PATH="/home/dawn/miniconda3/envs/temp312/bin/python"

# 检查 Python 环境
echo "[1/3] 检查 Python 环境..."
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ 未找到 Python 环境: $PYTHON_PATH"
    exit 1
fi
echo "✅ Python 环境就绪"
$PYTHON_PATH --version

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APP_FILE="$PROJECT_ROOT/app.py"
ENV_FILE="$PROJECT_ROOT/.env"

# 检查应用文件
echo ""
echo "[2/3] 检查项目文件..."
if [ ! -f "$APP_FILE" ]; then
    echo "❌ 未找到 app.py: $APP_FILE"
    exit 1
fi
echo "✅ 应用文件就绪: $APP_FILE"

# 检查环境配置
if [ -f "$ENV_FILE" ]; then
    echo "✅ 环境配置文件就绪: $ENV_FILE"
    # 读取环境变量用于显示
    FLASK_HOST=$(grep -E "^FLASK_HOST=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' || echo "0.0.0.0")
    FLASK_PORT=$(grep -E "^FLASK_PORT=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '"' || echo "5000")
else
    echo "⚠️  未找到 .env 文件，将使用默认配置"
    FLASK_HOST="0.0.0.0"
    FLASK_PORT="5000"
fi

echo ""
echo "[3/3] 启动 Flask 应用..."
echo ""
echo "============================================"
echo "应用配置:"
echo "  - 绑定地址: $FLASK_HOST"
echo "  - 监听端口: $FLASK_PORT"
echo "  - 访问地址: http://localhost:$FLASK_PORT"
echo "============================================"
echo ""
echo "启动中... (按 Ctrl+C 停止)"
echo ""

# 切换到项目目录并启动应用
cd "$PROJECT_ROOT"
exec "$PYTHON_PATH" "$APP_FILE"
