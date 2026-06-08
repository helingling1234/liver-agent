#!/bin/bash
# HepatoAI 一键启动脚本（Ollama 本地版）
# 用法：bash "/Users/lingling/AI/liver agent/start.sh"

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "🔬 HepatoAI 启动中..."

# 检查 Ollama 是否运行
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "⚡ 启动 Ollama 服务..."
    ollama serve &
    sleep 3
fi

# 检查模型是否存在
if ! ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    echo "📥 下载模型 qwen2.5:7b（首次运行需要下载约 4.7GB）..."
    ollama pull qwen2.5:7b
fi

echo "✅ Ollama 运行中"
echo "🌐 打开浏览器访问: http://localhost:8000"
echo "⏹  按 Ctrl+C 停止"
echo ""

uvicorn web_app:app --port 8000
