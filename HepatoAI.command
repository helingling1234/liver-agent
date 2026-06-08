#!/bin/bash
# HepatoAI 双击启动器
# 在 Finder 中双击此文件即可启动

DIR="/Users/lingling/AI/liver agent"
cd "$DIR"

echo "🔬 HepatoAI 肝病AI助手"
echo "========================"

# 启动 Ollama（如果没有运行）
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "⚡ 正在启动 Ollama..."
    ollama serve > /tmp/ollama_hepatoai.log 2>&1 &
    sleep 3
    echo "✅ Ollama 已启动"
fi

# 检查端口是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t > /dev/null 2>&1; then
    echo "✅ 服务已在运行"
else
    echo "🚀 启动 HepatoAI 服务..."
    uvicorn web_app:app --port 8000 > /tmp/hepatoai.log 2>&1 &
    sleep 2
fi

# 打开浏览器
echo "🌐 正在打开浏览器..."
open http://localhost:8000
echo ""
echo "✅ HepatoAI 已启动！"
echo "   浏览器地址: http://localhost:8000"
echo "   关闭此窗口不影响使用（服务在后台运行）"
echo "   要停止服务请运行: pkill -f 'uvicorn web_app'"
