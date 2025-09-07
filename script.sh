create_scripts() {
    echo "📝 创建启动脚本..."
    
    # 确保在项目根目录
    PROJECT_ROOT=$(pwd)
    
    # 在NeMo-Agent-Toolkit目录中创建启动脚本
    cd NeMo-Agent-Toolkit
    
    # 创建启动脚本
    cat > start.sh << 'EOF'
#!/bin/bash

echo "🚀 启动 NVIDIA NeMo Agent Toolkit AI对话机器人"
echo "=============================================="

# 获取项目根目录和NeMo目录
NEMO_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$NEMO_DIR")

# 设置环境变量
export TAVILY_API_KEY=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

# 激活Python虚拟环境
source .venv/bin/activate

# 启动后端服务
echo "📡 启动后端服务..."
aiq serve --config_file configs/hackathon_config.yml --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 10

# 启动前端服务
echo "🎨 启动前端服务..."
cd "$PROJECT_ROOT/external/aiqtoolkit-opensource-ui"
npm run dev &
FRONTEND_PID=$!

# 返回NeMo目录
cd "$NEMO_DIR"

echo ""
echo "✅ 系统启动完成！"
echo ""
echo "🌐 访问地址:"
echo "   前端界面: http://localhost:3000"
echo "   API文档:  http://localhost:8001/docs"
echo ""
echo "📝 测试建议:"
echo "   1. 天气查询: '北京今天的天气怎么样，气温是多少？'"
echo "   2. 公司信息: '帮我介绍一下NVIDIA Agent Intelligence Toolkit'"
echo "   3. 时间查询: '现在几点了？'"
echo ""
echo "🛑 停止服务: 按 Ctrl+C 或运行 ./stop.sh"
echo ""

# 保存进程ID
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# 等待用户中断
wait
EOF

    # 创建停止脚本
    cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 停止 NVIDIA NeMo Agent Toolkit AI对话机器人"
echo "=============================================="

# 停止后端服务
if [ -f .backend.pid ]; then
    BACKEND_PID=$(cat .backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        echo "停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
    fi
    rm -f .backend.pid
fi

# 停止前端服务
if [ -f .frontend.pid ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
    fi
    rm -f .frontend.pid
fi

# 清理其他相关进程
pkill -f "aiq serve" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true

echo "✅ 所有服务已停止"
EOF

    # 添加执行权限
    chmod +x start.sh stop.sh
    
    echo "✅ 启动脚本创建完成"
    
    # 返回项目根目录
    cd "$PROJECT_ROOT"
}

# 主安装流程
main() {
    detect_os
    check_requirements
    install_uv
    setup_project
    setup_frontend
    create_config
    create_scripts
    
    echo ""
    echo "🎉 安装完成！"
    echo "=============="
    echo ""
    echo "📁 项目根目录: $(pwd)"
    echo "📁 NeMo项目目录: $(pwd)/NeMo-Agent-Toolkit"
    echo ""
    echo "🚀 快速启动:"
    echo "   cd NeMo-Agent-Toolkit && ./start.sh"
    echo ""
    echo "🛑 停止服务:"
    echo "   cd NeMo-Agent-Toolkit && ./stop.sh"
    echo ""
    echo "⚙️  自定义配置:"
    echo "   编辑 NeMo-Agent-Toolkit/configs/hackathon_config.yml 文件"
    echo "   可修改 API密钥、模型名称、base_url 等"
    echo ""
    echo "📚 更多信息:"
    echo "   https://github.com/NVIDIA/NeMo-Agent-Toolkit"
    echo ""
}

# 运行主函数
main "$@"