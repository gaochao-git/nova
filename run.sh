#!/bin/bash

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file with your actual API keys"
    exit 1
fi

# 启动服务
echo "Starting server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001