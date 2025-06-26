#!/bin/bash

# Enhanced Freeze-Omni Demo Test Script

echo "🚀 Starting Enhanced Freeze-Omni Demo Server..."
echo "📊 This version includes real-time state prediction visualization"
echo ""

# Check if required paths exist
if [ ! -d "checkpoints" ]; then
    echo "❌ Error: checkpoints directory not found"
    echo "Please make sure you're running this from the Freeze-Omni root directory"
    exit 1
fi

if [ ! -d "Qwen2-7B-Instruct" ]; then
    echo "❌ Error: Qwen2-7B-Instruct directory not found"
    echo "Please make sure the LLM model is downloaded"
    exit 1
fi

# Default parameters - adjust as needed
MODEL_PATH="./checkpoints"
LLM_PATH="./Qwen2-7B-Instruct"
IP="localhost"
PORT="8765"
MAX_USERS=1
llm_exec_nums=1 # Recommended to set to 1, requires about 15GB GPU memory per exec. Try setting a value greater than 1 on a better GPU to improve concurrency performance.

echo "🔧 Configuration:"
echo "   Model Path: $MODEL_PATH"
echo "   LLM Path: $LLM_PATH"
echo "   Server IP: $IP"
echo "   Server Port: $PORT"
echo "   Max Users: $MAX_USERS"
echo ""

echo "🌐 Access URLs:"
echo "   Original Demo: https://$IP:$PORT/"
echo "   Enhanced Demo: https://$IP:$PORT/enhanced"
echo ""

echo "📝 Features in Enhanced Demo:"
echo "   ✅ Real-time state prediction display"
echo "   ✅ State 1 & State 2 probability bars"
echo "   ✅ System status indicators"
echo "   ✅ Model generation status"
echo "   ✅ VAD status monitoring"
echo ""

echo "🎯 State Meanings:"
echo "   State 0 (Continue): Keep listening to user"
echo "   State 1 (Interrupt): Stop listening, start generation"
echo "   State 2 (End): Stop listening, no generation"
echo ""

echo "⚠️  Note: You may need to accept the self-signed certificate in your browser"
echo ""


# # On local machine: forward remote port
# ssh -L 8765:localhost:8765 eez170.ece.ust.hk


# Start the server
CUDA_VISIBLE_DEVICES=1 python bin/server.py \
    --llm_exec_nums $llm_exec_nums \
    --model_path "$MODEL_PATH" \
    --llm_path "$LLM_PATH" \
    --ip "$IP" \
    --port "$PORT" \
    --max_users "$MAX_USERS"
