#!/bin/bash

# Enhanced Freeze-Omni Demo Test Script

echo "🚀 Starting Enhanced Freeze-Omni Demo Server..."
echo "📊 This version includes real-time state prediction visualization"
echo ""

# Define the config file path
CONFIG_FILE="/home/eeyifanshen/e2e_audio_LLM/dialog_turntaking_new/Freeze-Omni/configs/server_config.yaml"
# LLM_PATH_FROM_CONFIG=$(grep 'llm_path:' $CONFIG_FILE | awk '{print $2}')


# # Check if required paths exist
# if [ ! -f "$CONFIG_FILE" ]; then
#     echo "❌ Error: Config file not found at $CONFIG_FILE"
#     exit 1
# fi

# if [ ! -d "checkpoints" ]; then
#     echo "❌ Error: checkpoints directory not found"
#     echo "Please make sure you're running this from the Freeze-Omni root directory"
#     exit 1
# fi

# if [ ! -d "$LLM_PATH_FROM_CONFIG" ]; then
#     echo "❌ Error: LLM directory '$LLM_PATH_FROM_CONFIG' not found"
#     echo "Please make sure the LLM model is downloaded and path is correct in $CONFIG_FILE"
#     exit 1
# fi

echo "🔧 Loading configuration from: $CONFIG_FILE"
echo ""

# Extract server info for display
IP=$(grep 'ip:' $CONFIG_FILE | awk '{print $2}')
PORT=$(grep 'port:' $CONFIG_FILE | awk '{print $2}')

echo "🌐 Access URLs:"
echo "   Monitoring dialogue state: https://$IP:$PORT/monitor"
echo ""


# # On local machine: forward remote port
# ssh -L 8765:localhost:8765 eez170.ece.ust.hk


# Start the server
# Note: model_path and llm_path are now primarily controlled by the config file.
# You can still override them here with --model_path or --llm_path if needed.
CUDA_VISIBLE_DEVICES=1 python bin/server_alt.py --config "$CONFIG_FILE"