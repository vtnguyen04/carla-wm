#!/bin/bash

# Check if a port argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> [additional_training_parameters]"
    exit 1
fi

# Configuration
CARLA_PORT=$1
GPU_DEVICE=$2
TM_PORT=$((CARLA_PORT + 6001))
MONITOR_PORT=$((CARLA_PORT + 7000))
LOG_FILE="log_${CARLA_PORT}.log"

# Environment Setup
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Adjust CARLA root path if necessary
CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10 -quality-level=Low"
TRAINING_SCRIPT="-m torch_wm.rl.train"

# Updated for Twister/PyTorch
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --device cuda --seed 0"
ADDITIONAL_PARAMS="${@:3}"  # Capture all additional parameters
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

TRAINING_COMMAND="PATH=/usr/bin:$PATH uv run python $TRAINING_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

# Clear log file before starting
> $LOG_FILE

# Function to log messages with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# Function to start or restart CARLA
launch_carla() {
    # Check if CARLA is running on the specific port
    if ! nc -z localhost $CARLA_PORT; then
        log_with_timestamp "CARLA server is not running on port $CARLA_PORT. Starting or restarting..."
        # Kill any existing processes on these ports to be safe
        fuser -k ${CARLA_PORT}/tcp >/dev/null 2>&1
        fuser -k ${TM_PORT}/tcp >/dev/null 2>&1
        fuser -k ${MONITOR_PORT}/tcp >/dev/null 2>&1

        # Start CARLA
        # Note: Ensure CARLA_ROOT is set in your environment
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &

        # Wait for CARLA to fully start
        while ! nc -z localhost $CARLA_PORT; do
            log_with_timestamp "Waiting for CARLA server to start on port $CARLA_PORT..."
            sleep 2
        done
        log_with_timestamp "CARLA server is up and running on port $CARLA_PORT."
    fi
}

# Function to start the training script
start_training() {
    # start_training should NOT call launch_carla because launch_carla is already in the main loop
    # or we can keep it here for initial start.

    # Start the training script in background
    eval $TRAINING_COMMAND >> $LOG_FILE 2>&1 &
    TRAINING_PID=$!
    log_with_timestamp "Training session started successfully (PID: $TRAINING_PID). Logs: $LOG_FILE"
    echo -e "\033[1;32mTraining started (Twister/PyTorch). Port: $CARLA_PORT, GPU: $GPU_DEVICE\033[0m"
}

# Function to clean up processes on exit
cleanup() {
    log_with_timestamp "Cleaning up and exiting..."
    # kill -TERM $TRAINING_PID >/dev/null 2>&1
    # Find and kill any uv run RL/train.py processes related to this port
    pkill -f "$TRAINING_SCRIPT .*--env.world.carla_port $CARLA_PORT"
    exit
}

trap cleanup SIGINT SIGTERM

# Main loop
launch_carla
start_training

while true; do
    # Check if the training script is still running
    if ! ps -p $TRAINING_PID > /dev/null; then
        log_with_timestamp "Training script crashed or finished. Restarting..."
        start_training
    fi

    # Periodic CARLA health check
    if ! nc -z localhost $CARLA_PORT; then
        log_with_timestamp "CARLA server lost. Restarting everything..."
        pkill -f "$TRAINING_SCRIPT .*--env.world.carla_port $CARLA_PORT"
        launch_carla
        start_training
    fi

    sleep 60
done
