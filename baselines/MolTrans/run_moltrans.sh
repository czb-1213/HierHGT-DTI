#!/bin/bash

# 定义任务列表
TASKS=(
    "drugbank_cold_protein"
)

# 定义输出目录
OUTPUT_DIR="./model/output_moltrans"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 最大并行运行数
MAX_PARALLEL=1

# 当前运行的任务数
RUNNING=0

# 任务ID
TASK_ID=0

# 循环处理所有任务
for TASK in "${TASKS[@]}"; do
    # 等待直到有空闲槽位
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        sleep 60
        # 检查后台任务状态
        RUNNING=$(jobs -p | wc -l)
    done
    
    echo "启动任务: $TASK"
    
    # 使用nohup后台运行任务
    nohup python baselines/MolTrans/train.py \
        --task "$TASK" \
        --output_dir "$OUTPUT_DIR" \
        --batch-size 16 \
        --epochs 50 \
        --lr 1e-4 \
        > "$OUTPUT_DIR/${TASK}_log.txt" 2>&1 &
    
    # 增加运行计数
    RUNNING=$((RUNNING + 1))
    TASK_ID=$((TASK_ID + 1))
    
    echo "任务 $TASK_ID: $TASK 已启动，PID: $!"
    
    # 短暂延迟，避免同时启动多个任务
    sleep 10
done

# 等待所有任务完成
echo "所有任务已启动，等待完成..."
wait
echo "所有任务已完成!"
