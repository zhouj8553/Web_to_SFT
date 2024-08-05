#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

FILE_NAME=$1
PROMPT_COLUMN=$2
TARGET_COLUMN=$3
MODEL_DIR=$4
MODEL_NAME=$5

# MODEL="Qwen/Qwen1.5-7B" # Set the path if you do not want to load from huggingface directly
# # ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# # See the section for finetuning in README for more information.
# DATA="path_to_data"
OUTPUT_DIR="../../../../ckpts"
DS_CONFIG_PATH="ds_config_zero2.json"
USE_LORA=False
Q_LORA=False
GPUS_PER_NODE=8

# function usage() {
#     echo '
# Usage: bash finetune/finetune_lora_ds.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH] [--use_lora USE_LORA] [--q_lora Q_LORA]
# '
# }

# while [[ "$1" != "" ]]; do
#     case $1 in
#         -m | --model )
#             shift
#             MODEL=$1
#             ;;
#         -d | --data )
#             shift
#             DATA=$1
#             ;;
#         --deepspeed )
#             shift
#             DS_CONFIG_PATH=$1
#             ;;
#         --use_lora  )
#             shift
#             USE_LORA=$1
#             ;;
#         --q_lora    )
#             shift
#             Q_LORA=$1
#             ;;
#         -h | --help )
#             usage
#             exit 0
#             ;;
#         * )
#             echo "Unknown argument ${1}"
#             exit 1
#             ;;
#     esac
#     shift
# done

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# LR=2e-5
# CUDA_VISIBLE_DEVICES=7 torchrun $DISTRIBUTED_ARGS finetune.py \
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path ${MODEL_DIR}/${MODEL_NAME} \
    --data_path ../../../data/${FILE_NAME}.jsonl \
    --bf16 True \
    --output_dir ${OUTPUT_DIR}/${MODEL_NAME}_${FILE_NAME}-8-1-16_512_5e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "none" \
    --model_max_length 512 \
    --lazy_preprocess True \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --prompt_column ${PROMPT_COLUMN} \
    --response_column ${TARGET_COLUMN} \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH} \


# bash finetune.sh high_precision_train_data question raw_analysis ../../../checkpoint Qwen1.5-7B-Chat

