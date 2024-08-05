FILE_NAME=$1
PROMPT_COLUMN=$2
TARGET_COLUMN=$3
MODEL_DIR=$4
MODEL_NAME=$5

DATA_DIR=../../../data
OUTPUT_DIR=../../../ckpts


SOURCE_LENGTH=512
TARGET_LENGTH=512
for LR in 5e-5 
do
for file_name in FILE_NAME
do
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --num_gpus=8 --master_port $MASTER_PORT train_chatglm2.py \
    --deepspeed deepspeed/ds_config_zero2_bf16.json \
    --bf16 True \
    --do_train \
    --train_file ${DATA_DIR}/${FILE_NAME}.jsonl \
    --prompt_column ${PROMPT_COLUMN} \
    --response_column ${TARGET_COLUMN} \
    --overwrite_cache \
    --model_name_or_path ${MODEL_DIR}/${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}/${MODEL_NAME}_${FILE_NAME}-8-1-16_${SOURCE_LENGTH}_${TARGET_LENGTH}_$LR \
    --overwrite_output_dir \
    --max_source_length ${SOURCE_LENGTH} \
    --max_target_length ${TARGET_LENGTH} \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100000 \
    --learning_rate $LR \
    --report_to "none" \
    --cache_dir /pfs-LLM/common/edu_workspace/cache
    wait
done
done


# nohup bash train_chatglm2.sh high_precision_train_data question raw_analysis /pfs-LLM/common/edu_workspace/Math/checkpoint chatglm2-6b > myout.file 2>&1 &



