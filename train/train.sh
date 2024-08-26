CUDA_VISIBLE_DEVICES=0,1,2,3
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=0
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.1
MAX_STEPS=8000
LORA_RANK=32

MODEL=LLaMA-7B
TASK_NAME=LLaMA-7B-S12-E30-P2-D0-KD

export HF_ENDPOINT=https://hf-mirror.com

deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} --master_port 29501 train_lora_kd.py \
    --bf16 --deepspeed ds_config.json \
    --dataset_name datasets/LLaMA_tokenized_2K \
    --model_name_or_path models/${MODEL} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs 1 --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} --max_grad_norm 0.0 --lora_rank ${LORA_RANK} \
    --save_strategy steps --save_steps 1000 --logging_steps 10 \
    --output_dir output/${TASK_NAME} --report_to tensorboard \
    --do_train --overwrite_output_dir --seed 42