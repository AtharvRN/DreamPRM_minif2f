# Example configuration for different training scenarios

## Single GPU Development (LoRA)
```bash
python launch_training.py \
    --use_lora \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr 2e-4 \
    --total_steps 1000 \
    --train_json_file prm_minif2f_valid_train.json \
    --meta_json_file prm_minif2f_valid_train.json
```

## Multi-GPU Production (4 GPUs with LoRA)
```bash
python launch_training.py \
    --num_gpus 4 \
    --use_lora \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --total_steps 5000 \
    --lora_r 32 \
    --lora_alpha 64 \
    --precision bf16
```

## High-End Full Fine-tuning (8 GPUs)
```bash
python launch_training.py \
    --num_gpus 8 \
    --no_lora \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr 1e-4 \
    --total_steps 3000 \
    --precision bf16 \
    --max_length 2048
```

## Quick Test Run
```bash
python launch_training.py \
    --use_lora \
    --batch_size 1 \
    --total_steps 10 \
    --dry_run
```