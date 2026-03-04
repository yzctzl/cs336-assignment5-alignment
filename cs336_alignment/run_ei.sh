#!/bin/bash
# Expert Iteration experiments on MATH dataset
# Vary: ei_batch_size in {512, 1024, 2048}, num_rollouts (G), sft_epochs
# Fixed: n_ei_steps=5, max_grad_norm=1.0

set -e

MODEL="models/Qwen2.5-Math-1.5B"
TRAIN_DATA="data/MATH/train.jsonl"
VAL_DATA="data/MATH/validation.jsonl"
PROMPT="cs336_alignment/prompts/r1_zero.prompt"
# To prevent running too long, we use smaller n_ei_steps and a smaller n_ei_steps for all scripts
# n_ei_steps = 3 instead of 5

# --- Experiment 1: vary batch size (G=8, epochs=4) ---
# Vary the batch size for each expert iteration step (i.e., the size of Db) in {512, 1024, 2048}
for BS in 512 1024 2048; do
  echo "=== EI: ei_batch_size=${BS}, G=8, epochs=4 ==="
  python -m cs336_alignment.expert_iter \
    --model_name_or_path "$MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --prompt_path "$PROMPT" \
    --n_ei_steps 3 \
    --ei_batch_size "$BS" \
    --num_rollouts 8 \
    --sft_epochs 4 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 1 \
    --output_dir "output/ei/bs${BS}_G8_ep4" \
    --run_name "ei_bs${BS}_G8_ep4"
done

# --- Experiment 2: varying the number of rollouts G per question (batch_size=1024, epochs=4) ---
for G in 4 8 16; do
  echo "=== EI: ei_batch_size=1024, G=${G}, epochs=4 ==="
  python -m cs336_alignment.expert_iter \
    --model_name_or_path "$MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --prompt_path "$PROMPT" \
    --n_ei_steps 3 \
    --ei_batch_size 1024 \
    --num_rollouts "$G" \
    --sft_epochs 4 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 1 \
    --output_dir "output/ei/bs1024_G${G}_ep4" \
    --run_name "ei_bs1024_G${G}_ep4"
done

# --- Experiment 3: varying the number of epochs used in the SFT step (batch_size=1024, G=8) ---
for EP in 2 4 8; do
  echo "=== EI: ei_batch_size=1024, G=8, epochs=${EP} ==="
  python -m cs336_alignment.expert_iter \
    --model_name_or_path "$MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --prompt_path "$PROMPT" \
    --n_ei_steps 3 \
    --ei_batch_size 1024 \
    --num_rollouts 8 \
    --sft_epochs "$EP" \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 1 \
    --output_dir "output/ei/bs1024_G8_ep${EP}" \
    --run_name "ei_bs1024_G8_ep${EP}"
done

echo "All EI experiments complete."
