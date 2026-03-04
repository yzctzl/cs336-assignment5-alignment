#!/bin/bash
set -e
# For SFT on subamples we want to train for a bit longer to see effect. Let's do 2 epochs.
# Because the GPU is 40GB, batch size is 1 or 2. We'll stick to 1 with grad acc 16 to simulate global batch 16.
# We'll evaluate every 20 steps.

for size in 256 512 1024 -1; do
    echo "======================================"
    echo "Running SFT with dataset size $size"
    echo "======================================"
    
    uv run python cs336_alignment/train_sft.py \
    --dataset_size $size \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-5 \
    --epochs 2 \
    --run_name "sft_size_${size}" \
    --output_dir "output/sft_size_${size}"
done
