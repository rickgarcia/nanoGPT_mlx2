#!/bin/bash
# Train GPT-2 (124M) on OpenWebText using MLX
# Automatically resumes from checkpoint if one exists

python train.py configs/train_gpt2_owt.py --gradient_accumulation_steps=128
