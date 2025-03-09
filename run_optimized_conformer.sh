#!/bin/bash

# Run training with the optimized conformer configuration
# Focuses on reducing insertion errors through aggressive regularization and masking

python -m emg2qwerty.train \
  user="single_user" \
  model=conformer_ctc \
  optimizer=adamw_transformer \
  lr_scheduler=stepped_warmup \
  decoder=ctc_greedy \
  +trainer.gradient_clip_val=0.5 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=200 \
  batch_size=128 \
  seed=42 