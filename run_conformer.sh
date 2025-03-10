#!/bin/bash

# Run training with Conformer model
python -m emg2qwerty.train \
  user="single_user" \
  model=conformer_ctc \
  optimizer=adamw_transformer \
  lr_scheduler=transformer_warmup \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  +trainer.gradient_clip_val=1.0 \
  trainer.max_epochs=200 \
  batch_size=64 \
  seed=42 \
  +callback=decoding_visualization 