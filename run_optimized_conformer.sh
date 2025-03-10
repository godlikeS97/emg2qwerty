#!/bin/bash

# Run training with optimized Conformer model
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
  model.d_model=384 \
  model.nhead=8 \
  model.num_encoder_layers=8 \
  model.d_ff=2048 \
  model.kernel_size=31 \
  model.expansion_factor=2 \
  model.dropout=0.1 \
  +callback=decoding_visualization 