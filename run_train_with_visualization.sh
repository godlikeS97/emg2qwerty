#!/bin/bash

# Run training with CNN model and visualization
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=2 \
  batch_size=256 \
  +callback=decoding_visualization 