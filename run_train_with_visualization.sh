#!/bin/bash

# Run training with CNN model and visualization
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=200 \
  batch_size=384 \
  +callback=decoding_visualization 