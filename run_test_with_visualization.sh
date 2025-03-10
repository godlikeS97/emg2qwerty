#!/bin/bash

# Run testing with visualization
# Usage: ./run_test_with_visualization.sh [PATH_TO_CHECKPOINT]

# Default checkpoint path if not provided
CHECKPOINT=${1:-"/root/autodl-tmp/emg2qwerty/logs/2025-03-10/18-05-48/checkpoints/last.ckpt"}

python -m emg2qwerty.train \
  user="single_user" \
  checkpoint="$CHECKPOINT" \
  train=False \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  decoder=ctc_greedy \
  +callback=decoding_visualization 