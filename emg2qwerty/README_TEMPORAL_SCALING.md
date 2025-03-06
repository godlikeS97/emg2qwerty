# Temporal Scaling Augmentation for EMG2QWERTY

## Overview

The Temporal Scaling augmentation is a powerful data augmentation technique that simulates natural variations in typing speed and rhythm. It works by randomly stretching or compressing the time dimension of EMG signals, creating more diverse training examples from limited data.

This feature is particularly valuable for single-user models where training data may be limited.

## How It Works

The `TemporalScaling` transform:

1. Randomly selects a scaling factor between `min_scale` and `max_scale`
2. Applies time-domain interpolation to stretch or compress the signal
3. Preserves the original sequence length through appropriate cropping or padding
4. Maintains all channel information without distortion

## Benefits for Single-User Models

- **Typing Speed Variations**: Simulates the same user typing at different speeds
- **Rhythm Adaptation**: Makes the model robust to changes in typing rhythm that naturally occur due to fatigue or context
- **Data Amplification**: Effectively increases training data diversity without requiring more recording sessions
- **Session Generalization**: Improves model performance across sessions where the user's typing speed might differ

## Configuration Options

The transform can be configured with these parameters:

- `min_scale`: Minimum scaling factor (values < 1.0 slow down the signal)
- `max_scale`: Maximum scaling factor (values > 1.0 speed up the signal)
- `time_dim`: The dimension to scale (usually 0 for time-first tensors)

## How to Use

### Standard Training (No Temporal Scaling)

To use the original configuration without temporal scaling:

```bash
python -m emg2qwerty.train \
  user="single_user" \
  transforms=log_spectrogram \
  trainer.accelerator=gpu trainer.devices=1
```

### With Temporal Scaling

To enable temporal scaling with enhanced augmentation:

```bash
python -m emg2qwerty.train \
  user="single_user" \
  transforms=temporal_scaling \
  trainer.accelerator=gpu trainer.devices=1
```

The `temporal_scaling.yaml` configuration includes:
- Temporal scaling with range of 0.85-1.15 (±15% speed variation)
- Enhanced electrode rotation options (±2 positions)
- Increased temporal jittering (75ms vs 60ms in standard config)
- More aggressive SpecAugment parameters

### Custom Configuration

You can also use parameter overrides to customize scaling:

```bash
python -m emg2qwerty.train \
  user="single_user" \
  transforms=temporal_scaling \
  transforms.temporal_scaling.min_scale=0.8 \
  transforms.temporal_scaling.max_scale=1.2
```

## Tips for Best Results

- **Combine with Other Augmentations**: Temporal scaling works best when combined with other augmentations like band rotation and SpecAugment (already configured in `temporal_scaling.yaml`)
- **Tune the Range**: If performance is not improving, try adjusting the min/max scale values
- **Apply Early**: In our implementation, temporal scaling is applied early in the transform pipeline, before frequency-domain transformations
- **Monitor Validation**: Watch validation metrics closely, as excessive augmentation can sometimes harm performance

## Implementation Details

The transform is implemented in `emg2qwerty/transforms.py` using PyTorch's interpolation functionality. It preserves tensor shapes and handles edge cases appropriately, making it compatible with the rest of the EMG2QWERTY pipeline. 