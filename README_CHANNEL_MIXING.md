# EMG Cross-Channel Mixing

This extension adds cross-channel mixing to the EMG2QWERTY project to improve model robustness to electrode placement variations and cross-talk effects.

## Benefits of Cross-Channel Mixing for EMG

1. **Simulates Electrode Placement Variations**: Makes the model more robust to slight changes in electrode positioning
2. **Models Cross-Talk**: EMG electrodes often pick up signals from multiple muscles; mixing helps the model learn this cross-talk
3. **Data Augmentation**: Increases the effective size of the dataset by creating new synthetic channel combinations
4. **User Independence**: Improves generalization across different users with varying muscle anatomy

## Implementation Details

Cross-channel mixing creates synthetic channels by blending pairs of channels together:

1. For each selected channel, a different channel from the same band (left/right) is randomly chosen
2. A mixing ratio (α) is sampled from a specified range, typically 0.1-0.3
3. The mixed channel is created as: (1-α) × original + α × other
4. This preserves most of the original signal while introducing controlled variations

Default parameters:
- Mixing ratio range: 0.1-0.3 (subtle mixing)
- Number of channels to mix: 8 out of 16 (preserves some original channels)
- Applied only during training (not for validation or inference)

## Usage

### Training with Cross-Channel Mixing

To train a model with cross-channel mixing, run:

```bash
python -m emg2qwerty.train transforms=cross_channel_mixing
```

### Training with Both Bandpass and Channel Mixing

To use both preprocessing techniques together:

```bash
python -m emg2qwerty.train transforms=bandpass_and_mixing
```

### Testing the Mixing Effect

To visualize the effect of cross-channel mixing on EMG signals:

```bash
python scripts/test_channel_mixing.py path/to/your/session_file.h5 --output mixing_viz.png
```

Optional arguments:
- `--mix_ratio_min`: Minimum mixing ratio (default: 0.1)
- `--mix_ratio_max`: Maximum mixing ratio (default: 0.3)
- `--num_channels`: Number of channels to mix (default: 8)
- `--seed`: Random seed for reproducibility (default: 42)
- `--display_channels`: Comma-separated list of channel indices to visualize (default: "0,1,2,3")

### Customizing Parameters

You can customize the cross-channel mixing parameters:

```bash
python -m emg2qwerty.train transforms=cross_channel_mixing channel_mixing.mix_ratio_range=[0.15,0.4] channel_mixing.num_channels_to_mix=12
```

## Baseline vs. Enhanced Models

- **Baseline Model**: Uses the original EMG signals without special preprocessing
  ```bash
  python -m emg2qwerty.train
  ```

- **Bandpass Filter Model**: Applies frequency domain filtering (20-450 Hz)
  ```bash
  python -m emg2qwerty.train transforms=bandpass_filter
  ```

- **Channel Mixing Model**: Applies synthetic channel mixing
  ```bash
  python -m emg2qwerty.train transforms=cross_channel_mixing
  ```

- **Combined Model**: Applies both bandpass filtering and channel mixing
  ```bash
  python -m emg2qwerty.train transforms=bandpass_and_mixing
  ``` 