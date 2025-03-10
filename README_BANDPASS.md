# EMG Bandpass Filtering

This extension adds bandpass filtering to the EMG2QWERTY project, focusing the signal on the most relevant frequency range for muscle activity (20-450 Hz).

## Benefits of Bandpass Filtering for EMG

1. **Noise Reduction**: Removes high-frequency noise and low-frequency motion artifacts
2. **Improved Signal-to-Noise Ratio**: Focuses on the frequency range where EMG information is concentrated
3. **Better Feature Extraction**: Cleaner signals lead to more meaningful spectrogram features
4. **Reduced Baseline Drift**: Eliminates low-frequency baseline wander caused by electrode movement

## Implementation Details

The bandpass filter is implemented in the frequency domain using FFT for computational efficiency:

1. Convert the signal to the frequency domain using FFT
2. Apply a bandpass mask with smooth transitions to avoid ringing artifacts
3. Convert back to the time domain using inverse FFT

Default parameters:
- Low cutoff frequency: 20 Hz
- High cutoff frequency: 450 Hz
- Sampling rate: 2000 Hz
- Transition width: 10% of cutoff frequencies

## Usage

### Training with Bandpass Filtering

To train a model with bandpass filtering, run:

```bash
python -m emg2qwerty.train transforms=bandpass_filter
```

### Testing the Filter

To visualize the effect of bandpass filtering on EMG signals:

```bash
python scripts/test_bandpass.py path/to/your/session_file.h5 --output filter_viz.png
```

Optional arguments:
- `--low_cut`: Lower cutoff frequency (default: 20.0 Hz)
- `--high_cut`: Upper cutoff frequency (default: 450.0 Hz)
- `--channel`: EMG channel to visualize (default: 0)
- `--window_length`: Number of samples to analyze (default: 2000)

### Customizing Parameters

You can customize the bandpass filter parameters:

```bash
python -m emg2qwerty.train transforms=bandpass_filter bandpass.low_cut=15 bandpass.high_cut=500
```

## Baseline vs. Bandpass Filter

The baseline model uses the original EMG signals without bandpass filtering. To run the baseline:

```bash
python -m emg2qwerty.train
```

This will use the default `log_spectrogram` transform without bandpass filtering. 