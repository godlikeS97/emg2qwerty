## Data Pipeline Overview

### Training Pipeline

1. `ToTensor` → 2. `RandomBandRotation` (via ForEach) → 3. `TemporalAlignmentJitter` → 4. `LogSpectrogram` → 5. `SpecAugment`

### Validation/Testing Pipeline

1. `ToTensor` → 2. `LogSpectrogram`

## Detailed Breakdown of Transformations

### 1. ToTensor

```python
@dataclass
class ToTensor:
    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1
```

- **Purpose**: Converts structured numpy arrays into PyTorch tensors
- **Input**: Numpy structured array containing EMG data
- **Process**: Extracts the fields (`emg_left` and `emg_right`) and stacks them along dimension 1
- **Output**: Tensor of shape (T, 2, C) where:
  - T = time steps
  - 2 = left and right arm EMG
  - C = electrode channels (typically 16 per arm)

### 2. RandomBandRotation (Training only)

```python
@dataclass
class RandomBandRotation:
    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1
```

- **Purpose**: Data augmentation by rotating electrode channels
- **Input**: Tensor of shape (T, 2, C)
- **Process**: Shifts electrode channels by a random offset (-1, 0, or 1)
- **Note**: Applied independently to each arm using `ForEach` wrapper to ensure different random rotations
- **Output**: Tensor with same shape, but with shifted channel positions

### 3. TemporalAlignmentJitter (Training only)

```python
@dataclass
class TemporalAlignmentJitter:
    max_offset: int
    stack_dim: int = 1
```

- **Purpose**: Data augmentation simulating misalignment between left/right arm signals
- **Input**: Tensor of shape (T, 2, C)
- **Process**: 
  - Randomly jitters alignment between left and right EMG data
  - Configured with `max_offset=120` (60ms jitter at 2kHz sampling rate)
- **Output**: Tensor with same shape, but with left/right signals slightly misaligned

### 4. LogSpectrogram

```python
@dataclass
class LogSpectrogram:
    n_fft: int = 64
    hop_length: int = 16
```

- **Purpose**: Converts time-domain signals to frequency-domain spectrograms
- **Input**: Tensor of shape (T, 2, C)
- **Process**:
  - Applies Short-Time Fourier Transform (STFT) using specified parameters
  - Creates spectrograms with `n_fft=64` (33 frequency bins)
  - `hop_length=16` downsamples from 2kHz to 125Hz
  - Applies log10 scaling with small epsilon to avoid log(0)
  - Handles tensor dimension reordering
- **Output**: Log-spectrogram tensor of shape (T', 2, C, F) where:
  - T' = downsampled time steps
  - F = frequency bins (n_fft/2 + 1 = 33)

### 5. SpecAugment (Training only)

```python
@dataclass
class SpecAugment:
    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0
```

- **Purpose**: Advanced data augmentation for spectrograms
- **Input**: Log-spectrogram tensor of shape (T', 2, C, F)
- **Process**:
  - Applies up to 3 time masks (`n_time_masks=3`) of max length 25 timesteps (~200ms)
  - Applies up to 2 frequency masks (`n_freq_masks=2`) of max length 4 frequency bins
  - Masks are implemented by setting regions to 0 (`mask_value=0.0`)
- **Output**: Augmented log-spectrogram tensor with same shape

## Key Insights

1. **Downsampling**: The original 2kHz EMG signal is downsampled to 125Hz through the spectrogram transformation, reducing computational load while preserving important frequency information.

2. **Multiple Augmentations**: Training uses three different augmentation techniques:
   - Channel rotation to simulate electrode position variations
   - Temporal jittering to handle misalignment between arms
   - Spectrogram augmentation to improve model robustness

3. **Clean Validation Pipeline**: Validation and testing pipelines avoid augmentations, ensuring consistent evaluation.

4. **Signal Processing Details**:
   - FFT window size (64) creates 33 frequency bins
   - Disables centering of FFT windows to avoid padding inconsistencies
   - Uses log scaling to enhance low-amplitude components

This transformation pipeline effectively converts raw EMG signals into frequency-domain representations with appropriate augmentations for robust model training.

godlikeS97: I do not quite understand logSpectrogram and SpecAgment. Could you please further explain them with an example?

GitHub Copilot: # Understanding LogSpectrogram and SpecAugment

These two transforms are particularly important for processing EMG signals before feeding them into neural networks. Let me break them down with examples:

## LogSpectrogram Explained

The `LogSpectrogram` transform converts time-domain EMG signals into frequency-domain spectrograms and applies log scaling to enhance lower-amplitude features.

### How It Works:

1. **Input**: Raw EMG signal tensor of shape (T, 2, C) where:

   - T = time steps (e.g., 2000 samples for 1 second at 2kHz)
   - 2 = left and right arm data
   - C = electrode channels (typically 16 per arm)

2. **Dimension Reordering**:

   ```python
   x = tensor.movedim(0, -1)  # (T, 2, C) -> (2, C, T)
   ```

   This moves the time dimension to the end, as required by torchaudio's spectrogram function.

3. **Spectrogram Computation**:

   ```python
   spec = self.spectrogram(x)  # (2, C, freq, T')
   ```

   - The STFT creates a time-frequency representation
   - n_fft=64 produces 33 frequency bins (n_fft//2 + 1)
   - hop_length=16 means windows overlap by (64-16)/64 = 75%
   - T' becomes T/16 (e.g., 2000 samples → 125 time bins)

4. **Log Scaling**:

   ```python
   logspec = torch.log10(spec + 1e-6)
   ```

   - Applies log10 to compress dynamic range (small epsilon prevents log(0))
   - Makes lower amplitude components more visible/significant

5. **Output Reordering**:

   ```python
   return logspec.movedim(-1, 0)  # (T', 2, C, 33)
   ```

### Example:

If we have 2000 time steps of raw EMG data:

- Raw input: (2000, 2, 16) tensor
- After spectrogram: (2, 16, 33, 125) tensor
- After log scaling and reordering: (125, 2, 16, 33) tensor

The 125 time bins now represent the same 1 second of data, but in the frequency domain with 33 frequency bins per channel.

## SpecAugment Explained

`SpecAugment` performs data augmentation on spectrograms by masking random sections in both time and frequency domains.

### How It Works:

1. **Input**: Log-scaled spectrogram tensor of shape (T', 2, C, F)

2. **Dimension Reordering**:

   ```python
   x = specgram.movedim(0, -1)  # (T', 2, C, F) -> (2, C, F, T')
   ```

3. **Time Masking**:

   ```python
   n_t_masks = np.random.randint(self.n_time_masks + 1)
   for _ in range(n_t_masks):
       x = self.time_mask(x, mask_value=self.mask_value)
   ```

   - Randomly applies 0-3 time masks
   - Each mask covers up to 25 consecutive time steps (200ms at 125Hz)
   - Sets those regions to 0

4. **Frequency Masking**:

   ```python
   n_f_masks = np.random.randint(self.n_freq_masks + 1)
   for _ in range(n_f_masks):
       x = self.freq_mask(x, mask_value=self.mask_value)
   ```

   - Randomly applies 0-2 frequency masks
   - Each mask covers up to 4 consecutive frequency bins
   - Sets those regions to 0

5. **Output Reordering**:

   ```python
   return x.movedim(-1, 0)  # (2, C, F, T') -> (T', 2, C, F)
   ```

### Visual Example:

Let's visualize what this looks like for a single channel of EMG data:

```
Original Spectrogram (T' × F):
┌─────────────────────────────────────────────┐
│                                             │
│                                             │
│                                             │
│                                             │
└─────────────────────────────────────────────┘

After Time Masking:
┌─────────────────┬─────┬───────────┬────────┐
│                 │     │           │        │
│                 │  0  │           │   0    │
│                 │     │           │        │
│                 │     │           │        │
└─────────────────┴─────┴───────────┴────────┘

After Frequency Masking:
┌─────────────────┬─────┬───────────┬────────┐
│                 │     │           │        │
│                 │  0  │           │   0    │
├─────────────────┼─────┼───────────┼────────┤
│        0        │  0  │     0     │   0    │
└─────────────────┴─────┴───────────┴────────┘
```

## Why These Are Important

1. **LogSpectrogram**:
   - Converts raw signals to time-frequency representation
   - Reduces data dimensionality (2kHz → 125Hz) while preserving important information
   - Highlights muscle activation patterns in frequency space
   - Makes patterns more visible by compressing dynamic range with log scaling

2. **SpecAugment**:
   - Forces the model to learn robust features
   - Helps prevent overfitting by creating artificial variations
   - Simulates scenarios like temporary signal loss or interference
   - Improves generalization to new users or environments

Together, these transforms create an efficient, noise-robust representation of muscle activity patterns that neural networks can more effectively learn from.

Similar code found with 2 license types