# TDSConvCTCModule Structure Brief Introduction 

`TDSConvCTCModule` is a PyTorch Lightning module that implements a Connectionist Temporal Classification (CTC) model for EMG-to-text decoding using Time-Depth Separable (TDS) convolutions. Here's a detailed breakdown of its components:

## Class Constants
- `NUM_BANDS: ClassVar[int] = 2`: The number of EMG bands (left and right armband)
- `ELECTRODE_CHANNELS: ClassVar[int] = 16`: The number of electrode channels per band

## Initialization Parameters
- `in_features`: Number of input features (528 based on config)
- `mlp_features`: List of feature dimensions for the MLP layers ([384] in config)
- `block_channels`: List defining the number of channels in each TDS conv block ([24, 24, 24, 24] in config)
- `kernel_width`: Kernel size for temporal convolutions (32 in config)
- `optimizer`: Configuration for the optimizer
- `lr_scheduler`: Configuration for the learning rate scheduler
- `decoder`: Configuration for the CTC decoder

## Model Architecture
The model processes EMG data with shape `(T, N, bands=2, electrode_channels=16, freq)`:

1. **SpectrogramNorm**: 
   - Normalizes spectrograms across each electrode channel
   - Uses batch normalization to compute stats over (N, freq, time) slices

2. **MultiBandRotationInvariantMLP**:
   - Applies a separate RotationInvariantMLP to each band
   - Creates rotation invariance by processing shifted versions of the input
   - Maps input features to a higher-level representation
   - Output shape: `(T, N, bands=2, mlp_features[-1])`

3. **Flatten**:
   - Flattens band and MLP features into a single dimension
   - Output shape: `(T, N, num_features)` where `num_features = NUM_BANDS * mlp_features[-1]`

4. **TDSConvEncoder**:
   - Series of TDSConv2dBlock and TDSFullyConnectedBlock modules
   - Temporal convolutions with depth-separable architecture
   - Processes sequential data while capturing temporal patterns
   - Maintains shape: `(T, N, num_features)`

5. **Linear + LogSoftmax**:
   - Final projection to character probabilities (num_classes = charset size)
   - LogSoftmax for numerical stability in CTC loss
   - Output shape: `(T, N, num_classes)`

## Loss and Metrics
- **CTC Loss**: Connects frame-level predictions to character sequences
- **Decoder**: Converts model predictions to character sequences during inference
- **CharacterErrorRates**: Measures model performance with character error rate metrics

## Training/Validation Logic
The module implements PyTorch Lightning methods:
- `forward`: Runs inputs through the model
- `_step`: Handles a single batch for training/validation/testing
  - Computes emissions, adjusts lengths for CTC, calculates loss
  - Decodes predictions and updates metrics
- `_epoch_end`: Computes and logs metrics at the end of each epoch
- `configure_optimizers`: Sets up optimizer and learning rate scheduler

The architecture effectively processes EMG signals through spectral normalization, rotation-invariant feature extraction, and temporal context modeling before performing character-level sequence prediction.



# Detailed Explanation of Each Class in modules.py

## SpectrogramNorm

This class normalizes EMG spectrograms across different electrode channels and frequency bands:

```python
class SpectrogramNorm(nn.Module):
```

- **Purpose**: Normalizes spectrograms from EMG data to improve training stability and model performance
- **Input shape**: `(T, N, num_bands, electrode_channels, frequency_bins)` where:
  - `T`: Sequence length (time)
  - `N`: Batch size
  - `num_bands`: Number of bands (usually 2 for left/right armbands)
  - `electrode_channels`: Number of electrodes (usually 16)
  - `frequency_bins`: Number of frequency components in the spectrogram
- **Implementation details**:
  - Reshapes input to apply 2D batch normalization across channels
  - Transforms shape to `(N, bands*C, freq, T)` for batch normalization
  - Computes statistics over `(N, freq, time)` dimensions for each electrode channel
  - Returns data in the original input shape

**Example with given parameters**:

```python
# Input: EMG spectrogram with shape (T=100, N=32, bands=2, electrodes=16, freq=33)
# where freq = n_fft//2 + 1 = 33 (for n_fft=64)
spectrogram_norm = SpectrogramNorm(channels=2*16)  # 32 total channels

# Forward pass
# inputs shape: (100, 32, 2, 16, 33)
normalized_output = spectrogram_norm(inputs)
# normalized_output has same shape: (100, 32, 2, 16, 33)
# but each channel is normalized independently
```

## 

## RotationInvariantMLP

```python
class RotationInvariantMLP(nn.Module):
```

- **Purpose**: Creates rotation invariance for EMG electrode readings by applying the same MLP to multiple rotated versions of the input
- **Key insight**: Electrode placement might not be consistent between sessions, so this creates robustness to rotational shifts
- **Parameters**:
  - `in_features`: Input dimension size
  - `mlp_features`: List of hidden layer sizes
  - `pooling`: Method to combine results from different rotations ("mean" or "max")
  - `offsets`: List of rotational shifts to apply (default: `-1, 0, 1`)
- **Processing flow**:
  1. Creates rotated versions of input by shifting electrode channels
  2. Flattens and processes each rotated input through the same MLP
  3. Pools results across rotations using mean or max pooling
- **Output shape**: `(T, N, mlp_features[-1])`

**Example with given parameters**:

```python
# For a single band's input: (T=100, N=32, C=16, freq=33)
# in_features = 16 * 33 = 528 (electrodes * freq bins)
ri_mlp = RotationInvariantMLP(
    in_features=528, 
    mlp_features=[384],  # As specified in config
    pooling="mean",
    offsets=(-1, 0, 1)  # Process original + left/right rotations
)

# Forward pass
# input shape: (100, 32, 16, 33)
output = ri_mlp(input)
# output shape: (100, 32, 384)
# Each time step now has 384 features that are invariant to electrode rotation
```

## 



## MultiBandRotationInvariantMLP

```python
class MultiBandRotationInvariantMLP(nn.Module):
```

- **Purpose**: Applies separate `RotationInvariantMLP` modules to each EMG band
- **Use case**: Processes left and right arm bands separately, then combines results
- **Parameters**:
  - Similar to `RotationInvariantMLP`, plus:
  - `num_bands`: Number of bands to process (default: 2)
  - `stack_dim`: Dimension where bands are stacked (default: 2)
- **Implementation**:
  - Creates a separate MLP instance for each band
  - Splits input along the band dimension
  - Processes each band independently
  - Stacks results back together
- **Output shape**: `(T, N, num_bands, mlp_features[-1])`

**Example with given parameters**:

```python
# For full input: (T=100, N=32, bands=2, C=16, freq=33)
mb_ri_mlp = MultiBandRotationInvariantMLP(
    in_features=528,        # 16 electrodes * 33 freq bins
    mlp_features=[384],     # As specified in config
    num_bands=2             # Left and right armbands
)

# Forward pass
# input shape: (100, 32, 2, 16, 33)
output = mb_ri_mlp(input)
# output shape: (100, 32, 2, 384)
# Each band processed separately but with same architecture
```

## 



## TDSConv2dBlock

```python
class TDSConv2dBlock(nn.Module):
```

- **Purpose**: Implements a Time-Depth Separable (TDS) convolution block for temporal processing
- **Based on**: "Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions" (Hannun et al.)
- **Parameters**:
  - `channels`: Number of input/output channels
  - `width`: Feature width (such that `channels * width = num_features`)
  - `kernel_width`: Temporal convolution kernel size
- **Architecture**:
  - 2D convolution along the temporal dimension
  - ReLU activation
  - Skip connection from input to output
  - Layer normalization
- **Shape transformation**: Preserves `(T, N, channels*width)` shape with potential temporal reduction

**Example with given parameters**:

```python
# With config parameters: block_channels = [24, 24, 24, 24]
# and num_features = 2*384 = 768 (from MultiBandRotationInvariantMLP output)
# width = num_features // channels = 768 // 24 = 32

tds_conv_block = TDSConv2dBlock(
    channels=24,
    width=32,            # 768 / 24 = 32
    kernel_width=32      # As specified in config
)

# Forward pass
# input shape: (100, 32, 768)
output = tds_conv_block(input)
# output shape: (T_out, 32, 768) where T_out ≤ 100 due to convolution
# T_out depends on kernel_width and padding
```

## 



## TDSFullyConnectedBlock

```python
class TDSFullyConnectedBlock(nn.Module):
```

- **Purpose**: Complements the TDSConv2dBlock with fully connected layers for feature mixing
- **Parameters**:
  - `num_features`: Feature dimension size
- **Architecture**:
  - Two linear layers with ReLU activation in between
  - Skip connection from input to output
  - Layer normalization
- **Key feature**: Preserves the temporal structure while mixing features

**Example with given parameters**:

```python
# Using num_features from previous example: 2*384 = 768
tds_fc_block = TDSFullyConnectedBlock(num_features=768)

# Forward pass
# input shape: (T_out, 32, 768)
output = tds_fc_block(input)
# output shape: (T_out, 32, 768)
# Same shape but with mixed features
```

## 



## TDSConvEncoder

```python
class TDSConvEncoder(nn.Module):
```

- **Purpose**: Combines TDSConv2dBlocks and TDSFullyConnectedBlocks into a complete encoder
- **Parameters**:
  - `num_features`: Feature dimension size
  - `block_channels`: List of channel counts for each TDS block (default: `[24, 24, 24, 24]`)
  - `kernel_width`: Convolution kernel size (default: 32)
- **Architecture**:
  - Alternates between TDSConv2dBlocks and TDSFullyConnectedBlocks
  - Creates a receptive field that grows with depth, capturing long-range dependencies
  - Maintains feature dimensionality throughout the network
- **Requirement**: Each channel count must evenly divide `num_features`

Each class works together in the broader EMG-to-text system to process electromyography signals, handle spatial invariances, and extract temporal patterns needed for character prediction.

**Example with given parameters**:

```python
# Using block_channels from config: [24, 24, 24, 24]
tds_encoder = TDSConvEncoder(
    num_features=768,              # 2*384 from MultiBandRotationInvariantMLP
    block_channels=[24, 24, 24, 24],
    kernel_width=32
)

# The encoder creates 4 pairs of blocks:
# 1. TDSConv2dBlock(24, 32, 32) + TDSFullyConnectedBlock(768) 
# 2. TDSConv2dBlock(24, 32, 32) + TDSFullyConnectedBlock(768)
# 3. TDSConv2dBlock(24, 32, 32) + TDSFullyConnectedBlock(768)
# 4. TDSConv2dBlock(24, 32, 32) + TDSFullyConnectedBlock(768)

# Forward pass
# input shape: (100, 32, 768)
output = tds_encoder(input)
# output shape: (~69, 32, 768) with temporal dimension reduced
# with kernel_width=32, each TDSConv2dBlock reduces the sequence length by 31 timesteps.
# because of the 4 convolutional layers (receptive field of 125)
```

## 

## Full Pipeline with TDSConvCTCModule

Given the config parameters, here's how data flows through the entire pipeline:

1. **Input**: EMG spectrograms with shape `(T, N, bands=2, electrodes=16, freq=33)` 

2. **SpectrogramNorm**: Normalizes each channel
   - Output: `(T, N, 2, 16, 33)` (same shape)

3. **MultiBandRotationInvariantMLP**: Processes each band with rotation invariance
   - Output: `(T, N, 2, 384)`

4. **Flatten**: Combines band features
   - Output: `(T, N, 768)`

5. **TDSConvEncoder**: Processes temporal sequences
   - Output: `(~T-125, N, 768)` ( temporal dimension reduced due to 4 conv blocks (each of kernel_width=32) )

6. **Linear + LogSoftmax**: Projects to character probabilities
   - Output: `(~T-125, N, num_classes)` (typically ~30-40 classes for text)

7. **CTC Loss/Decoder**: Connects frame predictions to character sequences
   - Final output: predicted text characters

The configuration creates a model with a receptive field of 125 samples, allowing it to capture temporal dependencies spanning approximately 62.5ms at 2kHz sampling rate, which is sufficient for character-level EMG patterns.