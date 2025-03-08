### ConformerCTCModule Structure:

1. Feature extraction (same as TDS):

- SpectrogramNorm: Normalize spectrograms

- MultiBandRotationInvariantMLP: Process each band with rotation invariance

- Flatten: Combine band features

1. Encoder: ConformerEncoder for sequence processing

1. Output: Linear projection to vocabulary size + LogSoftmax

1. Loss: CTC loss computation

✅ The structures match closely, with the main difference being the encoder architecture.

## Dimensional Analysis

Let's verify the dimensions at each step of the Conformer:

### Input dimensions:

- Raw input: (T, N, bands=2, electrode_channels=16, freq=33)

- T = sequence length in time

- N = batch size

- Bands = 2 (left/right arm)

- Electrode channels = 16 per band

- Frequency bins = 33

### Feature extraction:

1. SpectrogramNorm: preserves shape (T, N, 2, 16, 33)

1. MultiBandRotationInvariantMLP: transforms to (T, N, 2, 384) (where 384 is the mlp_features[-1])

1. Flatten: combines to (T, N, 768) (where 768 = 2 * 384)

### ConformerEncoder:

1. Input projection: (T, N, 768) → (T, N, d_model=256)

1. Positional encoding: preserves shape (T, N, 256)

1. ConformerBlocks: preserve shape (T, N, 256) through multiple blocks

### Output:

1. Linear projection: (T, N, 256) → (T, N, vocab_size)

1. LogSoftmax: preserves shape (T, N, vocab_size)