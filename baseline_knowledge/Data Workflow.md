# EMG2QWERTY Workflow: From Raw EMG to Model Training

The EMG2QWERTY project follows a structured pipeline from raw EMG sensor data to training samples. Here's a detailed explanation of the workflow from data to `WindowedEMGDataModule`:

## 1. EMGSessionData: Raw Data Interface

The starting point is `EMGSessionData` which provides an interface to the raw HDF5 files containing:
- EMG signals from left and right wrists (16 channels each at 2kHz)
- Timestamps for each EMG sample
- Metadata including keystrokes and prompts that serve as ground truth

```python
# Raw data access through EMGSessionData
session = EMGSessionData(hdf5_path)
emg_window = session[start_idx:end_idx]  # Get a slice of raw EMG data
```

## 2. LabelData: Ground Truth Processing

The `LabelData` class handles the textual labels and their timestamps:

```python
# Get ground truth for a specific time window
start_t = timestamps[offset]
end_t = timestamps[offset + window_length - 1]
label_data = session.ground_truth(start_t, end_t)
```

The ground truth is extracted in two different ways:
- For "on_keyboard" sessions: Uses `from_keystrokes()` with precise keystroke timing
- For other conditions: Uses `from_prompts()` with sentence-level prompts

`LabelData` handles:
- Cleaning and normalizing text using `CharacterSet`
- Converting between text, character labels, and integer indices
- Aligning timestamps with characters

## 3. WindowedEMGDataset: Creating Training Samples

`WindowedEMGDataset` converts raw session data into training samples by:

1. Segmenting the continuous EMG stream into windows
2. Adding contextual padding around each window
3. Applying transforms to the raw EMG data
4. Associating each window with its corresponding labels

```python
# Each item from the dataset contains:
emg_tensor, label_tensor = dataset[idx]
```

Key parameters:
- `window_length`: Size of each training window (8000 samples = 4 seconds at 2kHz)
- `stride`: Controls overlap between consecutive windows
- `padding`: Adds context (1800 samples past, 200 samples future) as specified in config
- `jitter`: Adds random offsets for data augmentation during training
- `transform`: Processing applied to raw EMG data (e.g., ToTensor, LogSpectrogram)

The dataset's `__getitem__` method:
1. Calculates the window offset based on index and stride
2. Adds jitter during training to increase variability
3. Applies contextual padding to the window
4. Extracts the corresponding EMG data and applies transforms
5. Gets the ground truth labels for the window's time span
6. Returns a tuple of (transformed_emg, labels)

## 4. WindowedEMGDataModule: Managing Datasets for Training

`WindowedEMGDataModule` integrates with PyTorch Lightning to manage datasets across training stages:

```python
# Creates separate datasets for train/val/test
def setup(self, stage):
    self.train_dataset = ConcatDataset([
        WindowedEMGDataset(
            hdf5_path, 
            transform=self.train_transform,
            window_length=self.window_length,
            padding=self.padding,
            jitter=True
        )
        for hdf5_path in self.train_sessions
    ])
    # Similar for val_dataset and test_dataset
```

Key responsibilities:
1. **Dataset Creation**:
   - Creates concatenated datasets from multiple sessions
   - Applies different transforms for train/val/test
   - Enables jittering only for training
   - Uses full sessions without windowing for testing

2. **DataLoader Configuration**:
   - Configures batch size, workers, and collation
   - Enables shuffling for training
   - Uses custom collation to handle variable-length sequences

3. **Batching Strategy**:
   - Training/validation: Uses configured batch size
   - Testing: Uses batch size of 1 for full sessions to avoid padding issues

The **collation function** `WindowedEMGDataset.collate` handles:
- Padding sequences to equal length
- Recording original sequence lengths for loss calculation
- Producing batches in time-first format (T, N, ...)

This **entire workflow** creates a comprehensive system that:
1. Efficiently handles raw EMG data
2. Provides accurate ground truth alignment
3. Creates windowed training samples with appropriate context
4. Applies different transforms for training vs. evaluation
5. Properly batches data for model training

The configuration in tds_conv_ctc.yaml sets the window length to 8000 samples (4 seconds) with padding of 1800 samples past context and 200 samples future context to ensure the model has sufficient information for accurate prediction.



# Question: What is the difference between Dataset and DataLoader ?

The EMG2QWERTY project follows PyTorch's data handling paradigm, which separates the concepts of Dataset and DataLoader. Here's a detailed explanation of their differences and how they work together in this project:

## Dataset (WindowedEMGDataset)

**Purpose:** Defines what data is available and how to access individual samples.

### Key Characteristics:
- **Data Source Definition**: Maps to the EMG session files in HDF5 format
- **Sample Access**: Implements `__getitem__(idx)` to retrieve individual EMG windows and corresponding labels
- **Data Transformation**: Applies transforms to raw EMG data (e.g., converting to tensors, spectrograms)
- **Window Management**: Handles windowing, padding, and jittering of raw EMG signals
- **Ground Truth Association**: Maps EMG windows to corresponding keystroke labels

### In EMG2QWERTY:
```python
# Creates a dataset that accesses one EMG session
dataset = WindowedEMGDataset(
    hdf5_path,
    transform=transform,
    window_length=8000,  # 4 seconds at 2kHz
    padding=(1800, 200),  # Past and future context
    jitter=True,  # Random offsets for training
)

# Retrieving a single sample
emg_tensor, label_tensor = dataset[42]
```

## DataLoader (torch.utils.data.DataLoader)

**Purpose:** Controls how samples are batched, shuffled, and loaded during training.

### Key Characteristics:
- **Batch Creation**: Combines multiple dataset items into training batches
- **Parallelism**: Manages parallel data loading across multiple workers
- **Shuffling**: Randomizes data access order for training
- **Memory Management**: Handles pinned memory, persistent workers, and other optimization settings
- **Collation**: Uses custom collation function to handle variable-length sequences

### In EMG2QWERTY:
```python
# Creates a dataloader that efficiently serves batches from the dataset
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,  # Randomize order for training
    num_workers=4,  # Parallel data loading
    collate_fn=WindowedEMGDataset.collate,  # Custom batch creation
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
)

# Iterating through batches during training
for batch in dataloader:
    inputs = batch["inputs"]     # (T, N, features)
    targets = batch["targets"]   # (T, N)
    # Training step...
```

