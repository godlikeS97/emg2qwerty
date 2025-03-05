# EMG2QWERTY Pipeline

The EMG2QWERTY pipeline converts electromyography (EMG) signals from muscle activity into keyboard inputs. Here's the complete data and model flow:

## Data Flow

1. **Data Collection**
   - EMG data is collected from wearable sensors with 32 channels (16 per arm)
   - Data is stored in HDF5 files in the data directory
   - Each file contains EMG readings and corresponding keystroke ground truth

2. **Preprocessing**
   - The raw EMG data goes through several transforms defined in transforms.py
   - Key transforms include:
     - `ToTensor`: Converts structured numpy arrays to PyTorch tensors
     - `LogSpectrogram`: Creates time-frequency representations
     - `RandomBandRotation`: Augments data with random rotations for robustness

3. **Dataset Creation**
   - data.py implements `WindowedEMGDataset` for batching and padding
   - Data is split into train/val/test sets via configurations in user
   - The `collate` function pads sequences to handle variable length inputs

## Model Architecture

1. **Feature Extraction**
   - The EMG signals are processed by a neural network defined in modules.py
   - TDS-Conv architecture (Temporal Depth Separable Convolution) is used
   - Components include:
     - `SpectrogramNorm` for normalization
     - `TDSConvEncoder` for feature extraction
     - `MultiBandRotationInvariantMLP` for further processing

2. **Sequence Modeling**
   - The model uses CTC (Connectionist Temporal Classification) for sequence prediction
   - The main model is implemented in `TDSConvCTCModule` in lightning.py

## Decoding

1. **CTC Decoding**
   - Raw model outputs are decoded using methods in decoder.py
   - Two decoding strategies are available:
     - `CTCGreedyDecoder`: Simple greedy approach (fastest)
     - `CTCBeamDecoder`: Beam search with language model integration (more accurate)

2. **Language Model Integration**
   - A 6-gram character-level language model is used to improve decoding
   - The LM is built using KenLM from WikiText-103 dataset
   - Built using build_char_lm.sh

## Training and Evaluation

1. **Training**
   - Training is performed using PyTorch Lightning
   - Run via train.py
   - Configured with YAML files in config
   - Command: `python -m emg2qwerty.train user="single_user" trainer.accelerator=gpu trainer.devices=1`

2. **Evaluation**
   - Models are evaluated on Character Error Rate (CER)
   - Results can be seen in experimental_results.py
   - CER results range from ~60% (generic model, no LM) to ~4-5% (personalized model with LM)

3. **Personalization**
   - The system supports both generic (user-agnostic) and personalized models
   - Personalization significantly improves performance

This pipeline allows for converting muscle activity to text, with the goal of enabling typing without physical keyboards.