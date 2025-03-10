# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T', 2, C, freq) -> (2, C, freq, T')
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)  # (T', 2, C, freq)


@dataclass
class TemporalScaling:
    """Applies random temporal scaling (time warping) to EMG signals.
    
    This transform simulates natural variations in typing speed and rhythm
    by stretching or compressing the time dimension of the EMG signal.
    The input must be of shape (T, ...) where T is the time dimension.
    
    Args:
        min_scale (float): Minimum scaling factor (values < 1.0 slow down the signal)
        max_scale (float): Maximum scaling factor (values > 1.0 speed up the signal)
        time_dim (int): The time dimension to scale (default: 0)
    """
    
    min_scale: float = 0.9  # Slow down to 90%
    max_scale: float = 1.1  # Speed up to 110%
    time_dim: int = 0
    
    def __post_init__(self) -> None:
        assert 0.0 < self.min_scale <= self.max_scale, "Scaling factors must be positive with min_scale <= max_scale"
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Choose random scaling factor
        scale_factor = torch.FloatTensor(1).uniform_(self.min_scale, self.max_scale).item()
        
        # Original sequence length in the time dimension
        seq_len = tensor.shape[self.time_dim]
        
        # Target length after scaling
        new_len = int(seq_len * scale_factor)
        
        # Ensure we have at least 1 timestep
        new_len = max(1, new_len)
        
        # Get original shape for later reshaping
        original_shape = tensor.shape
        
        # Handle dimensionality: we need to reshape to a 3D tensor for interpolation
        # where the last dimension is time (required by F.interpolate with mode='linear')
        
        # First move time to the last dimension
        tensor_t_last = tensor.movedim(self.time_dim, -1)
        
        # Reshape to (C, H, T) format needed for linear interpolation
        # Flatten all dimensions except the last (time) into a single batch dim
        reshaped_tensor = tensor_t_last.reshape(-1, 1, tensor_t_last.shape[-1])
        
        # Apply interpolation (now with proper 3D input)
        scaled_tensor = torch.nn.functional.interpolate(
            reshaped_tensor,
            size=new_len,
            mode='linear',
            align_corners=False
        )
        
        # Reshape back to original dimensions but with new time length
        new_shape = list(tensor_t_last.shape)
        new_shape[-1] = scaled_tensor.shape[-1]
        scaled_tensor = scaled_tensor.reshape(new_shape)
        
        # Move time dimension back to original position
        scaled_tensor = scaled_tensor.movedim(-1, self.time_dim)
        
        # Crop or pad to match original length
        if new_len > seq_len:
            # Crop to original length
            slices = [slice(None)] * tensor.ndim
            slices[self.time_dim] = slice(0, seq_len)
            return scaled_tensor[tuple(slices)]
        elif new_len < seq_len:
            # Pad to original length
            padding = list(original_shape)
            padding[self.time_dim] = seq_len - new_len
            padding_tensor = torch.zeros(padding, dtype=tensor.dtype, device=tensor.device)
            
            # Concatenate along time dimension
            return torch.cat([scaled_tensor, padding_tensor], dim=self.time_dim)
        else:
            return scaled_tensor


@dataclass
class BandpassFilter:
    """Applies bandpass filtering to EMG signals to reduce noise and focus on
    the most informative frequency range for muscle activity.
    
    The filter is implemented in the frequency domain using FFT.
    The input must be of shape (T, ...) where T is the time dimension.
    
    Args:
        low_cut (float): Lower cutoff frequency in Hz. Frequencies below this will be attenuated.
        high_cut (float): Upper cutoff frequency in Hz. Frequencies above this will be attenuated.
        sample_rate (float): Sampling rate of the EMG signal in Hz.
        time_dim (int): The time dimension to filter (default: 0)
        transition_width (float): Width of the transition band as a fraction of the cutoff frequency.
    """
    
    low_cut: float = 20.0  # Lower cutoff frequency in Hz
    high_cut: float = 450.0  # Upper cutoff frequency in Hz
    sample_rate: float = 2000.0  # EMG sample rate (Hz)
    time_dim: int = 0
    transition_width: float = 0.1  # Transition width as fraction of cutoff
    
    def __post_init__(self) -> None:
        assert 0 < self.low_cut < self.high_cut < self.sample_rate / 2, \
            f"Invalid frequency range: low_cut={self.low_cut}, high_cut={self.high_cut}, " \
            f"sample_rate/2={self.sample_rate/2}"
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Move time dimension to last position for FFT
        x = tensor.movedim(self.time_dim, -1)
        
        # Get original shape and time length
        orig_shape = x.shape
        time_length = orig_shape[-1]
        
        # Reshape to 2D for batch processing: (batch, time)
        x_flat = x.reshape(-1, time_length)
        x_filtered = self._apply_bandpass(x_flat)
        
        # Reshape back to original dimensions
        x_filtered = x_filtered.reshape(orig_shape)
        
        # Move time dimension back to original position
        return x_filtered.movedim(-1, self.time_dim)
    
    def _apply_bandpass(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter to batched 2D input of shape (batch, time)"""
        # Get FFT
        X = torch.fft.rfft(x, dim=-1)
        
        # Construct frequency array
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/self.sample_rate)
        
        # Create bandpass filter mask with smooth transitions
        low_mask = self._transition_band(
            freqs, self.low_cut * (1 - self.transition_width), self.low_cut
        )
        high_mask = 1 - self._transition_band(
            freqs, self.high_cut, self.high_cut * (1 + self.transition_width)
        )
        mask = low_mask * high_mask
        
        # Apply filter
        X_filtered = X * mask.to(X.device)
        
        # Inverse FFT to get back to time domain
        x_filtered = torch.fft.irfft(X_filtered, n=x.shape[-1], dim=-1)
        
        return x_filtered
    
    def _transition_band(self, freqs: torch.Tensor, f0: float, f1: float) -> torch.Tensor:
        """Create a smooth transition band between f0 and f1 using cosine transition"""
        mask = torch.ones_like(freqs, dtype=torch.float32)
        
        # Transition band indices
        idx = (freqs >= f0) & (freqs <= f1)
        
        if idx.any():
            # Apply cosine transition (Tukey window segment)
            mask[idx] = 0.5 * (1 + torch.cos(
                torch.pi * (freqs[idx] - f0) / (f1 - f0) + torch.pi
            ))
        
        # Set mask to 0 for all frequencies below f0
        mask[freqs < f0] = 0.0
        
        return mask

@dataclass
class CrossChannelMixing:
    """Creates synthetic EMG channels by mixing existing channels.
    
    This transform helps simulate electrode cross-talk and variations in
    electrode placement, improving model robustness to these factors.
    
    The input must be of shape (T, B, C) where:
      - T is time dimension
      - B is band dimension (left/right hand)
      - C is channel dimension (number of electrodes)
    
    Args:
        mix_ratio_range (tuple[float, float]): Range of mixing ratios to sample from.
            Values closer to (0.5, 0.5) create more balanced mixes, while values
            closer to (0.0, 1.0) keep more of the original signal.
        num_channels_to_mix (int): Number of channels to apply mixing to.
            If set to -1, applies to all channels.
        seed (int): Random seed for reproducibility. None for random behavior.
    """
    
    mix_ratio_range: tuple[float, float] = (0.1, 0.3)  
    num_channels_to_mix: int = 8  # Mix half of the 16 channels by default
    seed: int | None = None
    
    def __post_init__(self) -> None:
        assert 0 <= self.mix_ratio_range[0] <= self.mix_ratio_range[1] <= 0.5, \
            "Mix ratio range must be between 0 and 0.5 with min <= max"
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expect (Time, Band, Channel) format
        assert tensor.ndim >= 3, "Expected at least 3D tensor (T, B, C)"
        
        # Make a copy to avoid modifying the original
        result = tensor.clone()
        
        # Get shape info
        time_dim, band_dim, channel_dim = 0, 1, 2
        num_channels = tensor.shape[channel_dim]
        
        # Determine how many channels to mix
        channels_to_mix = min(self.num_channels_to_mix, num_channels) if self.num_channels_to_mix > 0 else num_channels
        
        for band_idx in range(tensor.shape[band_dim]):
            # Select channels to mix (randomly)
            selected_channels = torch.randperm(num_channels)[:channels_to_mix]
            
            # Mix each selected channel with another random channel
            for ch_idx in selected_channels:
                # Pick another channel to mix with (different from current)
                other_channels = [i for i in range(num_channels) if i != ch_idx]
                mix_with = other_channels[torch.randint(len(other_channels), (1,)).item()]
                
                # Sample mixing ratio from specified range
                alpha = torch.empty(1).uniform_(self.mix_ratio_range[0], self.mix_ratio_range[1]).item()
                
                # Create the mix: (1-alpha)*original + alpha*other
                original_signal = tensor[:, band_idx, ch_idx]
                other_signal = tensor[:, band_idx, mix_with]
                
                # Apply the mix
                result[:, band_idx, ch_idx] = (1 - alpha) * original_signal + alpha * other_signal
        
        return result
