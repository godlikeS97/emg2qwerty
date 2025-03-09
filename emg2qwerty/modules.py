# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N=batch_, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)  (T, N, bands=2, C=16, freq)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model.
    Based on the original implementation in the 'Attention Is All You Need' paper.
    
    Args:
        d_model (int): The dimension of the transformer model.
        dropout (float): Dropout rate for the positional encoding.
        max_len (int): Maximum sequence length for positional encoding.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        import math
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Save these for generating extra positions if needed
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        
        # If sequence length is longer than our precomputed positional encodings
        if seq_len > self.pe.size(0):
            # Generate additional positional encodings
            import math
            additional_len = seq_len - self.pe.size(0)
            # Make sure position is on the same device as x
            position = torch.arange(self.pe.size(0), seq_len, device=x.device).unsqueeze(1)
            additional_pe = torch.zeros(additional_len, 1, self.d_model, device=x.device)
            additional_pe[:, 0, 0::2] = torch.sin(position * self.div_term)
            additional_pe[:, 0, 1::2] = torch.cos(position * self.div_term)
            
            # Concatenate with existing pe
            pe_extended = torch.cat([self.pe.to(x.device), additional_pe], dim=0)
            
            # Use the extended positional encoding
            x = x + pe_extended[:seq_len]
        else:
            # Standard case - use precomputed positional encoding
            x = x + self.pe[:seq_len].to(x.device)
            
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for EMG sequence modeling.
    Combines custom components with PyTorch's TransformerEncoder.
    
    Args:
        num_features (int): Input feature dimension size.
        d_model (int): Hidden dimension of the transformer model.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of transformer encoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        activation (str): Activation function to use (relu or gelu).
        max_seq_length (int): Maximum sequence length for positional encoding.
        norm_first (bool): Whether to use pre-normalization (default: True).
        layer_norm_eps (float): Epsilon for layer normalization (default: 1e-5).
    """
    def __init__(
        self, 
        num_features: int,
        d_model: int = 384,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 500,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        # Project input features to transformer dimension
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # PyTorch expects (seq_len, batch, features)
            norm_first=norm_first,  # Pre-norm architecture for better stability
            layer_norm_eps=layer_norm_eps  # Customize LayerNorm epsilon
        )
        
        # Full transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through transformer encoder.
        
        Args:
            x: Input tensor of shape (T, N, features)
            src_key_padding_mask: Optional mask for padded positions (N, T)
            
        Returns:
            Tensor of shape (T, N, d_model)
        """
        # Project to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder with optional mask
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        return x


class LSTMEncoder(nn.Module):
    """LSTM encoder for EMG sequence modeling.
    
    Args:
        input_size (int): Input feature dimension size.
        hidden_size (int): Hidden dimension of the LSTM model.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
        bidirectional (bool): Whether to use bidirectional LSTM.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        
        # LSTM takes care of its own initialization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # Match TDS-Conv behavior: (T, N, features)
        )
        
        # Projection needed if using bidirectional LSTM
        self.output_size = hidden_size * (2 if bidirectional else 1)
        if self.output_size != input_size:
            self.projection = nn.Linear(self.output_size, input_size)
        else:
            self.projection = nn.Identity()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM encoder.
        
        Args:
            inputs: Tensor of shape (T, N, input_size)
            
        Returns:
            Tensor of shape (T, N, input_size)
        """
        # LSTM forward pass
        outputs, _ = self.lstm(inputs)
        
        # Project back to input size if necessary
        if self.output_size != inputs.shape[2]:
            outputs = self.projection(outputs)
        
        return outputs

class CNNLSTMEncoder(nn.Module):
    """A hybrid CNN + LSTM encoder that first uses convolutional layers to
    extract local features, then uses LSTM layers to model temporal dependencies.
    
    Args:
        num_features (int): Number of input features (C).
        cnn_channels (list): List of output channels for each CNN block.
        kernel_size (int): Kernel size for each convolution.
        lstm_hidden_size (int): Hidden size for the LSTM layers.
        lstm_num_layers (int): Number of LSTM layers.
        dropout (float): Dropout probability for LSTM layers.
    """
    
    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 5,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        # CNN layers to extract local features
        cnn_layers: list[nn.Module] = []
        in_channels = num_features  # C

        # 注意：如果 T 不是很大，kernel_size=5 + 多次池化可能导致输出维度过小甚至为 0
        # 可尝试改小 kernel_size=3，或者减少池化层数
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                # 这里每加一次 MaxPool(stride=2)，时间维度就会减半
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,  # Expect (time, batch, channels)
        )
        
        # Final projection to maintain consistent output size
        # 双向 LSTM 输出维度 = 2 * lstm_hidden_size
        self.projection = nn.Linear(lstm_hidden_size * 2, num_features)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (T, N, C)  # time, batch, features
        Returns:
            outputs: (T, N, C)  # 与输入时间维度一致 (如果使用了上采样)
        """
        T, N, C = inputs.shape
        
        # 1) 重新排列为 (N, C, T)，让 T 成为 CNN 的卷积/池化方向
        x = inputs.permute(1, 2, 0)  # (N, C, T)
        
        # 2) 经过 CNN 模块后，形状 (N, out_channels, T')
        x = self.cnn(x)  # 多层卷积+池化后，时间维度 T -> T'
        
        # 3) 变回 (T', N, out_channels)，以便喂入 LSTM
        x = x.permute(2, 0, 1)  # (T', N, out_channels)
        
        # 4) LSTM 前向
        x, _ = self.lstm(x)  # (T', N, 2*lstm_hidden_size)
        
        # 5) 投影回 num_features
        x = self.projection(x)  # (T', N, num_features)
        
        # ============== 可选：上采样回原始 T ==============
        # 如果希望输出与输入时间序列等长 (T)，可以插值上采样
        # 如果不需要等长，去掉这一步，直接 return x
        x = x.permute(1, 2, 0)  # (N, num_features, T')
        x = F.interpolate(x, size=T, mode='linear', align_corners=False)  # (N, num_features, T)
        x = x.permute(2, 0, 1)  # (T, N, num_features)
        # =============================================
        
        return x