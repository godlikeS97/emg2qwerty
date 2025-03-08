# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from emg2qwerty.modules import PositionalEncoding


class SwishActivation(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class MultiHeadedSelfAttentionModule(nn.Module):
    """Multi-Headed Self-Attention Module for Conformer.
    
    Args:
        d_model (int): Input dimension
        nhead (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,  # Time-first format (T, N, ...)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for multi-headed self-attention.
        
        Args:
            x: Input tensor of shape (T, N, d_model)
            key_padding_mask: Optional mask of shape (N, T) where True values are positions to be masked
            
        Returns:
            Output tensor of shape (T, N, d_model)
        """
        x_norm = self.layer_norm(x)
        attn_output, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(attn_output)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer encoder.
    
    This implements the convolution module of the Conformer architecture with:
    1. LayerNorm
    2. Pointwise Conv (1x1)
    3. Gated Linear Unit (GLU) with Swish activation
    4. 1D Depthwise Conv
    5. Batch Normalization
    6. Swish activation
    7. Pointwise Conv (1x1)
    8. Dropout
    
    Args:
        d_model (int): Input dimension (256 for emg2qwerty)
        kernel_size (int): Kernel size for depthwise convolution (31 for emg2qwerty)
        expansion_factor (int): Expansion factor for pointwise convolution (2 for emg2qwerty)
        dropout (float): Dropout rate (0.15 for emg2qwerty)
    """
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Layer normalization - input/output: (T, N, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # First pointwise convolution (expands channels)
        # Input: (N, d_model, T) -> Output: (N, d_model*expansion_factor, T)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.activation = SwishActivation()
        
        # Depthwise convolution (processes each channel separately)
        # Input: (N, d_model*expansion_factor, T) -> Output: (N, d_model*expansion_factor, T)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model * expansion_factor,
            out_channels=d_model * expansion_factor,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,  # Same padding
            groups=d_model * expansion_factor,  # Depthwise convolution
            bias=True,
        )
        
        # Batch normalization - input/output: (N, d_model*expansion_factor, T)
        self.batch_norm = nn.BatchNorm1d(d_model * expansion_factor)
        
        # Second pointwise convolution (reduces channels back)
        # Input: (N, d_model*expansion_factor, T) -> Output: (N, d_model, T)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model * expansion_factor,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for convolution module.
        
        Args:
            x: Input tensor of shape (T, N, d_model)
            
        Returns:
            Output tensor of shape (T, N, d_model)
        """
        # Apply layer normalization - (T, N, d_model)
        x = self.layer_norm(x)
        
        # Transpose for 1D convolution - (T, N, d_model) -> (N, d_model, T)
        x = x.permute(1, 2, 0)
        
        # First pointwise convolution - (N, d_model, T) -> (N, d_model*expansion_factor, T)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        
        # Depthwise convolution - preserves shape (N, d_model*expansion_factor, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Second pointwise convolution - (N, d_model*expansion_factor, T) -> (N, d_model, T)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back - (N, d_model, T) -> (T, N, d_model)
        x = x.permute(2, 0, 1)
        
        return x


class FeedForwardModule(nn.Module):
    """Feed Forward module for Conformer encoder.
    
    Args:
        d_model (int): Input dimension
        d_ff (int): Hidden dimension of feed forward network
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = SwishActivation()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed forward module.
        
        Args:
            x: Input tensor of shape (T, N, d_model)
            
        Returns:
            Output tensor of shape (T, N, d_model)
        """
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConformerBlock(nn.Module):
    """Conformer block that combines MHSA, convolution, and feed-forward modules.
    
    The architecture follows the paper "Conformer: Convolution-augmented Transformer for Speech Recognition"
    with the following sequence:
    1. Half-step Feed-Forward Module
    2. Multi-Head Self-Attention Module
    3. Convolution Module
    4. Half-step Feed-Forward Module
    5. Layer Normalization
    
    Each module has residual connections internally.
    
    Args:
        d_model (int): Input dimension (256 for emg2qwerty)
        nhead (int): Number of attention heads (4 for emg2qwerty)
        d_ff (int): Hidden dimension of feed forward network (2048 for emg2qwerty)
        kernel_size (int): Kernel size for convolution module (31 for emg2qwerty)
        expansion_factor (int): Expansion factor for convolution module (2 for emg2qwerty)
        dropout (float): Dropout rate (0.15 for emg2qwerty)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int = 2048,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # First FFN module (half-step) - input/output: (T, N, d_model)
        self.feed_forward1 = FeedForwardModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        
        # MHSA module - input/output: (T, N, d_model)
        self.self_attention = MultiHeadedSelfAttentionModule(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        
        # Convolution module - input/output: (T, N, d_model)
        self.conv_module = ConvolutionModule(
            d_model=d_model,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        
        # Second FFN module (half-step) - input/output: (T, N, d_model)
        self.feed_forward2 = FeedForwardModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        
        # Final layer norm - input/output: (T, N, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for conformer block.
        
        Args:
            x: Input tensor of shape (T, N, d_model)
            key_padding_mask: Optional mask of shape (N, T) where True values are positions to be masked
            
        Returns:
            Output tensor of shape (T, N, d_model)
        """
        # Half-step FFN: (T, N, d_model) -> (T, N, d_model)
        x = x + 0.5 * self.feed_forward1(x)
        
        # MHSA with residual: (T, N, d_model) -> (T, N, d_model)
        x = x + self.self_attention(x, key_padding_mask)
        
        # Convolution with residual: (T, N, d_model) -> (T, N, d_model)
        x = x + self.conv_module(x)
        
        # Half-step FFN: (T, N, d_model) -> (T, N, d_model)
        x = x + 0.5 * self.feed_forward2(x)
        
        # Final layer norm: (T, N, d_model) -> (T, N, d_model)
        x = self.final_layer_norm(x)
        
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for EMG sequence modeling.
    
    Processes input features through a series of Conformer blocks.
    
    Args:
        num_features (int): Input feature dimension size (768 for emg2qwerty).
        d_model (int): Hidden dimension of the Conformer model (default: 256).
        nhead (int): Number of attention heads (default: 4).
        num_encoder_layers (int): Number of Conformer encoder layers (default: 6).
        d_ff (int): Dimension of the feedforward network (default: 2048).
        kernel_size (int): Kernel size for depthwise convolution (default: 31).
        expansion_factor (int): Expansion factor for convolution module (default: 2).
        dropout (float): Dropout rate (default: 0.1).
        max_seq_length (int): Maximum sequence length for positional encoding (default: 500).
    """
    def __init__(
        self, 
        num_features: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        d_ff: int = 2048,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        max_seq_length: int = 500
    ):
        super().__init__()
        
        # Project input features to model dimension
        # Input: (T, N, num_features=768) -> Output: (T, N, d_model=256)
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        # Input/Output: (T, N, d_model=256) -> (T, N, d_model=256)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # Conformer blocks - each preserves shape
        # Input/Output for each block: (T, N, d_model=256) -> (T, N, d_model=256)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                kernel_size=kernel_size,
                expansion_factor=expansion_factor,
                dropout=dropout,
            ) for _ in range(num_encoder_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through conformer encoder.
        
        Args:
            x: Input tensor of shape (T, N, num_features)
            src_key_padding_mask: Optional mask for padding positions of shape (N, T)
            
        Returns:
            Tensor of shape (T, N, d_model)
        """
        # Project to model dimension: (T, N, num_features) -> (T, N, d_model) 
        x = self.input_projection(x)
        
        # Add positional encoding: (T, N, d_model) -> (T, N, d_model)
        x = self.pos_encoder(x)
        
        # Apply Conformer blocks: (T, N, d_model) -> (T, N, d_model)
        for block in self.conformer_blocks:
            x = block(x, src_key_padding_mask)
        
        return x 