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
        dropout (float): Dropout rate for general dropout
        attention_dropout (float): Dropout rate specific to attention weights
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        attention_dropout: float = None,
    ):
        super().__init__()
        # If attention_dropout is not specified, use the general dropout rate
        if attention_dropout is None:
            attention_dropout = dropout
            
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attention_dropout,  # Use attention-specific dropout
            batch_first=False,  # Time-first format (T, N, ...)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None,
        enforce_mask: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (T, N, d_model)
            key_padding_mask: Boolean mask for padding tokens (N, T)
                where True indicates positions to mask out
            enforce_mask: If True, creates a causal mask to prevent
                attention to future timesteps
        """
        x_norm = self.layer_norm(x)
        
        # Handle masking based on sequence lengths
        T = x.size(0)
        
        # Create causal mask if enforcing masking
        attn_mask = None
        if enforce_mask:
            # For very long sequences, avoid creating a huge T×T mask
            if T > 1000:  # Threshold for switching to a more memory-efficient approach
                # Use a custom_function parameter for PyTorch's built-in masking
                # This is more memory efficient than creating the full mask matrix
                out, _ = self.self_attn(
                    x_norm, x_norm, x_norm,
                    key_padding_mask=key_padding_mask,
                    attn_mask=None,
                    need_weights=False,
                    is_causal=True  # Built-in causal masking (more efficient)
                )
                return x + self.dropout(out)
            else:
                # For shorter sequences, use the explicit mask as before
                attn_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
        
        out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )
        return x + self.dropout(out)


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
    """Conformer block combining self-attention, convolution, and feed-forward modules.
    
    Args:
        d_model (int): Input dimension
        nhead (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        kernel_size (int): Kernel size for convolution module
        expansion_factor (int): Expansion factor for convolution module
        dropout (float): General dropout rate
        attention_dropout (float): Dropout specific to attention mechanism
        conv_dropout (float): Dropout specific to convolution module
        norm_first (bool): Whether to apply layer norm before each sub-block
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int = 2048,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = None,
        conv_dropout: float = None,
        norm_first: bool = False,
    ):
        super().__init__()
        
        # Use provided specific dropout rates or fall back to general dropout
        if attention_dropout is None:
            attention_dropout = dropout
        if conv_dropout is None:
            conv_dropout = dropout
            
        # Half-step feed-forward modules
        self.ff1 = FeedForwardModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Multi-headed self-attention module
        self.self_attention = MultiHeadedSelfAttentionModule(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
        # Convolution module
        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            dropout=conv_dropout,
        )
        
        # Second half-step feed-forward module
        self.ff2 = FeedForwardModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        
    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None,
        enforce_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (T, N, d_model)
            key_padding_mask: Boolean mask of shape (N, T) where True 
                indicates positions to mask out
            enforce_mask: Whether to enforce causal attention masking
                
        Returns:
            Output tensor of shape (T, N, d_model)
        """
        # Apply 1st feed-forward module (with residual)
        if self.norm_first:
            ff1_out = x + 0.5 * self.ff1(self.final_layer_norm(x))
        else:
            ff1_out = x + 0.5 * self.ff1(x)
        
        # Apply self-attention (with residual)
        if self.norm_first:
            attn_out = ff1_out + self.self_attention(
                self.final_layer_norm(ff1_out), 
                key_padding_mask, 
                enforce_mask
            )
        else:
            attn_out = self.self_attention(
                ff1_out, 
                key_padding_mask, 
                enforce_mask
            )
        
        # Apply convolution module (with residual)
        if self.norm_first:
            conv_out = attn_out + self.conv(self.final_layer_norm(attn_out))
        else:
            conv_out = attn_out + self.conv(attn_out)
        
        # Apply 2nd feed-forward module (with residual)
        if self.norm_first:
            output = conv_out + 0.5 * self.ff2(self.final_layer_norm(conv_out))
        else:
            output = conv_out + 0.5 * self.ff2(conv_out)
            
        # Apply final layer norm if not using pre-norm architecture
        if not self.norm_first:
            output = self.final_layer_norm(output)
            
        return output


class ConformerEncoder(nn.Module):
    """Conformer encoder consisting of multiple conformer blocks.
    
    Args:
        num_features (int): Number of input features
        d_model (int): Model dimension (256 default)
        nhead (int): Number of attention heads
        num_encoder_layers (int): Number of conformer blocks
        d_ff (int): Feed-forward dimension
        kernel_size (int): Kernel size for convolution modules
        expansion_factor (int): Expansion factor for convolution modules
        dropout (float): General dropout rate
        attention_dropout (float): Dropout rate for attention weights
        conv_dropout (float): Dropout rate for convolution modules
        max_seq_length (int): Maximum sequence length for positional encoding
        norm_first (bool): Whether to use pre-normalization
        layer_dropout (float): Probability of dropping entire layers during training
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
        attention_dropout: float = None,
        conv_dropout: float = None,
        max_seq_length: int = 500,
        norm_first: bool = False,
        layer_dropout: float = 0.0,
    ):
        super().__init__()
        
        # Default specific dropouts to general dropout if not specified
        if attention_dropout is None:
            attention_dropout = dropout
        if conv_dropout is None:
            conv_dropout = dropout
            
        # Linear projection to d_model dimension
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_seq_length,
        )
        
        # Create a stack of conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                kernel_size=kernel_size,
                expansion_factor=expansion_factor,
                dropout=dropout,
                attention_dropout=attention_dropout,
                conv_dropout=conv_dropout,
                norm_first=norm_first,
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Layer dropout probability
        self.layer_dropout = layer_dropout
        self.num_layers = num_encoder_layers
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor] = None,
        enforce_mask: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (T, N, num_features)
            src_key_padding_mask: Boolean mask of shape (N, T) where True
                indicates positions to mask out
            enforce_mask: Whether to enforce causal attention masking
                
        Returns:
            Output tensor of shape (T, N, d_model)
        """
        # Project to d_model dimension: (T, N, num_features) -> (T, N, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding: (T, N, d_model) -> (T, N, d_model)
        x = self.pos_encoder(x)
        
        # For very long sequences in test mode, use a memory-efficient approach
        is_test = not torch.is_grad_enabled()
        is_long_seq = x.size(0) > 1000
        
        if is_test and is_long_seq:
            # For long test sequences, process in chunks to save memory
            chunk_size = 1000
            T, N, D = x.size()
            outputs = []
            
            for i in range(0, T, chunk_size):
                end_idx = min(i + chunk_size, T)
                x_chunk = x[i:end_idx]
                
                # Create chunk-specific padding mask
                chunk_mask = None
                if src_key_padding_mask is not None:
                    chunk_mask = src_key_padding_mask[:, i:end_idx]
                
                # Process chunk through layers
                for j, layer in enumerate(self.layers):
                    # Skip some layers randomly during training if using layer dropout
                    if self.training and self.layer_dropout > 0:
                        dropout_prob = (j / (self.num_layers - 1)) * self.layer_dropout
                        if torch.rand(1).item() < dropout_prob:
                            continue
                    
                    # Apply layer with chunk-specific mask
                    x_chunk = layer(x_chunk, chunk_mask, enforce_mask)
                
                outputs.append(x_chunk)
                
            # Concatenate chunks back together
            x = torch.cat(outputs, dim=0)
            
        else:
            # Normal processing for shorter sequences
            for i, layer in enumerate(self.layers):
                # Apply stochastic layer dropout during training
                if self.training and self.layer_dropout > 0:
                    dropout_prob = (i / (self.num_layers - 1)) * self.layer_dropout
                    if torch.rand(1).item() < dropout_prob:
                        continue
                
                # Apply standard conformer layer
                x = layer(x, src_key_padding_mask, enforce_mask)
            
        # Final norm
        return self.norm(x) 