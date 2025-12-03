"""
LMC feature encoder using MLP or 1D CNN architecture.
Encodes LMC hand joint features into a dense embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LMCEncoderMLP(nn.Module):
    """
    MLP-based encoder for LMC hand features.
    Maps LMC features to a dense embedding space.
    """
    
    def __init__(
        self,
        input_dim: int = 115,  # 81 raw joints + 34 geometric features
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize LMC MLP encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(LMCEncoderMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        # Handle sequential input
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size * seq_len, -1)
            x = self.encoder(x)
            x = x.view(batch_size, seq_len, -1)
        else:
            x = self.encoder(x)
        
        return x


class LMCEncoder1DCNN(nn.Module):
    """
    1D CNN-based encoder for LMC hand features.
    Processes temporal sequences of hand joint data.
    """
    
    def __init__(
        self,
        input_dim: int = 115,
        num_filters: Tuple[int, ...] = (64, 128, 256),
        kernel_sizes: Tuple[int, ...] = (3, 3, 3),
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize LMC 1D CNN encoder.
        
        Args:
            input_dim: Input feature dimension
            num_filters: Tuple of filter counts for each conv layer
            kernel_sizes: Tuple of kernel sizes for each conv layer
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(LMCEncoder1DCNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build 1D CNN layers
        conv_layers = []
        prev_channels = input_dim
        
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(prev_channels, num_filter, kernel_size, padding=kernel_size // 2)
            )
            conv_layers.append(nn.BatchNorm1d(num_filter))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(dropout))
            prev_channels = num_filter
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(prev_channels, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, seq_len, output_dim)
        """
        # Transpose for 1D convolution: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # For temporal encoding, we want per-frame embeddings
        # Transpose back: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x


class LMCEncoderHybrid(nn.Module):
    """
    Hybrid encoder combining CNN for spatial features and MLP for global features.
    """
    
    def __init__(
        self,
        input_dim: int = 115,
        cnn_filters: Tuple[int, ...] = (64, 128),
        mlp_hidden: Tuple[int, ...] = (256,),
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize hybrid LMC encoder.
        
        Args:
            input_dim: Input feature dimension
            cnn_filters: Filter counts for CNN branch
            mlp_hidden: Hidden dimensions for MLP branch
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(LMCEncoderHybrid, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # CNN branch for temporal patterns
        cnn_layers = []
        prev_channels = input_dim
        for num_filter in cnn_filters:
            cnn_layers.append(nn.Conv1d(prev_channels, num_filter, 3, padding=1))
            cnn_layers.append(nn.BatchNorm1d(num_filter))
            cnn_layers.append(nn.ReLU(inplace=True))
            prev_channels = num_filter
        
        self.cnn_branch = nn.Sequential(*cnn_layers)
        cnn_output_dim = prev_channels
        
        # MLP branch for global features
        mlp_layers = []
        prev_dim = input_dim
        for hidden_dim in mlp_hidden:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.mlp_branch = nn.Sequential(*mlp_layers)
        mlp_output_dim = prev_dim
        
        # Fusion and projection
        combined_dim = cnn_output_dim + mlp_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN branch
        x_cnn = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq_len, cnn_output_dim)
        
        # MLP branch (per-frame)
        x_mlp = x.view(batch_size * seq_len, -1)
        x_mlp = self.mlp_branch(x_mlp)
        x_mlp = x_mlp.view(batch_size, seq_len, -1)
        
        # Concatenate and fuse
        x_combined = torch.cat([x_cnn, x_mlp], dim=-1)
        x_out = self.fusion(x_combined)
        
        return x_out


def create_lmc_encoder(
    encoder_type: str = 'mlp',
    input_dim: int = 115,
    output_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create LMC encoder.
    
    Args:
        encoder_type: Type of encoder ('mlp', 'cnn', or 'hybrid')
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoder
    
    Returns:
        LMC encoder module
    """
    if encoder_type == 'mlp':
        return LMCEncoderMLP(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif encoder_type == 'cnn':
        return LMCEncoder1DCNN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif encoder_type == 'hybrid':
        return LMCEncoderHybrid(input_dim=input_dim, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
