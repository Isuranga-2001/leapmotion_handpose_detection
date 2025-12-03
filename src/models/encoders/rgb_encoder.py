"""
RGB feature encoder for facial landmarks and pose data.
Encodes facial features into a dense embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RGBEncoderMLP(nn.Module):
    """
    MLP-based encoder for RGB facial features.
    Maps facial landmark and pose features to a dense embedding space.
    """
    
    def __init__(
        self,
        input_dim: int = 189,  # 62 landmarks × 3 + 3 pose angles
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize RGB MLP encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(RGBEncoderMLP, self).__init__()
        
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


class RGBEncoder1DCNN(nn.Module):
    """
    1D CNN-based encoder for RGB facial features.
    Processes temporal sequences of facial landmark data.
    """
    
    def __init__(
        self,
        input_dim: int = 189,
        num_filters: Tuple[int, ...] = (64, 128, 256),
        kernel_sizes: Tuple[int, ...] = (3, 3, 3),
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize RGB 1D CNN encoder.
        
        Args:
            input_dim: Input feature dimension
            num_filters: Tuple of filter counts for each conv layer
            kernel_sizes: Tuple of kernel sizes for each conv layer
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(RGBEncoder1DCNN, self).__init__()
        
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
        
        # Projection to output dimension
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
        
        # Transpose back: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x


class RGBEncoderLSTM(nn.Module):
    """
    LSTM-based encoder for RGB facial features.
    Captures temporal dynamics in facial expressions and head pose.
    """
    
    def __init__(
        self,
        input_dim: int = 189,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize RGB LSTM encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output embedding dimension
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(RGBEncoderLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
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
        # Project input
        x = self.input_projection(x)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Project output
        x = self.output_projection(x)
        
        return x


class RGBEncoderTransformer(nn.Module):
    """
    Transformer-based encoder for RGB facial features.
    Uses self-attention to model temporal relationships.
    """
    
    def __init__(
        self,
        input_dim: int = 189,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize RGB Transformer encoder.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(RGBEncoderTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Encoded tensor of shape (batch_size, seq_len, output_dim)
        """
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Project output
        x = self.output_projection(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


def create_rgb_encoder(
    encoder_type: str = 'mlp',
    input_dim: int = 189,
    output_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create RGB encoder.
    
    Args:
        encoder_type: Type of encoder ('mlp', 'cnn', 'lstm', or 'transformer')
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoder
    
    Returns:
        RGB encoder module
    """
    if encoder_type == 'mlp':
        return RGBEncoderMLP(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif encoder_type == 'cnn':
        return RGBEncoder1DCNN(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif encoder_type == 'lstm':
        return RGBEncoderLSTM(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif encoder_type == 'transformer':
        return RGBEncoderTransformer(input_dim=input_dim, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
