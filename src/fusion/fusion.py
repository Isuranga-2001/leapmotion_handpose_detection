"""
Fusion strategies for combining LMC and RGB modalities.
Includes simple concatenation and more sophisticated fusion methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion.
    Concatenates LMC and RGB embeddings along the feature dimension.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        output_dim: Optional[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize concatenation fusion.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            output_dim: Output dimension (None = lmc_dim + rgb_dim)
            dropout: Dropout rate
        """
        super(ConcatenationFusion, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.output_dim = output_dim if output_dim is not None else (lmc_dim + rgb_dim)
        
        # Optional projection
        if self.output_dim != (lmc_dim + rgb_dim):
            self.projection = nn.Sequential(
                nn.Linear(lmc_dim + rgb_dim, self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.projection = None
    
    def forward(
        self,
        lmc_emb: torch.Tensor,
        rgb_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lmc_emb: LMC embeddings (batch_size, seq_len, lmc_dim)
            rgb_emb: RGB embeddings (batch_size, seq_len, rgb_dim)
        
        Returns:
            Fused embeddings (batch_size, seq_len, output_dim)
        """
        # Concatenate along feature dimension
        fused = torch.cat([lmc_emb, rgb_emb], dim=-1)
        
        # Optional projection
        if self.projection is not None:
            fused = self.projection(fused)
        
        return fused


class WeightedFusion(nn.Module):
    """
    Weighted fusion using learned weights.
    Allows the model to learn the importance of each modality.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize weighted fusion.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(WeightedFusion, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.output_dim = output_dim
        
        # Projection layers for each modality
        self.lmc_projection = nn.Linear(lmc_dim, output_dim)
        self.rgb_projection = nn.Linear(rgb_dim, output_dim)
        
        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Linear(lmc_dim + rgb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        lmc_emb: torch.Tensor,
        rgb_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lmc_emb: LMC embeddings (batch_size, seq_len, lmc_dim)
            rgb_emb: RGB embeddings (batch_size, seq_len, rgb_dim)
        
        Returns:
            Fused embeddings (batch_size, seq_len, output_dim)
        """
        # Project to same dimension
        lmc_proj = self.lmc_projection(lmc_emb)
        rgb_proj = self.rgb_projection(rgb_emb)
        
        # Compute weights
        concat_features = torch.cat([lmc_emb, rgb_emb], dim=-1)
        weights = self.weight_net(concat_features)  # (batch, seq_len, 2)
        
        # Apply weighted fusion
        lmc_weight = weights[..., 0:1]
        rgb_weight = weights[..., 1:2]
        
        fused = lmc_weight * lmc_proj + rgb_weight * rgb_proj
        fused = self.dropout(fused)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.
    Uses gating to control information flow from each modality.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize gated fusion.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(GatedFusion, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.output_dim = output_dim
        
        # Projection layers
        self.lmc_projection = nn.Linear(lmc_dim, output_dim)
        self.rgb_projection = nn.Linear(rgb_dim, output_dim)
        
        # Gate networks
        self.lmc_gate = nn.Sequential(
            nn.Linear(lmc_dim + rgb_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.rgb_gate = nn.Sequential(
            nn.Linear(lmc_dim + rgb_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        lmc_emb: torch.Tensor,
        rgb_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lmc_emb: LMC embeddings (batch_size, seq_len, lmc_dim)
            rgb_emb: RGB embeddings (batch_size, seq_len, rgb_dim)
        
        Returns:
            Fused embeddings (batch_size, seq_len, output_dim)
        """
        # Project to same dimension
        lmc_proj = self.lmc_projection(lmc_emb)
        rgb_proj = self.rgb_projection(rgb_emb)
        
        # Compute gates
        concat_features = torch.cat([lmc_emb, rgb_emb], dim=-1)
        lmc_gate_val = self.lmc_gate(concat_features)
        rgb_gate_val = self.rgb_gate(concat_features)
        
        # Apply gating
        fused = lmc_gate_val * lmc_proj + rgb_gate_val * rgb_proj
        fused = self.dropout(fused)
        
        return fused


class BilinearFusion(nn.Module):
    """
    Bilinear fusion for capturing interactions between modalities.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        output_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Initialize bilinear fusion.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            output_dim: Output dimension
            hidden_dim: Hidden dimension for bilinear layer
            dropout: Dropout rate
        """
        super(BilinearFusion, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Bilinear layer
        self.bilinear = nn.Bilinear(lmc_dim, rgb_dim, hidden_dim)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        lmc_emb: torch.Tensor,
        rgb_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lmc_emb: LMC embeddings (batch_size, seq_len, lmc_dim)
            rgb_emb: RGB embeddings (batch_size, seq_len, rgb_dim)
        
        Returns:
            Fused embeddings (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = lmc_emb.shape
        
        # Reshape for bilinear operation
        lmc_flat = lmc_emb.view(batch_size * seq_len, -1)
        rgb_flat = rgb_emb.view(batch_size * seq_len, -1)
        
        # Bilinear fusion
        fused_flat = self.bilinear(lmc_flat, rgb_flat)
        
        # Reshape and project
        fused = fused_flat.view(batch_size, seq_len, -1)
        fused = self.projection(fused)
        
        return fused


def create_fusion_module(
    fusion_type: str = 'concat',
    lmc_dim: int = 256,
    rgb_dim: int = 256,
    output_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion module.
    
    Args:
        fusion_type: Type of fusion ('concat', 'weighted', 'gated', or 'bilinear')
        lmc_dim: Dimension of LMC embeddings
        rgb_dim: Dimension of RGB embeddings
        output_dim: Output dimension
        **kwargs: Additional arguments for specific fusion module
    
    Returns:
        Fusion module
    """
    if fusion_type == 'concat':
        return ConcatenationFusion(lmc_dim, rgb_dim, output_dim, **kwargs)
    elif fusion_type == 'weighted':
        return WeightedFusion(lmc_dim, rgb_dim, output_dim, **kwargs)
    elif fusion_type == 'gated':
        return GatedFusion(lmc_dim, rgb_dim, output_dim, **kwargs)
    elif fusion_type == 'bilinear':
        return BilinearFusion(lmc_dim, rgb_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
