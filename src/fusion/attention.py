"""
Cross-modal attention mechanisms for LMC and RGB fusion.
Enables modalities to attend to each other for better feature integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between LMC and RGB features.
    Allows each modality to attend to the other.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossModalAttention, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.num_heads = num_heads
        self.head_dim = lmc_dim // num_heads
        
        assert lmc_dim == rgb_dim, "LMC and RGB dimensions must match for cross-attention"
        assert lmc_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        
        # LMC attends to RGB
        self.lmc_to_rgb_q = nn.Linear(lmc_dim, lmc_dim)
        self.lmc_to_rgb_k = nn.Linear(rgb_dim, lmc_dim)
        self.lmc_to_rgb_v = nn.Linear(rgb_dim, lmc_dim)
        
        # RGB attends to LMC
        self.rgb_to_lmc_q = nn.Linear(rgb_dim, rgb_dim)
        self.rgb_to_lmc_k = nn.Linear(lmc_dim, rgb_dim)
        self.rgb_to_lmc_v = nn.Linear(lmc_dim, rgb_dim)
        
        # Output projections
        self.lmc_out = nn.Linear(lmc_dim, lmc_dim)
        self.rgb_out = nn.Linear(rgb_dim, rgb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch, num_heads, seq_len, head_dim)
            key: Key tensor (batch, num_heads, seq_len, head_dim)
            value: Value tensor (batch, num_heads, seq_len, head_dim)
            mask: Optional attention mask
        
        Returns:
            Attention output (batch, num_heads, seq_len, head_dim)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output
    
    def forward(
        self,
        lmc_emb: torch.Tensor,
        rgb_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            lmc_emb: LMC embeddings (batch_size, seq_len, lmc_dim)
            rgb_emb: RGB embeddings (batch_size, seq_len, rgb_dim)
        
        Returns:
            Tuple of (attended_lmc, attended_rgb)
        """
        batch_size, seq_len, _ = lmc_emb.shape
        
        # LMC attends to RGB
        lmc_q = self.lmc_to_rgb_q(lmc_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        rgb_k = self.lmc_to_rgb_k(rgb_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        rgb_v = self.lmc_to_rgb_v(rgb_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        lmc_attended = self._attention(lmc_q, rgb_k, rgb_v)
        lmc_attended = lmc_attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.lmc_dim)
        lmc_attended = self.lmc_out(lmc_attended)
        
        # RGB attends to LMC
        rgb_q = self.rgb_to_lmc_q(rgb_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        lmc_k = self.rgb_to_lmc_k(lmc_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        lmc_v = self.rgb_to_lmc_v(lmc_emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        rgb_attended = self._attention(rgb_q, lmc_k, lmc_v)
        rgb_attended = rgb_attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.rgb_dim)
        rgb_attended = self.rgb_out(rgb_attended)
        
        return lmc_attended, rgb_attended


class CrossModalAttentionFusion(nn.Module):
    """
    Complete cross-modal attention fusion module.
    Applies cross-attention and fuses the results.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        num_heads: int = 8,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention fusion.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            num_heads: Number of attention heads
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.output_dim = output_dim
        
        # Ensure equal dimensions for cross-attention
        if lmc_dim != rgb_dim:
            self.lmc_projection = nn.Linear(lmc_dim, rgb_dim)
            self.use_projection = True
            attn_dim = rgb_dim
        else:
            self.use_projection = False
            attn_dim = lmc_dim
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            lmc_dim=attn_dim,
            rgb_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.lmc_norm = nn.LayerNorm(attn_dim)
        self.rgb_norm = nn.LayerNorm(attn_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(attn_dim * 2, output_dim),
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
        # Project LMC if needed
        if self.use_projection:
            lmc_emb = self.lmc_projection(lmc_emb)
        
        # Apply cross-modal attention
        lmc_attended, rgb_attended = self.cross_attention(lmc_emb, rgb_emb)
        
        # Residual connection and normalization
        lmc_attended = self.lmc_norm(lmc_emb + lmc_attended)
        rgb_attended = self.rgb_norm(rgb_emb + rgb_attended)
        
        # Concatenate and fuse
        fused = torch.cat([lmc_attended, rgb_attended], dim=-1)
        fused = self.fusion(fused)
        
        return fused


class CoAttention(nn.Module):
    """
    Co-attention mechanism for simultaneous bidirectional attention.
    """
    
    def __init__(
        self,
        lmc_dim: int = 256,
        rgb_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize co-attention.
        
        Args:
            lmc_dim: Dimension of LMC embeddings
            rgb_dim: Dimension of RGB embeddings
            hidden_dim: Hidden dimension for attention computation
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(CoAttention, self).__init__()
        
        self.lmc_dim = lmc_dim
        self.rgb_dim = rgb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Projection to common space
        self.lmc_proj = nn.Linear(lmc_dim, hidden_dim)
        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim)
        
        # Attention computation
        self.tanh = nn.Tanh()
        self.attention_weight = nn.Linear(hidden_dim, 1)
        
        # Output projection
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
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
        # Project to common space
        lmc_proj = self.lmc_proj(lmc_emb)  # (batch, seq_len, hidden_dim)
        rgb_proj = self.rgb_proj(rgb_emb)  # (batch, seq_len, hidden_dim)
        
        # Compute affinity matrix
        affinity = torch.matmul(lmc_proj, rgb_proj.transpose(-2, -1))  # (batch, seq_len, seq_len)
        
        # Normalize affinities
        lmc_attention = F.softmax(affinity, dim=-1)  # Attention over RGB for each LMC
        rgb_attention = F.softmax(affinity.transpose(-2, -1), dim=-1)  # Attention over LMC for each RGB
        
        # Apply attention
        lmc_attended = torch.matmul(lmc_attention, rgb_proj)  # (batch, seq_len, hidden_dim)
        rgb_attended = torch.matmul(rgb_attention, lmc_proj)  # (batch, seq_len, hidden_dim)
        
        # Combine attended features with original
        lmc_combined = lmc_proj + lmc_attended
        rgb_combined = rgb_proj + rgb_attended
        
        # Fuse modalities
        fused = torch.cat([lmc_combined, rgb_combined], dim=-1)
        fused = self.fusion(fused)
        
        return fused
