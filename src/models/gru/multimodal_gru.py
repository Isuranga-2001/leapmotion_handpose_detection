"""
Multimodal GRU model for gesture recognition.
Combines LMC and RGB encoders with fusion and GRU for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.encoders.lmc_encoder import create_lmc_encoder
from models.encoders.rgb_encoder import create_rgb_encoder
from fusion.fusion import create_fusion_module
from fusion.attention import CrossModalAttentionFusion, CoAttention


class MultimodalGRUModel(nn.Module):
    """
    Complete multimodal GRU model for gesture recognition.
    Architecture: LMC Encoder -> RGB Encoder -> Fusion -> GRU -> Classifier
    """
    
    def __init__(
        self,
        # Input dimensions
        lmc_input_dim: int = 115,
        rgb_input_dim: int = 189,
        
        # Encoder parameters
        lmc_encoder_type: str = 'mlp',
        rgb_encoder_type: str = 'mlp',
        encoder_output_dim: int = 256,
        
        # Fusion parameters
        fusion_type: str = 'concat',
        fusion_output_dim: int = 512,
        use_cross_attention: bool = False,
        
        # GRU parameters
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.2,
        gru_bidirectional: bool = False,
        
        # Classifier parameters
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        """
        Initialize multimodal GRU model.
        
        Args:
            lmc_input_dim: Dimension of LMC features
            rgb_input_dim: Dimension of RGB features
            lmc_encoder_type: Type of LMC encoder ('mlp', 'cnn', or 'hybrid')
            rgb_encoder_type: Type of RGB encoder ('mlp', 'cnn', 'lstm', or 'transformer')
            encoder_output_dim: Output dimension of encoders
            fusion_type: Type of fusion ('concat', 'weighted', 'gated', 'bilinear', 'cross_attention', 'co_attention')
            fusion_output_dim: Output dimension of fusion layer
            use_cross_attention: Whether to use cross-modal attention before fusion
            gru_hidden_dim: GRU hidden dimension
            gru_num_layers: Number of GRU layers
            gru_dropout: GRU dropout rate
            gru_bidirectional: Whether to use bidirectional GRU
            num_classes: Number of gesture classes
            dropout: Dropout rate for classifier
        """
        super(MultimodalGRUModel, self).__init__()
        
        self.lmc_input_dim = lmc_input_dim
        self.rgb_input_dim = rgb_input_dim
        self.encoder_output_dim = encoder_output_dim
        self.fusion_output_dim = fusion_output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_bidirectional = gru_bidirectional
        self.num_classes = num_classes
        
        # LMC Encoder
        self.lmc_encoder = create_lmc_encoder(
            encoder_type=lmc_encoder_type,
            input_dim=lmc_input_dim,
            output_dim=encoder_output_dim
        )
        
        # RGB Encoder
        self.rgb_encoder = create_rgb_encoder(
            encoder_type=rgb_encoder_type,
            input_dim=rgb_input_dim,
            output_dim=encoder_output_dim
        )
        
        # Optional cross-modal attention
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            if fusion_type == 'cross_attention':
                self.fusion = CrossModalAttentionFusion(
                    lmc_dim=encoder_output_dim,
                    rgb_dim=encoder_output_dim,
                    output_dim=fusion_output_dim
                )
            elif fusion_type == 'co_attention':
                self.fusion = CoAttention(
                    lmc_dim=encoder_output_dim,
                    rgb_dim=encoder_output_dim,
                    output_dim=fusion_output_dim
                )
            else:
                self.cross_attention = CrossModalAttentionFusion(
                    lmc_dim=encoder_output_dim,
                    rgb_dim=encoder_output_dim,
                    output_dim=encoder_output_dim
                )
                self.fusion = create_fusion_module(
                    fusion_type=fusion_type,
                    lmc_dim=encoder_output_dim,
                    rgb_dim=encoder_output_dim,
                    output_dim=fusion_output_dim
                )
        else:
            # Standard fusion
            self.fusion = create_fusion_module(
                fusion_type=fusion_type,
                lmc_dim=encoder_output_dim,
                rgb_dim=encoder_output_dim,
                output_dim=fusion_output_dim
            )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=fusion_output_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0,
            bidirectional=gru_bidirectional
        )
        
        # Classifier
        gru_output_dim = gru_hidden_dim * 2 if gru_bidirectional else gru_hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim // 2, num_classes)
        )
    
    def forward(
        self,
        lmc_features: torch.Tensor,
        rgb_features: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            lmc_features: LMC features (batch_size, seq_len, lmc_input_dim)
            rgb_features: RGB features (batch_size, seq_len, rgb_input_dim)
            return_hidden: Whether to return GRU hidden states
        
        Returns:
            Class logits (batch_size, num_classes) or (logits, hidden_states) if return_hidden=True
        """
        # Encode modalities
        lmc_emb = self.lmc_encoder(lmc_features)  # (batch, seq_len, encoder_output_dim)
        rgb_emb = self.rgb_encoder(rgb_features)  # (batch, seq_len, encoder_output_dim)
        
        # Apply cross-attention if enabled
        if self.use_cross_attention and hasattr(self, 'cross_attention'):
            fused = self.cross_attention(lmc_emb, rgb_emb)
        else:
            # Fuse modalities
            fused = self.fusion(lmc_emb, rgb_emb)  # (batch, seq_len, fusion_output_dim)
        
        # Temporal modeling with GRU
        gru_out, hidden = self.gru(fused)  # gru_out: (batch, seq_len, gru_hidden_dim * directions)
        
        # Use the last output for classification
        if self.gru_bidirectional:
            # For bidirectional, concatenate forward and backward final states
            last_output = gru_out[:, -1, :]
        else:
            last_output = gru_out[:, -1, :]
        
        # Classify
        logits = self.classifier(last_output)  # (batch, num_classes)
        
        if return_hidden:
            return logits, gru_out
        else:
            return logits
    
    def predict(self, lmc_features: torch.Tensor, rgb_features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (apply softmax).
        
        Args:
            lmc_features: LMC features (batch_size, seq_len, lmc_input_dim)
            rgb_features: RGB features (batch_size, seq_len, rgb_input_dim)
        
        Returns:
            Class probabilities (batch_size, num_classes)
        """
        logits = self.forward(lmc_features, rgb_features)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def get_attention_weights(
        self,
        lmc_features: torch.Tensor,
        rgb_features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract attention weights if cross-attention is used.
        
        Args:
            lmc_features: LMC features
            rgb_features: RGB features
        
        Returns:
            Attention weights or None
        """
        if not self.use_cross_attention or not hasattr(self, 'cross_attention'):
            return None
        
        # This would require modifying attention modules to return weights
        # For now, return None
        return None


class MultimodalGRUEnsemble(nn.Module):
    """
    Ensemble of multimodal GRU models for improved performance.
    """
    
    def __init__(
        self,
        num_models: int = 3,
        **model_kwargs
    ):
        """
        Initialize ensemble.
        
        Args:
            num_models: Number of models in ensemble
            **model_kwargs: Arguments for MultimodalGRUModel
        """
        super(MultimodalGRUEnsemble, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            MultimodalGRUModel(**model_kwargs) for _ in range(num_models)
        ])
    
    def forward(
        self,
        lmc_features: torch.Tensor,
        rgb_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            lmc_features: LMC features
            rgb_features: RGB features
        
        Returns:
            Averaged logits from all models
        """
        outputs = []
        for model in self.models:
            logits = model(lmc_features, rgb_features)
            outputs.append(logits)
        
        # Average predictions
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        return ensemble_output
    
    def predict(
        self,
        lmc_features: torch.Tensor,
        rgb_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Make ensemble predictions.
        
        Args:
            lmc_features: LMC features
            rgb_features: RGB features
        
        Returns:
            Class probabilities
        """
        logits = self.forward(lmc_features, rgb_features)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities


def create_multimodal_gru_model(
    num_classes: int,
    lmc_input_dim: int = 115,
    rgb_input_dim: int = 189,
    **kwargs
) -> MultimodalGRUModel:
    """
    Factory function to create multimodal GRU model with default parameters.
    
    Args:
        num_classes: Number of gesture classes
        lmc_input_dim: Dimension of LMC features
        rgb_input_dim: Dimension of RGB features
        **kwargs: Additional arguments
    
    Returns:
        MultimodalGRUModel instance
    """
    return MultimodalGRUModel(
        lmc_input_dim=lmc_input_dim,
        rgb_input_dim=rgb_input_dim,
        num_classes=num_classes,
        **kwargs
    )
