"""
Dataset loader for multimodal (LMC + RGB) gesture recognition.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import glob


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for synchronized LMC and RGB data.
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 30,
        stride: int = 1,
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize the multimodal dataset.
        
        Args:
            data_dir: Directory containing synchronized data files
            sequence_length: Number of frames per sequence
            stride: Stride for sliding window over sequences
            augment: Whether to apply data augmentation
            normalize: Whether to normalize features
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.augment = augment
        self.normalize = normalize
        
        # Load all synchronized data files
        self.sequences = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        self._load_data()
        
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_data(self):
        """Load all synchronized data files from the data directory."""
        # Find all JSON files in the data directory
        json_files = glob.glob(os.path.join(self.data_dir, '**/*.json'), recursive=True)
        
        for json_file in json_files:
            # Extract label from filename or directory structure
            # Assuming format: data_dir/gesture_name/sample_timestamp.json
            label = self._extract_label(json_file)
            
            # Load synchronized data
            with open(json_file, 'r') as f:
                synchronized_frames = json.load(f)
            
            # Create sequences using sliding window
            sequences = self._create_sequences(synchronized_frames)
            
            # Add to dataset
            for seq in sequences:
                self.sequences.append(seq)
                self.labels.append(label)
            
            # Update label mappings
            if label not in self.label_to_idx:
                idx = len(self.label_to_idx)
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label
    
    def _extract_label(self, file_path: str) -> str:
        """Extract gesture label from file path."""
        # Assuming structure: data_dir/gesture_name/file.json
        parts = os.path.normpath(file_path).split(os.sep)
        
        # Try to find label in path
        for i, part in enumerate(parts):
            if part == os.path.basename(self.data_dir) and i + 1 < len(parts):
                return parts[i + 1]
        
        # Fallback: extract from filename
        filename = os.path.basename(file_path)
        label = filename.split('_')[0]
        return label
    
    def _create_sequences(self, frames: List[Dict]) -> List[Dict]:
        """
        Create fixed-length sequences from frames using sliding window.
        
        Args:
            frames: List of synchronized frames
        
        Returns:
            List of sequences
        """
        sequences = []
        
        for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
            sequence_frames = frames[i:i + self.sequence_length]
            
            # Extract LMC and RGB features
            lmc_features = []
            rgb_features = []
            
            for frame in sequence_frames:
                # LMC features
                if 'lmc_features' in frame and frame['lmc_features'] is not None:
                    lmc_features.append(frame['lmc_features'])
                elif 'lmc' in frame and frame['lmc'] is not None:
                    lmc_features.append(frame['lmc'])
                else:
                    lmc_features.append(np.zeros(81))  # Default: 27 joints × 3
                
                # RGB features
                if 'rgb_features' in frame and frame['rgb_features'] is not None:
                    rgb_features.append(frame['rgb_features'])
                elif 'rgb' in frame and frame['rgb'] is not None:
                    # Flatten RGB landmarks if available
                    rgb_flat = np.array(frame['rgb']).flatten()
                    rgb_features.append(rgb_flat)
                else:
                    rgb_features.append(np.zeros(189))  # Default feature dimension
            
            sequences.append({
                'lmc': np.array(lmc_features),
                'rgb': np.array(rgb_features)
            })
        
        return sequences
    
    def _compute_normalization_stats(self):
        """Compute mean and std for feature normalization."""
        all_lmc = []
        all_rgb = []
        
        for seq in self.sequences:
            all_lmc.append(seq['lmc'])
            all_rgb.append(seq['rgb'])
        
        all_lmc = np.concatenate(all_lmc, axis=0)
        all_rgb = np.concatenate(all_rgb, axis=0)
        
        self.lmc_mean = np.mean(all_lmc, axis=0)
        self.lmc_std = np.std(all_lmc, axis=0) + 1e-8
        
        self.rgb_mean = np.mean(all_rgb, axis=0)
        self.rgb_std = np.std(all_rgb, axis=0) + 1e-8
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its label.
        
        Args:
            idx: Index of the sequence
        
        Returns:
            Tuple of (lmc_sequence, rgb_sequence, label)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        # Get features
        lmc_features = sequence['lmc'].copy()
        rgb_features = sequence['rgb'].copy()
        
        # Normalize
        if self.normalize:
            lmc_features = (lmc_features - self.lmc_mean) / self.lmc_std
            rgb_features = (rgb_features - self.rgb_mean) / self.rgb_std
        
        # Augment
        if self.augment:
            lmc_features, rgb_features = self._augment(lmc_features, rgb_features)
        
        # Convert to tensors
        lmc_tensor = torch.FloatTensor(lmc_features)
        rgb_tensor = torch.FloatTensor(rgb_features)
        label_tensor = torch.LongTensor([label_idx])
        
        return lmc_tensor, rgb_tensor, label_tensor
    
    def _augment(self, lmc_features: np.ndarray, rgb_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to features.
        
        Args:
            lmc_features: LMC feature sequence
            rgb_features: RGB feature sequence
        
        Returns:
            Augmented features
        """
        # Random noise
        if np.random.rand() < 0.5:
            lmc_features += np.random.normal(0, 0.01, lmc_features.shape)
            rgb_features += np.random.normal(0, 0.01, rgb_features.shape)
        
        # Time warping (simple speed variation)
        if np.random.rand() < 0.3:
            speed_factor = np.random.uniform(0.8, 1.2)
            new_length = int(len(lmc_features) * speed_factor)
            if new_length > 0:
                indices = np.linspace(0, len(lmc_features) - 1, new_length).astype(int)
                lmc_features = lmc_features[indices]
                rgb_features = rgb_features[indices]
                
                # Pad or truncate to original length
                if len(lmc_features) < self.sequence_length:
                    pad_length = self.sequence_length - len(lmc_features)
                    lmc_features = np.pad(lmc_features, ((0, pad_length), (0, 0)), mode='edge')
                    rgb_features = np.pad(rgb_features, ((0, pad_length), (0, 0)), mode='edge')
                else:
                    lmc_features = lmc_features[:self.sequence_length]
                    rgb_features = rgb_features[:self.sequence_length]
        
        return lmc_features, rgb_features
    
    def get_num_classes(self) -> int:
        """Return the number of gesture classes."""
        return len(self.label_to_idx)
    
    def get_label_name(self, idx: int) -> str:
        """Get gesture label name from index."""
        return self.idx_to_label.get(idx, "Unknown")


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    sequence_length: int = 30,
    num_workers: int = 4,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size
        sequence_length: Number of frames per sequence
        num_workers: Number of worker processes
        augment_train: Whether to augment training data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = MultimodalDataset(
        train_dir,
        sequence_length=sequence_length,
        augment=augment_train,
        normalize=True
    )
    
    val_dataset = MultimodalDataset(
        val_dir,
        sequence_length=sequence_length,
        augment=False,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
