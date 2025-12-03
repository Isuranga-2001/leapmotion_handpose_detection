"""
Training script for multimodal GRU gesture recognition model.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gru.multimodal_gru import create_multimodal_gru_model
from utils.dataset import create_dataloaders


class Trainer:
    """Trainer for multimodal GRU model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        save_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use ('cuda' or 'cpu')
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Directories
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (lmc_features, rgb_features, labels) in enumerate(self.train_loader):
            # Move to device
            lmc_features = lmc_features.to(self.device)
            rgb_features = rgb_features.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(lmc_features, rgb_features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Log batch
            if batch_idx % 10 == 0:
                print(f'Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
        
        # Epoch statistics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for lmc_features, rgb_features, labels in self.val_loader:
                # Move to device
                lmc_features = lmc_features.to(self.device)
                rgb_features = rgb_features.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # Forward pass
                outputs = self.model(lmc_features, rgb_features)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        # Validation statistics
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def train(self, num_epochs: int):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Time
            epoch_time = time.time() - start_time
            
            # Log
            print(f"\nEpoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val   Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2f}%")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pth', is_best=True)
                print(f"✓ New best model saved! Val Acc: {self.best_val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final model and history
        self.save_checkpoint('final_model.pth')
        self.save_history()
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def save_history(self):
        """Save training history."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        filepath = os.path.join(self.save_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved training history: {filepath}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train multimodal GRU model')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--lmc-input-dim', type=int, default=115, help='LMC input dimension')
    parser.add_argument('--rgb-input-dim', type=int, default=189, help='RGB input dimension')
    parser.add_argument('--lmc-encoder', type=str, default='mlp', help='LMC encoder type')
    parser.add_argument('--rgb-encoder', type=str, default='mlp', help='RGB encoder type')
    parser.add_argument('--encoder-dim', type=int, default=256, help='Encoder output dimension')
    parser.add_argument('--fusion-type', type=str, default='concat', help='Fusion type')
    parser.add_argument('--fusion-dim', type=int, default=512, help='Fusion output dimension')
    parser.add_argument('--use-cross-attention', action='store_true', help='Use cross-modal attention')
    parser.add_argument('--gru-hidden-dim', type=int, default=256, help='GRU hidden dimension')
    parser.add_argument('--gru-layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--gru-bidirectional', action='store_true', help='Use bidirectional GRU')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers,
        augment_train=True
    )
    
    # Get number of classes
    num_classes = train_loader.dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print("Creating model...")
    model = create_multimodal_gru_model(
        num_classes=num_classes,
        lmc_input_dim=args.lmc_input_dim,
        rgb_input_dim=args.rgb_input_dim,
        lmc_encoder_type=args.lmc_encoder,
        rgb_encoder_type=args.rgb_encoder,
        encoder_output_dim=args.encoder_dim,
        fusion_type=args.fusion_type,
        fusion_output_dim=args.fusion_dim,
        use_cross_attention=args.use_cross_attention,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_layers,
        gru_bidirectional=args.gru_bidirectional
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
