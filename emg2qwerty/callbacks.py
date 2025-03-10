# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional
import os
import csv
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from emg2qwerty.data import LabelData


class DecodingVisualizationCallback(Callback):
    """
    A callback to visualize and compare decoding results with ground truth.
    
    This callback logs sample predictions and ground truth at the end of each epoch
    during training, validation, and testing.
    
    Args:
        num_samples: Number of samples to log per epoch
        log_dir: Directory to save the logs
    """
    
    def __init__(
        self, 
        num_samples: int = 5, 
        log_dir: Optional[str] = None
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.log_dir = log_dir
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        # Dictionary to store all samples across epochs
        self.all_samples = {
            'train': {},
            'val': {},
            'test': {}
        }
        
    def _reset_samples(self) -> None:
        """Reset sample collections."""
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize log directory at the start of training."""
        if self.log_dir is None:
            self.log_dir = Path(trainer.log_dir) / "decoding_visualization"
        else:
            # Ensure log_dir is a Path object
            self.log_dir = Path(self.log_dir)
        
        os.makedirs(self.log_dir, exist_ok=True)
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Collect samples during training."""
        if hasattr(pl_module, "train_sample_predictions") and len(self.train_samples) < self.num_samples:
            samples = pl_module.train_sample_predictions
            self.train_samples.extend(samples[:self.num_samples - len(self.train_samples)])
    
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Collect samples during validation."""
        if hasattr(pl_module, "val_sample_predictions") and len(self.val_samples) < self.num_samples:
            samples = pl_module.val_sample_predictions
            self.val_samples.extend(samples[:self.num_samples - len(self.val_samples)])
    
    def on_test_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Collect samples during testing."""
        if hasattr(pl_module, "test_sample_predictions") and len(self.test_samples) < self.num_samples:
            samples = pl_module.test_sample_predictions
            self.test_samples.extend(samples[:self.num_samples - len(self.test_samples)])
    
    def _write_samples_to_csv(self, samples: List[Dict[str, Any]], phase: str, epoch: int) -> None:
        """Write samples to CSV file."""
        if not samples:
            return
            
        # Convert log_dir to Path object if it's a string
        log_dir = Path(self.log_dir) if isinstance(self.log_dir, str) else self.log_dir
        filename = log_dir / f"{phase}_epoch_{epoch}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['prediction', 'target', 'cer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample in samples:
                writer.writerow({
                    'prediction': sample['prediction'].text,
                    'target': sample['target'].text,
                    'cer': sample['cer']
                })
                
        # Also print to console for immediate feedback
        print(f"\n{phase.upper()} EPOCH {epoch} - SAMPLE PREDICTIONS:")
        print("-" * 80)
        print(f"{'PREDICTION':<40} | {'TARGET':<40} | {'CER'}")
        print("-" * 80)
        for sample in samples:
            print(f"{sample['prediction'].text:<40} | {sample['target'].text:<40} | {sample['cer']:.2f}%")
        print("-" * 80)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log training samples at the end of each epoch."""
        self._write_samples_to_csv(self.train_samples, "train", trainer.current_epoch)
        # Store samples for later retrieval
        self.all_samples['train'][trainer.current_epoch] = self.train_samples.copy()
        self.train_samples = []
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log validation samples at the end of each epoch."""
        self._write_samples_to_csv(self.val_samples, "val", trainer.current_epoch)
        # Store samples for later retrieval
        self.all_samples['val'][trainer.current_epoch] = self.val_samples.copy()
        self.val_samples = []
    
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log test samples at the end of testing."""
        self._write_samples_to_csv(self.test_samples, "test", trainer.current_epoch)
        # Store samples for later retrieval
        self.all_samples['test'][trainer.current_epoch] = self.test_samples.copy()
        self.test_samples = []

    def get_predictions(self, phase: str = 'val', epoch: Optional[int] = None):
        """
        Retrieve the predictions and targets for a specific phase and epoch.
        
        Args:
            phase: One of 'train', 'val', or 'test'
            epoch: The epoch number. If None, returns the most recent epoch.
            
        Returns:
            A list of dictionaries containing 'prediction', 'target', and 'cer'
        """
        if phase not in self.all_samples or not self.all_samples[phase]:
            return []
            
        if epoch is None:
            # Get the most recent epoch
            epoch = max(self.all_samples[phase].keys())
            
        return self.all_samples[phase].get(epoch, []) 