# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    LSTMEncoder,
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TransformerEncoder,
)
from emg2qwerty.conformer_modules import ConformerEncoder
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        # Log metrics at the end of each epoch
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class LSTMCTCModule(pl.LightningModule):
    """LSTM-based CTC module to replace TDSConvCTCModule"""
    
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16
    
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        num_features = self.NUM_BANDS * mlp_features[-1]
        
        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            # Replace TDSConvEncoder with LSTMEncoder
            LSTMEncoder(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        
        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        
        # Decoder
        self.decoder = instantiate(decoder)
        
        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size
        
        emissions = self.forward(inputs)
        
        # Unlike the TDSConvEncoder, the LSTMEncoder doesn't reduce time dimensions
        # so we don't need to adjust the emission lengths
        emission_lengths = input_lengths
        
        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )
        
        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        
        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)
        
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss
    
    def _epoch_end(self, phase: str) -> None:
        # Log metrics at the end of each epoch
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()
    
    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)
    
    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)
    
    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)
    
    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")
    
    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")
    
    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")
    
    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TransformerCTCModule(pl.LightningModule):
    """Transformer model for EMG-to-text decoding using CTC loss."""

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int = 384,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 500,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        num_features = self.NUM_BANDS * mlp_features[-1]
        
        # Initial feature extraction - similar to LSTMCTCModule
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.feature_extractor = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
        )
        
        # Transformer encoder - needs separate handling for masking
        self.transformer_encoder = TransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            max_seq_length=max_seq_length,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
        )
        
        # Output projection - similar to LSTMCTCModule
        self.classifier = nn.Sequential(
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        
        # Decoder
        self.decoder = instantiate(decoder) if decoder else None
        
        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor, input_lengths=None) -> torch.Tensor:
        # Feature extraction
        x = self.feature_extractor(inputs)
        
        # Create attention mask if we have sequence lengths
        mask = None
        if input_lengths is not None:
            mask = torch.zeros(x.size(1), x.size(0), device=x.device, dtype=torch.bool)
            for i, length in enumerate(input_lengths):
                mask[i, length:] = True  # Mask positions beyond the sequence length
        
        # Apply transformer with masking
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Output classification
        x = self.classifier(x)
        
        return x
        
    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        # Apply gradient clipping in training phase
        if phase == "train" and hasattr(self.trainer, "gradient_clip_val") and self.trainer.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer=self.optimizers(), 
                gradient_clip_val=self.trainer.gradient_clip_val, 
                gradient_clip_algorithm="norm"
            )

        # Model forward pass to get log probabilities with masking
        emissions = self(inputs, input_lengths)

        # Adjust input_lengths based on model's sequence length reduction
        # In transformer, we don't have automatic length reduction like convolution
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff if T_diff > 0 else input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Log loss
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)

        # Only decode and compute metrics if we have a decoder
        if self.decoder:
            # Decode predictions using the same method as in other modules
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

            # Update metrics
            metrics = self.metrics[f"{phase}_metrics"]
            targets_np = targets.detach().cpu().numpy()
            target_lengths_np = target_lengths.detach().cpu().numpy()
            for i in range(N):
                # Unpad targets (T, N) for batch entry
                target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
                metrics.update(prediction=predictions[i], target=target)

        return loss

    def _epoch_end(self, phase: str) -> None:
        # Log metrics at the end of each epoch
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class ConformerCTCModule(pl.LightningModule):
    """Conformer model for EMG-to-text decoding using CTC loss.
    
    Similar structure to TDSConvCTCModule, with Conformer encoder instead of TDSConvEncoder.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        d_ff: int = 2048,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        activation: str = "swish",
        max_seq_length: int = 500,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Calculate expected feature dimensions
        num_features = self.NUM_BANDS * mlp_features[-1]
        
        # Feature extraction - matches TDSConvCTCModule structure
        self.feature_extractor = nn.Sequential(
            # (T, N, bands=2, C=16, freq) -> (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, C=16, freq) -> (T, N, bands=2, mlp_features[-1]=384)
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, bands=2, 384) -> (T, N, 768)
            nn.Flatten(start_dim=2),
        )
        
        # Conformer encoder - processes the flattened features
        # Input: (T, N, 768) -> Output: (T, N, d_model=256)
        self.conformer_encoder = ConformerEncoder(
            num_features=num_features,  # 768
            d_model=d_model,            # 256
            nhead=nhead,                # 4
            num_encoder_layers=num_encoder_layers,
            d_ff=d_ff,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )
        
        # Output projection - same as TDSConvCTCModule
        # Input: (T, N, d_model=256) -> Output: (T, N, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion - same as TDSConvCTCModule
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        
        # Decoder - same as TDSConvCTCModule
        self.decoder = instantiate(decoder) if decoder else None
        
        # Metrics - same as TDSConvCTCModule
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor, input_lengths=None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Input tensor of shape (T, N, bands=2, C=16, freq)
            input_lengths: Optional lengths of each sequence for masking
            
        Returns:
            Log probabilities of shape (T', N, num_classes)
        """
        # Feature extraction: (T, N, bands=2, C=16, freq) -> (T, N, 768)
        x = self.feature_extractor(inputs)
        
        # Create attention mask if we have sequence lengths
        mask = None
        if input_lengths is not None:
            mask = torch.zeros(x.size(1), x.size(0), device=x.device, dtype=torch.bool)
            for i, length in enumerate(input_lengths):
                mask[i, length:] = True  # Mask positions beyond the sequence length
        
        # Apply conformer with masking: (T, N, 768) -> (T, N, d_model=256)
        x = self.conformer_encoder(x, src_key_padding_mask=mask)
        
        # Output classification: (T, N, 256) -> (T, N, num_classes)
        x = self.classifier(x)
        
        return x
        
    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        # Apply gradient clipping in training phase
        if phase == "train" and hasattr(self.trainer, "gradient_clip_val") and self.trainer.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer=self.optimizers(), 
                gradient_clip_val=self.trainer.gradient_clip_val, 
                gradient_clip_algorithm="norm"
            )

        # Model forward pass to get log probabilities with masking
        emissions = self(inputs, input_lengths)

        # Adjust input_lengths based on model's sequence length reduction
        # In conformer, we don't have automatic length reduction like convolution
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff if T_diff > 0 else input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Log loss
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)

        # Only decode and compute metrics if we have a decoder
        if self.decoder:
            # Decode predictions using the same method as in other modules
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

            # Update metrics
            metrics = self.metrics[f"{phase}_metrics"]
            targets_np = targets.detach().cpu().numpy()
            target_lengths_np = target_lengths.detach().cpu().numpy()
            for i in range(N):
                # Unpad targets (T, N) for batch entry
                target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
                metrics.update(prediction=predictions[i], target=target)

        return loss

    def _epoch_end(self, phase: str) -> None:
        # Log metrics at the end of each epoch
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
