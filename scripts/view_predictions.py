#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_predictions(log_dir, phase='val', epoch=None):
    """
    Load predictions from CSV files.
    
    Args:
        log_dir: Path to the log directory
        phase: One of 'train', 'val', or 'test'
        epoch: The epoch number. If None, loads all epochs.
        
    Returns:
        A dictionary mapping epoch numbers to lists of prediction dictionaries
    """
    log_dir = Path(log_dir)
    decoding_dir = log_dir / "decoding_visualization"
    
    if not decoding_dir.exists():
        print(f"Decoding visualization directory not found: {decoding_dir}")
        return {}
    
    results = {}
    
    # Find all CSV files for the specified phase
    if epoch is not None:
        csv_files = [decoding_dir / f"{phase}_epoch_{epoch}.csv"]
    else:
        csv_files = list(decoding_dir.glob(f"{phase}_epoch_*.csv"))
    
    for csv_file in csv_files:
        if not csv_file.exists():
            continue
            
        # Extract epoch number from filename
        epoch_num = int(csv_file.stem.split('_')[-1])
        
        # Read CSV file
        samples = []
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    'prediction': row['prediction'],
                    'target': row['target'],
                    'cer': float(row['cer'])
                })
        
        results[epoch_num] = samples
    
    return results


def print_predictions(predictions, max_samples=None):
    """Print predictions in a formatted table."""
    if not predictions:
        print("No predictions found.")
        return
    
    for epoch, samples in sorted(predictions.items()):
        print(f"\nEPOCH {epoch} - PREDICTIONS:")
        print("-" * 80)
        print(f"{'PREDICTION':<40} | {'TARGET':<40} | {'CER'}")
        print("-" * 80)
        
        for i, sample in enumerate(samples):
            if max_samples is not None and i >= max_samples:
                break
                
            print(f"{sample['prediction']:<40} | {sample['target']:<40} | {sample['cer']:.2f}%")
        
        print("-" * 80)


def plot_cer_by_epoch(predictions):
    """Plot CER by epoch."""
    if not predictions:
        print("No predictions found.")
        return
    
    epochs = []
    cers = []
    
    for epoch, samples in sorted(predictions.items()):
        if not samples:
            continue
            
        # Calculate average CER for this epoch
        avg_cer = np.mean([sample['cer'] for sample in samples])
        epochs.append(epoch)
        cers.append(avg_cer)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cers, 'o-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Character Error Rate (%)')
    plt.title('Average Character Error Rate by Epoch')
    plt.grid(True)
    plt.savefig('cer_by_epoch.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='View predictions from training/testing')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the log directory')
    parser.add_argument('--phase', type=str, default='val', choices=['train', 'val', 'test'], 
                        help='Phase to view predictions for')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch number (default: all epochs)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to display')
    parser.add_argument('--plot', action='store_true', help='Plot CER by epoch')
    
    args = parser.parse_args()
    
    predictions = load_predictions(args.log_dir, args.phase, args.epoch)
    
    if args.plot:
        plot_cer_by_epoch(predictions)
    else:
        print_predictions(predictions, args.max_samples)


if __name__ == "__main__":
    main() 