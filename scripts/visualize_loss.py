#!/usr/bin/env python3
# Visualize loss curves from TensorBoard logs

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_scalars(log_dir):
    """Extract scalar values from TensorBoard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Print available tags
    tags = event_acc.Tags()['scalars']
    print("Available tags:", tags)
    
    # Extract loss values
    train_loss = [(s.step, s.value) for s in event_acc.Scalars('train/loss')]
    val_loss = [(s.step, s.value) for s in event_acc.Scalars('val/loss')]
    
    # Extract character error rate if available
    val_cer = None
    if 'val/CER' in tags:
        val_cer = [(s.step, s.value) for s in event_acc.Scalars('val/CER')]
    
    return train_loss, val_loss, val_cer, tags

def plot_loss_curves(train_loss, val_loss, val_cer=None, output_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy arrays for easier manipulation
    train_steps, train_values = zip(*train_loss)
    val_steps, val_values = zip(*val_loss)
    
    # Plot loss curves
    plt.subplot(2, 1, 1)
    plt.plot(train_steps, train_values, 'b-', label='Training Loss')
    plt.plot(val_steps, val_values, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot character error rate if available
    if val_cer:
        plt.subplot(2, 1, 2)
        cer_steps, cer_values = zip(*val_cer)
        plt.plot(cer_steps, cer_values, 'g-', label='Validation CER')
        plt.xlabel('Epoch')
        plt.ylabel('Character Error Rate')
        plt.title('Validation Character Error Rate')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize TensorBoard logs')
    parser.add_argument('--log_dir', type=str, default='emg2qwerty/logs/baseline', 
                        help='Path to TensorBoard logs directory')
    parser.add_argument('--output', type=str, default='loss_curves.png',
                        help='Output path for the plot image')
    args = parser.parse_args()
    
    # Find the most recent log
    if os.path.isdir(args.log_dir):
        date_dirs = sorted([d for d in os.listdir(args.log_dir) if os.path.isdir(os.path.join(args.log_dir, d))])
        if date_dirs:
            date_dir = date_dirs[-1]  # Most recent date
            time_dirs = sorted([d for d in os.listdir(os.path.join(args.log_dir, date_dir)) 
                               if os.path.isdir(os.path.join(args.log_dir, date_dir, d))])
            if time_dirs:
                time_dir = time_dirs[-1]  # Most recent time
                log_dir = os.path.join(args.log_dir, date_dir, time_dir, 'lightning_logs/version_0')
                print(f"Using log directory: {log_dir}")
                
                train_loss, val_loss, val_cer, tags = extract_tensorboard_scalars(log_dir)
                plot_loss_curves(train_loss, val_loss, val_cer, args.output)
                
                # Print final metrics
                if train_loss:
                    print(f"Final training loss: {train_loss[-1][1]:.4f}")
                if val_loss:
                    print(f"Final validation loss: {val_loss[-1][1]:.4f}")
                if val_cer:
                    print(f"Final validation CER: {val_cer[-1][1]:.4f}")
            else:
                print(f"No time directories found in {os.path.join(args.log_dir, date_dir)}")
        else:
            print(f"No date directories found in {args.log_dir}")
    else:
        print(f"Log directory {args.log_dir} not found")

if __name__ == '__main__':
    main() 