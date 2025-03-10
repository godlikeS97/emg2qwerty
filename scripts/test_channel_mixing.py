#!/usr/bin/env python3
"""
Script to test and visualize the effect of cross-channel mixing on EMG signals.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from emg2qwerty.data import EMGSessionData
from emg2qwerty.transforms import ToTensor, CrossChannelMixing


def plot_channels(ax, data, title, channels=None, sample_rate=2000):
    """Plot multiple EMG channels with offsets for better visualization."""
    time = np.arange(len(data)) / sample_rate
    channels = channels or list(range(data.shape[1]))
    
    for i, ch in enumerate(channels):
        # Add offset for better visualization
        offset = i * 0.5
        signal = data[:, ch] + offset
        ax.plot(time, signal, label=f"Channel {ch}")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (offset for clarity)")
    ax.set_title(title)
    ax.legend(loc='right')


def main():
    parser = argparse.ArgumentParser(description="Test cross-channel mixing on EMG data")
    parser.add_argument("session_file", type=str, help="Path to EMG session file (.h5)")
    parser.add_argument("--window_length", type=int, default=4000, 
                        help="Window length to analyze (default: 4000 samples = 2s at 2kHz)")
    parser.add_argument("--mix_ratio_min", type=float, default=0.1, 
                        help="Minimum mixing ratio")
    parser.add_argument("--mix_ratio_max", type=float, default=0.3, 
                        help="Maximum mixing ratio")
    parser.add_argument("--num_channels", type=int, default=8, 
                        help="Number of channels to mix")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save figure (default: show on screen)")
    parser.add_argument("--display_channels", type=str, default="0,1,2,3",
                        help="Comma-separated list of channel indices to display")
    args = parser.parse_args()

    # Load EMG data
    session_path = Path(args.session_file)
    if not session_path.exists():
        print(f"Error: Session file {session_path} does not exist.")
        return 1
        
    print(f"Loading EMG session from {session_path}...")
    with EMGSessionData(session_path) as session:
        # Get a window of data
        window = session[1000:1000+args.window_length]
        
        # Convert to tensor
        to_tensor = ToTensor(fields=("emg_left", "emg_right"))
        emg_tensor = to_tensor(window)
        
        # Create a copy for the mixed version
        emg_tensor_orig = emg_tensor.clone()
        
        # Apply cross-channel mixing
        channel_mixing = CrossChannelMixing(
            mix_ratio_range=(args.mix_ratio_min, args.mix_ratio_max),
            num_channels_to_mix=args.num_channels,
            seed=args.seed
        )
        emg_tensor_mixed = channel_mixing(emg_tensor)
        
        # Display channels specified by user
        display_channels = [int(ch) for ch in args.display_channels.split(',')]
        
        # Select left and right bands for visualization
        left_orig = emg_tensor_orig[:, 0, :].numpy()
        left_mixed = emg_tensor_mixed[:, 0, :].numpy()
        right_orig = emg_tensor_orig[:, 1, :].numpy()
        right_mixed = emg_tensor_mixed[:, 1, :].numpy()
        
        # Create the figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot channels
        plot_channels(axs[0, 0], left_orig, "Original Left EMG Channels", 
                     channels=display_channels)
        plot_channels(axs[0, 1], left_mixed, "Mixed Left EMG Channels", 
                     channels=display_channels)
        plot_channels(axs[1, 0], right_orig, "Original Right EMG Channels", 
                     channels=display_channels)
        plot_channels(axs[1, 1], right_mixed, "Mixed Right EMG Channels", 
                     channels=display_channels)
        
        plt.tight_layout()
        fig.suptitle(f"Cross-Channel Mixing with {args.mix_ratio_min}-{args.mix_ratio_max} ratio, {args.num_channels} channels", 
                    fontsize=16, y=1.02)
        
        if args.output:
            plt.savefig(args.output, bbox_inches='tight')
            print(f"Figure saved to {args.output}")
        else:
            plt.show()
            
    return 0


if __name__ == "__main__":
    sys.exit(main()) 