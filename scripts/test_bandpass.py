#!/usr/bin/env python3
"""
Script to test and visualize the effect of bandpass filtering on EMG signals.
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
from emg2qwerty.transforms import ToTensor, BandpassFilter


def plot_waveform(ax, signal, title, sample_rate=2000):
    """Plot a time-domain waveform."""
    time = np.arange(len(signal)) / sample_rate
    ax.plot(time, signal)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)


def plot_spectrum(ax, signal, title, sample_rate=2000):
    """Plot the frequency spectrum of a signal."""
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1.0/sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    
    ax.plot(freq, spectrum)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.set_xlim(0, 1000)  # Focus on 0-1000 Hz range


def main():
    parser = argparse.ArgumentParser(description="Test bandpass filtering on EMG data")
    parser.add_argument("session_file", type=str, help="Path to EMG session file (.h5)")
    parser.add_argument("--window_length", type=int, default=2000, 
                        help="Window length to analyze (default: 2000 samples = 1s at 2kHz)")
    parser.add_argument("--low_cut", type=float, default=20.0, 
                        help="Lower cutoff frequency (Hz)")
    parser.add_argument("--high_cut", type=float, default=450.0, 
                        help="Upper cutoff frequency (Hz)")
    parser.add_argument("--channel", type=int, default=0, 
                        help="EMG channel to analyze (0-15)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save figure (default: show on screen)")
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
        
        # Create a copy for the filtered version
        emg_tensor_orig = emg_tensor.clone()
        
        # Apply bandpass filter
        bandpass = BandpassFilter(
            low_cut=args.low_cut,
            high_cut=args.high_cut,
            sample_rate=2000.0
        )
        emg_tensor_filtered = bandpass(emg_tensor)
        
        # Select one channel from one band for visualization
        band_idx = 0  # 0=left, 1=right
        channel_idx = args.channel
        
        emg_orig = emg_tensor_orig[:, band_idx, channel_idx].numpy()
        emg_filtered = emg_tensor_filtered[:, band_idx, channel_idx].numpy()
        
        # Create the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot time domain signals
        plot_waveform(axs[0, 0], emg_orig, "Original EMG Signal")
        plot_waveform(axs[0, 1], emg_filtered, "Bandpass Filtered EMG Signal")
        
        # Plot frequency domain signals
        plot_spectrum(axs[1, 0], emg_orig, "Original Spectrum")
        plot_spectrum(axs[1, 1], emg_filtered, f"Filtered Spectrum ({args.low_cut}-{args.high_cut} Hz)")
        
        plt.tight_layout()
        
        if args.output:
            plt.savefig(args.output)
            print(f"Figure saved to {args.output}")
        else:
            plt.show()
            
    return 0


if __name__ == "__main__":
    sys.exit(main()) 