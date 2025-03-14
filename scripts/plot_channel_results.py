#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Channel Count Experiment Results Analysis Script

This script analyzes the results of the channel count experiments and creates visualizations
to illustrate the relationship between the number of electrode channels and test CER.
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_cer_from_logs(log_dir):
    """Extract test CER from the experiment log files."""
    channel_counts = [2, 4, 8, 16]
    results = {}
    missing_counts = []

    for count in channel_counts:
        exp_dir = os.path.join(log_dir, f"channels_{count}")
        
        # Check if experiment failed
        failed_marker = os.path.join(exp_dir, 'FAILED.txt')
        is_failed = os.path.exists(failed_marker)
        if is_failed:
            print(f"Experiment for {count} channels failed. See training log for details.")
            missing_counts.append(count)
            continue
            
        # Find the test metrics in the experiment logs
        cer_values = []
        
        # Try to find JSON output files which might contain test results
        json_files = glob.glob(f"{exp_dir}/**/*.json", recursive=True)
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0 and 'test_cer' in data[0]:
                        cer_values.append(data[0]['test_cer'])
            except:
                pass
        
        # If no JSON files found with test_cer, try to grep logs
        if not cer_values:
            try:
                log_files = glob.glob(f"{exp_dir}/**/*.log", recursive=True)
                log_files.extend(glob.glob(f"{exp_dir}/**/stdout.txt", recursive=True))
                
                for log_file in log_files:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        # Look for test_cer in the log
                        cer_matches = re.findall(r"test[/_]cer.*?([0-9.]+)", log_content, re.IGNORECASE)
                        if cer_matches:
                            cer_values.extend([float(cer) for cer in cer_matches])
            except:
                pass
        
        if cer_values:
            # Use the minimum CER value (best result)
            results[count] = min(cer_values)
        else:
            print(f"Warning: Could not find test CER for {count} channels experiment")
            missing_counts.append(count)
    
    # Report missing counts
    if missing_counts:
        print(f"Missing CER values for channel counts: {missing_counts}")
    
    return results, missing_counts


def plot_results(results, missing_counts, output_path):
    """Plot the relationship between number of channels and test CER."""
    if not results:
        print("No results to plot! Creating placeholder image.")
        plt.figure(figsize=(10, 6))
        plt.title('Channel Count Experiment - No Valid Results Found', fontsize=14)
        plt.xlabel('Number of Electrode Channels', fontsize=12)
        plt.ylabel('Test CER (%)', fontsize=12)
        plt.text(0.5, 0.5, 'All experiments failed or produced no valid results.', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path)
        return
    
    # Prepare data for plotting
    channels = sorted(results.keys())
    cer_values = [results[c] for c in channels]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Line plot for available data
    plt.plot(channels, cer_values, 'o-', color='blue', linewidth=2, markersize=8, label='Successful Experiments')
    
    # Indicate missing data points
    if missing_counts:
        # Add markers for missing values
        for count in missing_counts:
            plt.axvline(x=count, color='red', linestyle='--', alpha=0.5)
            y_pos = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1
            plt.text(count, y_pos, f"Failed", color='red', 
                     ha='center', va='bottom', rotation=90, alpha=0.7)
    
    # Add trendline if possible (only for available data)
    if len(channels) > 1:
        try:
            # Try log relationship
            z = np.polyfit(np.log(channels), cer_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(channels), max(channels), 100)
            plt.plot(x_trend, p(np.log(x_trend)), 'r--', linewidth=1, 
                    label=f'Trend: y = {z[0]:.4f}*log(x) + {z[1]:.4f}')
        except:
            # Fall back to linear if log fails
            try:
                z = np.polyfit(channels, cer_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(channels), max(channels), 100)
                plt.plot(x_trend, p(x_trend), 'r--', linewidth=1, 
                        label=f'Trend: y = {z[0]:.4f}*x + {z[1]:.4f}')
            except:
                print("Could not generate trendline with available data")
    
    # Add labels and title
    plt.xlabel('Number of Electrode Channels', fontsize=12)
    plt.ylabel('Test CER (%)', fontsize=12)
    plt.title('Relationship Between Number of Electrode Channels and Test CER', fontsize=14)
    
    # Set x-axis to use log scale if we have enough data points
    if len(channels) >= 3:
        plt.xscale('log', base=2)
    
    # Set x-ticks to show all channel counts
    all_channels = [2, 4, 8, 16]
    plt.xticks(all_channels, [str(c) for c in all_channels])
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data points annotations
    for channel, cer in zip(channels, cer_values):
        plt.annotate(f'{cer:.2f}%', (channel, cer), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    # Add legend
    plt.legend()
    
    # Customize appearance
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Create a summary text file that includes missing values
    summary_path = os.path.join(os.path.dirname(output_path), "plot_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Channel Count Experiment Results\n")
        f.write("==============================\n\n")
        f.write("Number of Channels | Test CER\n")
        f.write("------------------ | -------\n")
        
        for channel in all_channels:
            if channel in results:
                f.write(f"{channel:18d} | {results[channel]:.2f}%\n")
            else:
                f.write(f"{channel:18d} | N/A (Failed or Incomplete)\n")
    
    print(f"Plot summary saved to {summary_path}")
    
    # Show the plot (optional)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot channel experiment results')
    parser.add_argument('--log_dir', type=str, default='logs/channel_experiment',
                        help='Directory containing experiment logs')
    parser.add_argument('--output', type=str, default='channel_results.png',
                        help='Output path for the plot')
    args = parser.parse_args()
    
    # Extract results from logs
    results, missing_counts = extract_cer_from_logs(args.log_dir)
    print("Extracted results:")
    for count, cer in sorted(results.items()):
        print(f"{count} channels: CER = {cer:.2f}%")
    
    # Create and save the plot
    plot_results(results, missing_counts, args.output)


if __name__ == "__main__":
    main() 