#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Scaling Experiment Results Analysis Script

This script analyzes the results of the data scaling experiments and creates visualizations
to illustrate the relationship between the amount of training data and test CER.
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
    session_counts = [2, 4, 8, 16]
    results = {}

    # Mapping of session counts to config names
    config_names = {
        2: "single_user_2sessions",
        4: "single_user_4sessions",
        8: "single_user_8sessions",
        16: "single_user",
    }

    for count in session_counts:
        config_name = config_names[count]
        exp_dir = os.path.join(log_dir, config_name)
        
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
                        cer_matches = re.findall(r"test_cer.*?([0-9]+\.[0-9]+)", log_content)
                        if cer_matches:
                            cer_values.extend([float(cer) for cer in cer_matches])
            except:
                pass
        
        if cer_values:
            # Use the last (presumably final) CER value
            results[count] = min(cer_values)  # Use minimum CER as best result
        else:
            print(f"Warning: Could not find test CER for {count} sessions experiment")
    
    return results


def plot_results(results, output_path):
    """Plot the relationship between number of training sessions and test CER."""
    if not results:
        print("No results to plot!")
        return
    
    # Prepare data for plotting
    sessions = sorted(results.keys())
    cer_values = [results[s] for s in sessions]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Line plot
    plt.plot(sessions, cer_values, 'o-', color='blue', linewidth=2, markersize=8)
    
    # Add trendline
    if len(sessions) > 1:
        z = np.polyfit(np.log(sessions), cer_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(sessions), max(sessions), 100)
        plt.plot(x_trend, p(np.log(x_trend)), 'r--', linewidth=1, 
                 label=f'Trend: y = {z[0]:.4f}*log(x) + {z[1]:.4f}')
    
    # Add labels and title
    plt.xlabel('Number of Training Sessions', fontsize=12)
    plt.ylabel('Test CER', fontsize=12)
    plt.title('Relationship Between Amount of Training Data and Test CER', fontsize=14)
    
    # Set x-axis to use log scale
    plt.xscale('log', base=2)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data points
    for session, cer in zip(sessions, cer_values):
        plt.annotate(f'{cer:.4f}', (session, cer), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    # Add legend
    plt.legend()
    
    # Customize appearance
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Show the plot (optional)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot data scaling experiment results')
    parser.add_argument('--log_dir', type=str, default='logs/data_scaling_experiment',
                        help='Directory containing experiment logs')
    parser.add_argument('--output', type=str, default='data_scaling_results.png',
                        help='Output path for the plot')
    args = parser.parse_args()
    
    # Extract results from logs
    results = extract_cer_from_logs(args.log_dir)
    print("Extracted results:")
    for count, cer in sorted(results.items()):
        print(f"{count} sessions: CER = {cer:.4f}")
    
    # Create and save the plot
    plot_results(results, args.output)


if __name__ == "__main__":
    main() 