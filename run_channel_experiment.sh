#!/bin/bash

# Experiment to analyze the relationship between number of electrode channels and test CER
# Author: Claude AI Assistant
# Date: $(date +"%Y-%m-%d")

# Define output directory for results
RESULTS_DIR="logs/channel_experiment"

# Remove the results directory if it exists
rm -rf $RESULTS_DIR

# Create the results directory
mkdir -p $RESULTS_DIR

# Log file to record experiment status
LOG_FILE="${RESULTS_DIR}/experiment_log.txt"
echo "EMG2QWERTY Channel Count Experiment Log - $(date)" > ${LOG_FILE}
echo "=======================================" >> ${LOG_FILE}

# Function to run training with specific channel count
run_experiment() {
    channel_count=$1
    echo "Running experiment with $channel_count channels..." | tee -a ${LOG_FILE}
    
    # Create experiment directory
    EXPERIMENT_DIR="${RESULTS_DIR}/channels_${channel_count}"
    mkdir -p ${EXPERIMENT_DIR}
    
    echo "Start time: $(date)" | tee -a ${LOG_FILE}
    
    # Set the correct model configuration based on channel count
    if [ "$channel_count" -eq 16 ]; then
        # For baseline, use the default model config
        MODEL_CONFIG="tds_conv_ctc"
    else
        # For other channel counts, use specialized configs
        MODEL_CONFIG="tds_conv_ctc_${channel_count}channels"
    fi
    
    # Run training with appropriate model configuration
    CMD="python -m emg2qwerty.train \
      user=single_user \
      transforms=channels_${channel_count} \
      model=${MODEL_CONFIG} \
      trainer.accelerator=gpu \
      trainer.devices=1 \
      trainer.max_epochs=50 \
      batch_size=32 \
      +callback=decoding_visualization \
      hydra.run.dir=$EXPERIMENT_DIR"
    
    echo "Command: ${CMD}" >> ${LOG_FILE}
    
    # Execute the command and redirect output to a log file
    eval ${CMD} 2>&1 | tee "${EXPERIMENT_DIR}/training.log"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED"
        # Create a note in the experiment directory to indicate failure
        echo "Experiment failed. See training.log for details." > "${EXPERIMENT_DIR}/FAILED.txt"
    fi
    
    echo "End time: $(date)" | tee -a ${LOG_FILE}
    echo "Status: ${STATUS}" | tee -a ${LOG_FILE}
    echo "================================================" | tee -a ${LOG_FILE}
}

# Run experiments with different numbers of channels
echo "Starting channel count experiments..." | tee -a ${LOG_FILE}
echo "================================================" | tee -a ${LOG_FILE}

# 2 channels experiment
run_experiment 2

# 4 channels experiment
run_experiment 4

# 8 channels experiment
run_experiment 8

# 16 channels (baseline) experiment
run_experiment 16

echo "All experiments completed."
echo "Results are stored in $RESULTS_DIR"

# Create a summary of the channel count experiments
echo "Creating summary report..."
python -c "
import os
import glob
import re

results_dir = '${RESULTS_DIR}'
summary_file = os.path.join(results_dir, 'detailed_summary.txt')

# Define placeholder values for experiments that did not complete
placeholder_cer = 'N/A (Experiment Failed or Incomplete)'

with open(summary_file, 'w') as f:
    f.write('EMG2QWERTY Channel Count Experiment Detailed Summary\n')
    f.write('===========================================\n\n')
    
    # Find all experiment directories
    channel_counts = [2, 4, 8, 16]
    channel_results = {}
    
    for count in channel_counts:
        exp_dir = os.path.join(results_dir, f'channels_{count}')
        
        # Check if experiment failed
        failed_marker = os.path.join(exp_dir, 'FAILED.txt')
        is_failed = os.path.exists(failed_marker)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(exp_dir):
            f.write(f'Experiment: channels_{count} (Directory not found)\n\n')
            # Add placeholder for summary
            channel_results[count] = placeholder_cer
            continue
        
        # Find the training log file
        log_file = os.path.join(exp_dir, 'training.log')
        
        if not os.path.exists(log_file):
            # Try to find any log file
            log_files = glob.glob(os.path.join(exp_dir, '**/*.log'), recursive=True)
            log_files.extend(glob.glob(os.path.join(exp_dir, '**/stdout.txt'), recursive=True))
            
            if log_files:
                log_file = log_files[0]  # Use the first log file found
            else:
                f.write(f'Experiment: channels_{count} (No log files found)\n\n')
                # Add placeholder for summary
                channel_results[count] = placeholder_cer
                continue
        
        # Initialize metrics
        final_epoch = 'N/A'
        val_cer = 'N/A'
        test_cer = 'N/A'
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as log:
                content = log.read()
                
                # Look for epoch information
                epoch_matches = re.findall(r'Epoch ([0-9]+)', content)
                if epoch_matches:
                    final_epoch = epoch_matches[-1]
                
                # Look for validation CER
                val_cer_matches = re.findall(r'val[\w_/]*CER[: \']+([0-9.]+)', content)
                if val_cer_matches:
                    val_cer = val_cer_matches[-1]
                else:
                    # Alternative pattern for validation CER
                    val_cer_matches = re.findall(r'validation.*?CER[: \']+([0-9.]+)', content)
                    if val_cer_matches:
                        val_cer = val_cer_matches[-1]
                
                # Look for test CER
                test_cer_matches = re.findall(r'test[\w_/]*CER[: \']+([0-9.]+)', content)
                if test_cer_matches:
                    test_cer = test_cer_matches[-1]
                else:
                    # Alternative pattern for test CER
                    test_cer_matches = re.findall(r'test.*?CER[: \']+([0-9.]+)', content)
                    if test_cer_matches:
                        test_cer = test_cer_matches[-1]
                
                # If we still haven't found specific validation/test CER metrics,
                # let's check for the general CER and note it's not specified
                if val_cer == 'N/A' and test_cer == 'N/A':
                    cer_matches = re.findall(r'CER[: \']+([0-9.]+)', content)
                    if cer_matches:
                        val_cer = cer_matches[-1] + ' (unspecified)'
        except Exception as e:
            print(f'Error processing log file {log_file}: {str(e)}')
        
        # Store results for later, use placeholder for missing values
        if test_cer != 'N/A':
            channel_results[count] = test_cer
        else:
            status = 'Failed' if is_failed else 'Incomplete'
            channel_results[count] = f'N/A ({status})'
        
        # Write experiment summary
        f.write(f'Experiment: channels_{count}\n')
        f.write(f'  Channels: {count}\n')
        f.write(f'  Status: {\"Failed\" if is_failed else \"Complete\"}\n')
        f.write(f'  Final Epoch: {final_epoch}\n')
        f.write(f'  Validation CER: {val_cer}\n')
        f.write(f'  Test CER: {test_cer}\n')
        f.write('\n')
    
    # Create a simple tabular summary
    f.write('Channel Count Results Summary\n')
    f.write('=========================\n\n')
    f.write('Number of Channels | Test CER\n')
    f.write('------------------ | -------\n')
    
    # Write the tabular summary for all channels regardless of status
    for count in sorted(channel_results.keys()):
        f.write(f'{count:16d} | {channel_results[count]}\n')

# Also update the simple summary file
with open(os.path.join(results_dir, 'summary.txt'), 'w') as summary:
    summary.write('Number of Channels | Test CER\\n')
    summary.write('------------------ | --------\\n')
    
    for count in sorted(channel_results.keys()):
        summary.write(f'{count:18d} | {channel_results[count]}\\n')

print(f'Summary created at {summary_file}')
"

# Run the plotting script to visualize results
python scripts/plot_channel_results.py --log_dir $RESULTS_DIR --output "${RESULTS_DIR}/channel_results.png"

echo "Channel count experiment completed successfully!"
echo "Please check the detailed summary at $RESULTS_DIR/detailed_summary.txt"
echo "A simpler summary is available at $RESULTS_DIR/summary.txt"

# Print results overview
echo "Results overview:"
if [ -f "${RESULTS_DIR}/summary.txt" ]; then
    cat "${RESULTS_DIR}/summary.txt"
else
    echo "Summary file not found. Check individual experiment logs for results."
fi 