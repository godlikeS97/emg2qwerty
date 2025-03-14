#!/bin/bash

# Experiment to analyze the relationship between training data amount and test CER
# Author: Claude AI Assistant
# Date: $(date +"%Y-%m-%d")

# Define output directory for results
RESULTS_DIR="logs/data_scaling_experiment"

# remove the results directory if it exists
rm -rf $RESULTS_DIR

# create the results directory
mkdir -p $RESULTS_DIR

# Log file to record experiment status
LOG_FILE="${RESULTS_DIR}/experiment_log.txt"
echo "EMG2QWERTY Data Scaling Experiment Log - $(date)" > ${LOG_FILE}
echo "=======================================" >> ${LOG_FILE}

# Function to run training with specific user config and log results
run_experiment() {
    config_name=$1
    num_sessions=$2
    echo "Running experiment with $config_name configuration ($num_sessions sessions)..." | tee -a ${LOG_FILE}
    
    # Create experiment directory
    EXPERIMENT_DIR="${RESULTS_DIR}/${config_name}"
    mkdir -p ${EXPERIMENT_DIR}
    
    echo "Start time: $(date)" | tee -a ${LOG_FILE}
    
    # Run training with 50 epochs
    CMD="python -m emg2qwerty.train \
      user=\"$config_name\" \
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
    fi
    
    echo "End time: $(date)" | tee -a ${LOG_FILE}
    echo "Status: ${STATUS}" | tee -a ${LOG_FILE}
    echo "================================================" | tee -a ${LOG_FILE}
}

# Run experiments with different numbers of training sessions
echo "Starting data scaling experiments..." | tee -a ${LOG_FILE}
echo "================================================" | tee -a ${LOG_FILE}

# 2 sessions experiment
run_experiment "single_user_2sessions" 2

# 4 sessions experiment
run_experiment "single_user_4sessions" 4

# 8 sessions experiment
run_experiment "single_user_8sessions" 8

# 16 sessions (baseline) experiment
run_experiment "single_user" 16

echo "All experiments completed."
echo "Results are stored in $RESULTS_DIR"

# Create a summary of the data scaling experiments
echo "Creating summary report..."
python -c "
import os
import glob
import re

results_dir = '${RESULTS_DIR}'
summary_file = os.path.join(results_dir, 'detailed_summary.txt')

with open(summary_file, 'w') as f:
    f.write('EMG2QWERTY Data Scaling Experiment Detailed Summary\n')
    f.write('===========================================\n\n')
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(results_dir, '*'))
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        
        # Skip any non-directory files
        if not os.path.isdir(exp_dir) or exp_name.endswith('.txt'):
            continue
        
        # Get session count
        if exp_name == 'single_user':
            session_count = 16
        else:
            session_match = re.search(r'(\d+)', exp_name)
            session_count = int(session_match.group(1)) if session_match else 0
            
        # Find the training log file
        log_file = os.path.join(exp_dir, 'training.log')
        
        if not os.path.exists(log_file):
            # Try to find any log file
            log_files = glob.glob(os.path.join(exp_dir, '**/*.log'), recursive=True)
            log_files.extend(glob.glob(os.path.join(exp_dir, '**/stdout.txt'), recursive=True))
            
            if log_files:
                log_file = log_files[0]  # Use the first log file found
            else:
                f.write(f'Experiment: {exp_name} (No log files found)\n\n')
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
        
        # Write experiment summary
        f.write(f'Experiment: {exp_name}\n')
        f.write(f'  Sessions: {session_count}\n')
        f.write(f'  Final Epoch: {final_epoch}\n')
        f.write(f'  Validation CER: {val_cer}\n')
        f.write(f'  Test CER: {test_cer}\n')
        f.write('\n')
    
    # Create a simple tabular summary
    f.write('Data Scaling Results Summary\n')
    f.write('=========================\n\n')
    f.write('Number of Sessions | Test CER\n')
    f.write('------------------ | -------\n')
    
    # Collect results for each session count
    session_results = {}
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        
        # Skip any non-directory files
        if not os.path.isdir(exp_dir) or exp_name.endswith('.txt'):
            continue
        
        # Get session count
        if exp_name == 'single_user':
            session_count = 16
        else:
            session_match = re.search(r'(\d+)', exp_name)
            session_count = int(session_match.group(1)) if session_match else 0
        
        if session_count == 0:
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
                continue
        
        # Extract test CER
        test_cer = 'N/A'
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as log:
                content = log.read()
                
                # Look for test CER with the same patterns as above
                test_cer_matches = re.findall(r'test[\w_/]*CER[: \']+([0-9.]+)', content)
                if test_cer_matches:
                    test_cer = test_cer_matches[-1]
                else:
                    test_cer_matches = re.findall(r'test.*?CER[: \']+([0-9.]+)', content)
                    if test_cer_matches:
                        test_cer = test_cer_matches[-1]
                
                # If still not found, try general CER pattern
                if test_cer == 'N/A':
                    cer_matches = re.findall(r'CER[: \']+([0-9.]+)', content)
                    if cer_matches:
                        test_cer = cer_matches[-1] + ' (unspecified)'
                        
        except Exception as e:
            print(f'Error extracting metrics from {log_file}: {str(e)}')
        
        session_results[session_count] = test_cer
    
    # Write the tabular summary
    for session_count in sorted(session_results.keys()):
        f.write(f'{session_count:16d} | {session_results[session_count]}\n')

# Also update the simple summary file
with open(os.path.join(results_dir, 'summary.txt'), 'w') as summary:
    summary.write('Number of sessions | Test CER\\n')
    summary.write('----------------- | --------\\n')
    
    for session_count in sorted(session_results.keys()):
        if session_count == 2:
            summary.write(f'2 sessions        | {session_results[2]}\\n')
        elif session_count == 4:
            summary.write(f'4 sessions        | {session_results[4]}\\n')
        elif session_count == 8:
            summary.write(f'8 sessions        | {session_results[8]}\\n')
        elif session_count == 16:
            summary.write(f'16 sessions       | {session_results[16]}\\n')

print(f'Summary created at {summary_file}')
"

echo "Data scaling experiment completed successfully!"
echo "Please check the detailed summary at $RESULTS_DIR/detailed_summary.txt"
echo "A simpler summary is available at $RESULTS_DIR/summary.txt" 