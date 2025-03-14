#!/bin/bash

# Cross-experiment script to test different combinations of models and data processing techniques
# Author: Claude AI Assistant
# Date: $(date +"%Y-%m-%d")

# Setup common parameters
USER="single_user"
MAX_EPOCHS=50
ACCELERATOR="gpu"
DEVICES=1
BATCH_SIZE=32
RESULTS_DIR="cross_experiment_results_CNN"

# delete directory if it exists
rm -rf ${RESULTS_DIR}

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Log file to record experiment status
LOG_FILE="${RESULTS_DIR}/experiment_log.txt"
echo "EMG2QWERTY Cross-Experiment Log - $(date)" > ${LOG_FILE}
echo "=======================================" >> ${LOG_FILE}

# Model configurations
MODELS=(
    "tds_conv_ctc"       # CNN 
    # "lstm_ctc"           # RNN
    # "rnn_cnn"            # CNN+RNN
    # "transformer_ctc"    # Transformer
    # "conformer_ctc"      # Conformer
)


# Data processing techniques
TRANSFORMS=(
    # "log_spectrogram"      # baseline
    "temporal_scaling"     # Temporal Scaling
    "bandpass_filter"      # Bandpass Filter
    "cross_channel_mixing" # Cross-channel mixing
)


# Count total experiments
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#TRANSFORMS[@]}))
echo "Total experiments to run: ${TOTAL_EXPERIMENTS}" | tee -a ${LOG_FILE}
echo "" >> ${LOG_FILE}

# Run all combinations
EXPERIMENT_COUNT=0

for model in "${MODELS[@]}"; do
    for transform in "${TRANSFORMS[@]}"; do
        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        
        # Create experiment name
        EXPERIMENT_NAME="${model}_${transform}"
        EXPERIMENT_DIR="${RESULTS_DIR}/${EXPERIMENT_NAME}"
        mkdir -p ${EXPERIMENT_DIR}
        
        echo "=====================================================" | tee -a ${LOG_FILE}
        echo "Running experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}: ${EXPERIMENT_NAME}" | tee -a ${LOG_FILE}
        echo "Start time: $(date)" | tee -a ${LOG_FILE}
        
        # Set model-specific parameters
        EXTRA_PARAMS=""
        if [[ "$model" == "conformer_ctc" || "$model" == "transformer_ctc" ]]; then
            EXTRA_PARAMS="optimizer=adamw_transformer lr_scheduler=transformer_warmup +trainer.gradient_clip_val=1.0"
            # Use smaller batch size for transformer-based models
            BATCH_SIZE=32
        else
            BATCH_SIZE=32
        fi
        
        # Run the training command
        CMD="python -m emg2qwerty.train \
          user=\"${USER}\" \
          model=${model} \
          transforms=${transform} \
          trainer.max_epochs=${MAX_EPOCHS} \
          trainer.accelerator=${ACCELERATOR} \
          trainer.devices=${DEVICES} \
          batch_size=${BATCH_SIZE} \
          +callback=decoding_visualization \
          ${EXTRA_PARAMS} \
          +experiment.name=${EXPERIMENT_NAME} \
          +experiment.output_dir=${EXPERIMENT_DIR}"
          
        echo "Command: ${CMD}" >> ${LOG_FILE}
        echo "Running experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}: ${model} with ${transform}"
        
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
        echo "" >> ${LOG_FILE}
        
        # Write experiment summary
        echo "${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS} - ${EXPERIMENT_NAME}: ${STATUS}" >> "${RESULTS_DIR}/summary.txt"
        
        # Add a small delay to prevent resource contention
        sleep 5
    done
done

echo "=====================================================" | tee -a ${LOG_FILE}
echo "All experiments completed!" | tee -a ${LOG_FILE}
echo "Results are stored in ${RESULTS_DIR}/" | tee -a ${LOG_FILE}
echo "Check ${RESULTS_DIR}/summary.txt for a quick overview of all experiments." | tee -a ${LOG_FILE}

# Create a summary of all experiments
echo "Creating summary report..."
python -c "
import os
import glob
import re

results_dir = '${RESULTS_DIR}'
summary_file = os.path.join(results_dir, 'detailed_summary.txt')

with open(summary_file, 'w') as f:
    f.write('EMG2QWERTY Cross-Experiment Detailed Summary\n')
    f.write('===========================================\n\n')
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(results_dir, '*_*'))
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        log_file = os.path.join(exp_dir, 'training.log')
        
        if not os.path.exists(log_file):
            continue
            
        # Extract model and transform
        model, transform = exp_name.split('_', 1)
        
        # Initialize metrics
        final_accuracy = 'N/A'
        final_wer = 'N/A'
        val_cer = 'N/A'
        test_cer = 'N/A'
        
        with open(log_file, 'r') as log:
            content = log.read()
            
            # Look for the last occurrence of accuracy metrics
            acc_matches = re.findall(r'accuracy[: ]+([0-9.]+)', content)
            if acc_matches:
                final_accuracy = acc_matches[-1]
                
            wer_matches = re.findall(r'WER[: ]+([0-9.]+)', content)
            if wer_matches:
                final_wer = wer_matches[-1]
            
            # Look for validation CER
            val_cer_matches = re.findall(r'val[\w_]*CER[: ]+([0-9.]+)', content)
            if val_cer_matches:
                val_cer = val_cer_matches[-1]
            else:
                # Alternative pattern for validation CER
                val_cer_matches = re.findall(r'validation.*?CER[: ]+([0-9.]+)', content)
                if val_cer_matches:
                    val_cer = val_cer_matches[-1]
            
            # Look for test CER
            test_cer_matches = re.findall(r'test[\w_]*CER[: ]+([0-9.]+)', content)
            if test_cer_matches:
                test_cer = test_cer_matches[-1]
            else:
                # Alternative pattern for test CER
                test_cer_matches = re.findall(r'test.*?CER[: ]+([0-9.]+)', content)
                if test_cer_matches:
                    test_cer = test_cer_matches[-1]
            
            # If we still haven't found specific validation/test CER metrics,
            # let's check for the general CER and note it's not specified
            if val_cer == 'N/A' and test_cer == 'N/A':
                cer_matches = re.findall(r'CER[: ]+([0-9.]+)', content)
                if cer_matches:
                    val_cer = cer_matches[-1] + ' (unspecified)'
        
        # Write experiment summary
        f.write(f'Experiment: {exp_name}\n')
        f.write(f'  Model: {model}\n')
        f.write(f'  Transform: {transform}\n')
        f.write(f'  Final Accuracy: {final_accuracy}\n')
        f.write(f'  Final WER: {final_wer}\n')
        f.write(f'  Validation CER: {val_cer}\n')
        f.write(f'  Test CER: {test_cer}\n')
        f.write('\n')

print(f'Summary created at {summary_file}')
"

echo "Cross-experiment completed successfully!" 