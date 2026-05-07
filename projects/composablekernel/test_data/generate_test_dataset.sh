#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Generate Comprehensive Convolution Test Dataset for CK
# This script captures MIOpen commands from PyTorch models and generates test cases

set -e  # Exit on error
set +x  # Disable command echo (even if called with bash -x)

# Trap to kill all background jobs on script exit/interruption
cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    # Kill all jobs in the current process group
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete."
    exit 1
}

# Set up trap for common termination signals
trap cleanup SIGINT SIGTERM EXIT

echo "=========================================="
echo "CK Convolution Test Dataset Generator"
echo "=========================================="

# Check if PyTorch is installed, if not create a virtual environment
echo "Checking for PyTorch installation..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Creating virtual environment..."
    
    # Create a virtual environment in the current directory
    VENV_DIR="./.venv"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv $VENV_DIR || {
            echo "ERROR: Failed to create virtual environment."
            echo "Creating empty CSV files as fallback..."
            echo "# 2D Convolution Test Cases" > conv_test_set_2d_dataset.csv
            echo "# Combined from multiple models" >> conv_test_set_2d_dataset.csv
            echo "# 3D Convolution Test Cases" > conv_test_set_3d_dataset.csv
            echo "# Combined from multiple models" >> conv_test_set_3d_dataset.csv
            exit 1
        }
    fi
    
    # Activate virtual environment
    source $VENV_DIR/bin/activate
    
    # Install PyTorch in virtual environment with ROCm support
    echo "Installing PyTorch and torchvision with ROCm support in virtual environment..."
    # Since we're in a ROCm 6.4.1 environment, we need compatible PyTorch
    # PyTorch doesn't have 6.4 wheels yet, so we use 6.2 which should be compatible
    echo "Installing PyTorch with ROCm 6.2 support (compatible with ROCm 6.4)..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.2 || {
        echo "ERROR: Failed to install PyTorch with ROCm support."
        echo "Creating empty CSV files as fallback..."
        echo "# 2D Convolution Test Cases" > conv_test_set_2d_dataset.csv
        echo "# Combined from multiple models" >> conv_test_set_2d_dataset.csv
        echo "# 3D Convolution Test Cases" > conv_test_set_3d_dataset.csv
        echo "# Combined from multiple models" >> conv_test_set_3d_dataset.csv
        exit 1
    }
    echo "PyTorch installed successfully in virtual environment!"
    
    # Use the virtual environment's Python for the rest of the script
    export PYTHON_CMD="$VENV_DIR/bin/python3"
else
    echo "PyTorch is already installed."
    export PYTHON_CMD="python3"
fi

# Verify PyTorch installation and GPU support
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')"
$PYTHON_CMD -c "import torch; print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"
if ! $PYTHON_CMD -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: PyTorch installed but GPU support not available!"
    echo "MIOpen commands will not be generated without GPU support."
    echo "Continuing anyway to generate placeholder data..."
fi

# Parse command line arguments
CONFIG_MODE="full"  # Default configuration mode: 'small', 'half' or 'full'
MAX_PARALLEL_JOBS=1  # Default number of parallel jobs
NUM_GPUS=1  # Number of GPUs to use (0 means no GPU assignment)

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -j*)
            MAX_PARALLEL_JOBS="${1#-j}"
            shift
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        small|half|full)
            CONFIG_MODE="$1"
            shift
            ;;
        *)
            echo "Usage: $0 [small|half|full] [-j <num_jobs>] [--gpus <num_gpus>]"
            echo "  Configuration modes: small, half, full (default: full)"
            echo "  -j <num_jobs>: Number of parallel jobs (default: 1)"
            echo "  --gpus <num_gpus>: Number of GPUs to use (e.g., 8 for GPUs 0-7)"
            exit 1
            ;;
    esac
done

# Setup GPU array if GPUs are requested
if [ $NUM_GPUS -gt 0 ]; then
    # Auto-detect available GPUs
    AVAILABLE_GPUS_COUNT=$(rocm-smi --showid 2>/dev/null | grep -oP 'GPU\[\K[0-9]+' | wc -l)
    if [ "$AVAILABLE_GPUS_COUNT" -gt 0 ]; then
        MAX_AVAILABLE=$AVAILABLE_GPUS_COUNT
    else
        MAX_AVAILABLE=0
    fi
    
    # Validate requested GPU count
    if [ $NUM_GPUS -gt $MAX_AVAILABLE ]; then
        echo "WARNING: Requested $NUM_GPUS GPUs but only $MAX_AVAILABLE available. Using $MAX_AVAILABLE GPUs."
        NUM_GPUS=$MAX_AVAILABLE
    fi
    
    # Build GPU array (0 to NUM_GPUS-1)
    GPU_ARRAY=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_ARRAY+=($i)
    done
    
    echo "Using $NUM_GPUS GPU(s): ${GPU_ARRAY[*]}"
else
    echo "No GPU assignment specified, using default GPU behavior"
    GPU_ARRAY=()
fi

# Configuration
OUTPUT_DIR="generated_datasets"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

echo ""
echo "Step 1: Generating model configurations"
echo "-----------------------------------------"

# Generate model configuration files (with limit for testing)
echo "Generating model configuration files (mode: $CONFIG_MODE)..."
$PYTHON_CMD generate_model_configs.py \
    --mode $CONFIG_MODE \
    --output-2d $OUTPUT_DIR/model_configs_2d.csv \
    --output-3d $OUTPUT_DIR/model_configs_3d.csv 

if [ ! -f "$OUTPUT_DIR/model_configs_2d.csv" ] || [ ! -f "$OUTPUT_DIR/model_configs_3d.csv" ]; then
    echo "ERROR: Failed to generate configuration files"
    exit 1
fi


# Check if running on GPU
if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: ROCm not detected. Cannot generate MIOpen commands without GPU."
    echo "This script requires an AMD GPU with ROCm installed."
    echo "Creating empty CSV files as placeholder..."
    echo "# 2D Convolution Test Cases (No GPU available)" > conv_test_set_2d_dataset.csv
    echo "# 3D Convolution Test Cases (No GPU available)" > conv_test_set_3d_dataset.csv
    exit 1
fi

# Check if GPU is actually accessible
if ! rocm-smi &> /dev/null; then
    echo "ERROR: rocm-smi failed. GPU may not be accessible."
    echo "Creating empty CSV files as placeholder..."
    echo "# 2D Convolution Test Cases (GPU not accessible)" > conv_test_set_2d_dataset.csv
    echo "# 3D Convolution Test Cases (GPU not accessible)" > conv_test_set_3d_dataset.csv
    exit 1
fi

echo "GPU detected. ROCm version:"
rocm-smi --showdriverversion || true


echo ""
echo "Step 2: Running 2D/3D models and capturing MIOpen commands"
echo "-----------------------------------------"
echo "Using up to $MAX_PARALLEL_JOBS parallel jobs"
echo ""

# Process 2D models from CSV configuration file
echo "Processing 2D models from $OUTPUT_DIR/model_configs_2d.csv..."

# Count total configurations (excluding comments and header)
TOTAL_CONFIGS=$(grep -v "^#" $OUTPUT_DIR/model_configs_2d.csv | tail -n +2 | wc -l)
CURRENT_CONFIG=0

echo "Total configurations to process: $TOTAL_CONFIGS"
echo ""

# Array to track background job PIDs
declare -a job_pids=()
# Counter for round-robin GPU assignment
GPU_COUNTER=0

# Read 2D configurations from CSV (skip comments and header)
while IFS=',' read -r config_name model batch_size channels height width precision; do
    # Skip comments and empty lines
    [[ "$config_name" =~ ^#.*$ ]] && continue
    [[ "$config_name" == "config_name" ]] && continue  # Skip header
    [[ -z "$config_name" ]] && continue
    
    # Increment counter
    CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
    
    # Build configuration command
    CONFIG="--model $model --batch-size $batch_size --channels $channels --height $height --width $width --precision $precision"
    CONFIG_NAME="$config_name"
    
    # Assign GPU in round-robin fashion if GPUs are specified
    if [ $NUM_GPUS -gt 0 ]; then
        GPU_ID=${GPU_ARRAY[$((GPU_COUNTER % NUM_GPUS))]}
        GPU_COUNTER=$((GPU_COUNTER + 1))
        echo -e "${GREEN}[${CURRENT_CONFIG}/${TOTAL_CONFIGS}]${NC} ${CYAN}2D${NC} ${YELLOW}$CONFIG_NAME${NC} ${PURPLE}[GPU ${GPU_ID}]${NC} - Starting in background"
    else
        GPU_ID=""
        echo -e "${GREEN}[${CURRENT_CONFIG}/${TOTAL_CONFIGS}]${NC} ${CYAN}2D${NC} ${YELLOW}$CONFIG_NAME${NC} - Starting in background"
    fi
    
    # Run in background
    (
        # Set HIP_VISIBLE_DEVICES if GPU was assigned
        if [ -n "$GPU_ID" ]; then
            export HIP_VISIBLE_DEVICES=$GPU_ID
        fi
        
        MIOPEN_ENABLE_LOGGING_CMD=1 $PYTHON_CMD run_model_with_miopen.py \
            --model $model --batch-size $batch_size --channels $channels --height $height --width $width --precision $precision \
            > /dev/null 2>> $OUTPUT_DIR/${model}_miopen_log_2d.txt || true
        echo -e "${GREEN}[DONE]${NC} ${CYAN}2D${NC} ${YELLOW}$CONFIG_NAME${NC}"
    ) &
    
    job_pids+=($!)
    
    # Limit number of parallel jobs
    if [ ${#job_pids[@]} -ge $MAX_PARALLEL_JOBS ]; then
        # Wait for any job to complete
        wait -n
        # Remove completed jobs from array
        for i in "${!job_pids[@]}"; do
            if ! kill -0 "${job_pids[$i]}" 2>/dev/null; then
                unset 'job_pids[$i]'
            fi
        done
        job_pids=("${job_pids[@]}")  # Re-index array
    fi

done < $OUTPUT_DIR/model_configs_2d.csv

# Wait for all remaining 2D jobs to complete
echo "Waiting for remaining 2D jobs to complete..."
wait

echo "All 2D models processed!"
echo ""

# Process 3D models from CSV configuration file
echo "Processing 3D models from $OUTPUT_DIR/model_configs_3d.csv..."

# Count total 3D configurations (excluding comments and header)
TOTAL_3D_CONFIGS=$(grep -v "^#" $OUTPUT_DIR/model_configs_3d.csv | tail -n +2 | wc -l)
CURRENT_3D_CONFIG=0

echo "Total 3D configurations to process: $TOTAL_3D_CONFIGS"
echo ""

# Reset job tracking array
declare -a job_pids=()
# GPU counter continues from 2D models for round-robin assignment

# Read 3D configurations from CSV (skip comments and header)
while IFS=',' read -r config_name model batch_size channels temporal_size height width precision; do
    # Skip comments and empty lines  
    [[ "$config_name" =~ ^#.*$ ]] && continue
    [[ "$config_name" == "config_name" ]] && continue  # Skip header
    [[ -z "$config_name" ]] && continue
    
    # Increment counter
    CURRENT_3D_CONFIG=$((CURRENT_3D_CONFIG + 1))
    
    # Build configuration command for 3D models
    CONFIG="--model $model --batch-size $batch_size --channels $channels --temporal-size $temporal_size --height $height --width $width --precision $precision"
    CONFIG_NAME="$config_name"
    
    # Assign GPU in round-robin fashion if GPUs are specified
    if [ $NUM_GPUS -gt 0 ]; then
        GPU_ID=${GPU_ARRAY[$((GPU_COUNTER % NUM_GPUS))]}
        GPU_COUNTER=$((GPU_COUNTER + 1))
        echo -e "${GREEN}[${CURRENT_3D_CONFIG}/${TOTAL_3D_CONFIGS}]${NC} ${CYAN}3D${NC} ${YELLOW}$CONFIG_NAME${NC} ${PURPLE}[GPU ${GPU_ID}]${NC} - Starting in background"
    else
        GPU_ID=""
        echo -e "${GREEN}[${CURRENT_3D_CONFIG}/${TOTAL_3D_CONFIGS}]${NC} ${CYAN}3D${NC} ${YELLOW}$CONFIG_NAME${NC} - Starting in background"
    fi
    
    # Run in background
    (
        # Set HIP_VISIBLE_DEVICES if GPU was assigned
        if [ -n "$GPU_ID" ]; then
            export HIP_VISIBLE_DEVICES=$GPU_ID
        fi
        
        MIOPEN_ENABLE_LOGGING_CMD=1 $PYTHON_CMD run_model_with_miopen.py \
            --model $model --batch-size $batch_size --channels $channels --temporal-size $temporal_size --height $height --width $width --precision $precision \
            > /dev/null 2>> $OUTPUT_DIR/${model}_miopen_log_3d.txt || true
        echo -e "${GREEN}[DONE]${NC} ${CYAN}3D${NC} ${YELLOW}$CONFIG_NAME${NC}"
    ) &
    
    job_pids+=($!)
    
    # Limit number of parallel jobs
    if [ ${#job_pids[@]} -ge $MAX_PARALLEL_JOBS ]; then
        # Wait for any job to complete
        wait -n
        # Remove completed jobs from array
        for i in "${!job_pids[@]}"; do
            if ! kill -0 "${job_pids[$i]}" 2>/dev/null; then
                unset 'job_pids[$i]'
            fi
        done
        job_pids=("${job_pids[@]}")  # Re-index array
    fi

done < $OUTPUT_DIR/model_configs_3d.csv

# Wait for all remaining 3D jobs to complete
echo "Waiting for remaining 3D jobs to complete..."
wait

echo "All 3D models processed!"
echo ""

# Disable trap on successful completion
trap - SIGINT SIGTERM EXIT

echo ""
echo "Step 3: Converting MIOpen commands to CSV test cases"
echo "-----------------------------------------"

# Convert 2D MIOpen logs to CSV
echo "Converting 2D MIOpen logs to CSV..."
for log_file in $OUTPUT_DIR/*_miopen_log_2d.txt; do
    if [ -f "$log_file" ]; then
        # Extract model name from filename (e.g., resnet_miopen_log_2d.txt -> resnet)
        base_name=$(basename "$log_file" _miopen_log_2d.txt)
        output_csv="$OUTPUT_DIR/${base_name}_cases_2d.csv"
        
        echo "  Converting $log_file -> $output_csv"
        $PYTHON_CMD miopen_to_csv.py \
            --input "$log_file" \
            --output-2d "$output_csv" \
            --model-name "$base_name" \
            --filter-duplicates || true
    fi
done

# Convert 3D MIOpen logs to CSV
echo "Converting 3D MIOpen logs to CSV..."
for log_file in $OUTPUT_DIR/*_miopen_log_3d.txt; do
    if [ -f "$log_file" ]; then
        # Extract model name from filename (e.g., resnet3d_18_miopen_log_3d.txt -> resnet3d_18)
        base_name=$(basename "$log_file" _miopen_log_3d.txt)
        output_csv="$OUTPUT_DIR/${base_name}_cases_3d.csv"
        
        echo "  Converting $log_file -> $output_csv"
        $PYTHON_CMD miopen_to_csv.py \
            --input "$log_file" \
            --output-3d "$output_csv" \
            --model-name "$base_name" \
            --filter-duplicates || true
    fi
done

echo ""
echo "Step 4: Combining CSV files into final datasets"
echo "-----------------------------------------"

# Combine all 2D CSV files into one
echo "Combining all 2D test cases..."
# First create empty file with comment headers
echo "# 2D Convolution Test Cases" > conv_test_set_2d_dataset.csv
echo "# Combined from multiple models" >> conv_test_set_2d_dataset.csv
# Add header from first file as a comment
first_2d_file=$(ls $OUTPUT_DIR/*_cases_2d.csv 2>/dev/null | head -1)
if [ -f "$first_2d_file" ]; then
    # Get the CSV header line and prefix with #
    header_line=$(grep "^NDim," "$first_2d_file" | head -1)
    if [ ! -z "$header_line" ]; then
        echo "# $header_line" >> conv_test_set_2d_dataset.csv
    fi
fi
# Append all data rows (skip comment lines and CSV header) from all files
for csv_file in $OUTPUT_DIR/*_cases_2d.csv; do
    if [ -f "$csv_file" ]; then
        # Skip lines starting with # and the NDim header line
        grep -v "^#" "$csv_file" | grep -v "^NDim," >> conv_test_set_2d_dataset.csv 2>/dev/null || true
    fi
done

# Combine all 3D CSV files into one
echo "Combining all 3D test cases..."
# First create empty file with comment headers
echo "# 3D Convolution Test Cases" > conv_test_set_3d_dataset.csv
echo "# Combined from multiple models" >> conv_test_set_3d_dataset.csv
# Add header from first file as a comment
first_3d_file=$(ls $OUTPUT_DIR/*_cases_3d.csv 2>/dev/null | head -1)
if [ -f "$first_3d_file" ]; then
    # Get the CSV header line and prefix with #
    header_line=$(grep "^NDim," "$first_3d_file" | head -1)
    if [ ! -z "$header_line" ]; then
        echo "# $header_line" >> conv_test_set_3d_dataset.csv
    fi
fi
# Append all data rows (skip comment lines and CSV header) from all files
for csv_file in $OUTPUT_DIR/*_cases_3d.csv; do
    if [ -f "$csv_file" ]; then
        # Skip lines starting with # and the NDim header line
        grep -v "^#" "$csv_file" | grep -v "^NDim," >> conv_test_set_3d_dataset.csv 2>/dev/null || true
    fi
done

# Count test cases
COUNT_2D=0
COUNT_3D=0
if [ -f "conv_test_set_2d_dataset.csv" ]; then
    COUNT_2D=$(grep -v "^#" conv_test_set_2d_dataset.csv | tail -n +2 | wc -l)
fi
if [ -f "conv_test_set_3d_dataset.csv" ]; then
    COUNT_3D=$(grep -v "^#" conv_test_set_3d_dataset.csv | tail -n +2 | wc -l)
fi

echo ""
echo "=========================================="
echo "Dataset Generation Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
if [ $COUNT_2D -gt 0 ]; then
    echo "  - conv_test_set_2d_dataset.csv: $COUNT_2D test cases"
fi
if [ $COUNT_3D -gt 0 ]; then
    echo "  - conv_test_set_3d_dataset.csv: $COUNT_3D test cases"
fi
echo "  - Intermediate files in: $OUTPUT_DIR/"
echo ""
echo "To use these datasets for direction (bwd_data, bwd_weight, or fwd):"
echo "  1. Build the test: cd ../script && make -j64 test_grouped_convnd_<direction>_dataset_xdl"
echo "  2. Run the test: ./bin/test_grouped_convnd_<direction>_dataset_xdl"
echo ""
