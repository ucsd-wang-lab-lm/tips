#!/bin/bash

# Script to run target answers analysis
# Usage: ./run_target_analysis.sh [data_dir] [output_dir]

DATA_DIR=${1:-"data_preprocess/searchR1_processed"}
OUTPUT_DIR=${2:-"target_answers_analysis_results"}

echo "Starting target answers analysis..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the analysis with visualizations
python examples/data_preprocess/analyze_target_answers.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --create_plots

echo "Analysis completed!"
echo "Results saved to: $OUTPUT_DIR"
