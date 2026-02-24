#!/bin/bash
# Script to run data source distribution analysis

set -x

# Default paths
DATA_DIR=${DATA_DIR:-"searchR1_processed_direct"}
OUTPUT_FILE=${OUTPUT_FILE:-"data_source_analysis_results.json"}

echo "Starting data source distribution analysis..."
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"

# Run the analysis
python3 simple_data_source_analysis.py \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE"

echo "Analysis completed!"
echo "Results saved to: $OUTPUT_FILE"
