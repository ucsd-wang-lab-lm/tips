#!/usr/bin/env python3
"""
Example usage script for analyze_search_r1_dataset.py

This script demonstrates how to use the SearchR1 dataset analyzer with different configurations.
"""

import subprocess
import sys
import os

def run_analysis_example():
    """Run example analysis with different configurations."""
    
    # Base command
    base_cmd = [
        "python", "analyze_search_r1_dataset.py",
        "--data_dir", "~/data/searchR1_processed_direct",  # Update this path as needed
    ]
    
    print("="*80)
    print("SEARCH R1 DATASET ANALYSIS EXAMPLES")
    print("="*80)
    
    # Example 1: Basic analysis without filtering
    print("\n1. Basic Analysis (No Filtering)")
    print("-" * 40)
    cmd1 = base_cmd + ["--tokenizer_name", "Qwen/Qwen-7B-Chat"]
    print(f"Command: {' '.join(cmd1)}")
    print("This will load the data and show basic statistics without any filtering.")
    
    # Example 2: Filter by token length
    print("\n2. Filter by Token Length (100-2000 tokens)")
    print("-" * 40)
    cmd2 = base_cmd + [
        "--tokenizer_name", "Qwen/Qwen-7B-Chat",
        "--min_tokens", "100",
        "--max_tokens", "2000"
    ]
    print(f"Command: {' '.join(cmd2)}")
    print("This will filter out prompts that are too short (<100 tokens) or too long (>2000 tokens).")
    
    # Example 3: Sample by data source with custom ratios
    print("\n3. Sample by Data Source with Custom Ratios")
    print("-" * 40)
    sampling_ratios = '{"searchR1_source1": 0.5, "searchR1_source2": 0.8, "searchR1_source3": 0.3}'
    cmd3 = base_cmd + [
        "--tokenizer_name", "Qwen/Qwen-7B-Chat",
        "--min_tokens", "50",
        "--max_tokens", "3000",
        "--sampling_ratios", sampling_ratios,
        "--output_dir", "~/data/searchR1_analyzed"
    ]
    print(f"Command: {' '.join(cmd3)}")
    print("This will:")
    print("- Filter prompts between 50-3000 tokens")
    print("- Sample 50% from source1, 80% from source2, 30% from source3")
    print("- Save processed data to ~/data/searchR1_analyzed")
    
    # Example 4: Conservative filtering for high-quality data
    print("\n4. Conservative Filtering for High-Quality Data")
    print("-" * 40)
    cmd4 = base_cmd + [
        "--tokenizer_name", "Qwen/Qwen-7B-Chat",
        "--min_tokens", "200",
        "--max_tokens", "1500",
        "--output_dir", "~/data/searchR1_high_quality"
    ]
    print(f"Command: {' '.join(cmd4)}")
    print("This will keep only medium-length prompts (200-1500 tokens) for high-quality training.")
    
    print("\n" + "="*80)
    print("TO RUN ANY OF THESE EXAMPLES:")
    print("1. Update the --data_dir path to point to your processed data")
    print("2. Update the --sampling_ratios to match your actual data sources")
    print("3. Run the command in your terminal")
    print("="*80)

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# Sample configuration for SearchR1 dataset analysis

# Data paths
DATA_DIR = "~/data/searchR1_processed_direct"
OUTPUT_DIR = "~/data/searchR1_analyzed"

# Tokenizer settings
TOKENIZER_NAME = "Qwen/Qwen-7B-Chat"

# Token length filtering
MIN_TOKENS = 100
MAX_TOKENS = 2000

# Data source sampling ratios (adjust based on your actual data sources)
SAMPLING_RATIOS = {
    "searchR1_hotpotqa": 0.7,      # Sample 70% from HotpotQA
    "searchR1_natural_questions": 0.5,  # Sample 50% from Natural Questions
    "searchR1_triviaqa": 0.8,      # Sample 80% from TriviaQA
    "searchR1_web_questions": 0.6,  # Sample 60% from Web Questions
}

# Example usage:
# python analyze_search_r1_dataset.py \\
#     --data_dir $DATA_DIR \\
#     --tokenizer_name $TOKENIZER_NAME \\
#     --min_tokens $MIN_TOKENS \\
#     --max_tokens $MAX_TOKENS \\
#     --sampling_ratios '{"searchR1_hotpotqa": 0.7, "searchR1_natural_questions": 0.5}' \\
#     --output_dir $OUTPUT_DIR
"""
    
    with open("sample_config.py", "w") as f:
        f.write(config_content)
    
    print("Created sample_config.py with example configuration")

if __name__ == "__main__":
    run_analysis_example()
    print("\n")
    create_sample_config()
    print("\nSample configuration file created: sample_config.py")
