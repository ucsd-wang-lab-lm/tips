#!/usr/bin/env python3
"""
Simple data source distribution analyzer for parquet files.
Analyzes how different data sources are distributed across the dataset.
"""

import argparse
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_parquet_files(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load parquet files from the specified directory."""
    dataframes = {}
    
    for split in ["train", "test"]:
        parquet_path = os.path.join(data_dir, f"{split}.parquet")
        if os.path.exists(parquet_path):
            logger.info(f"Loading {split} split from {parquet_path}")
            df = pd.read_parquet(parquet_path)
            dataframes[split] = df
            logger.info(f"Loaded {len(df)} rows from {split} split")
        else:
            logger.warning(f"File not found: {parquet_path}")
    
    return dataframes


def analyze_data_source_positions(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Analyze data source position distribution."""
    logger.info(f"Analyzing data source positions for {split_name} split...")
    
    # Basic counts
    data_source_counts = df['data_source'].value_counts()
    total_samples = len(df)
    
    print(f"\n{'='*80}")
    print(f"DATA SOURCE POSITION ANALYSIS - {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples:,}")
    print(f"Unique data sources: {len(data_source_counts)}")
    
    # Analyze each data source
    print(f"\n📊 DETAILED POSITION ANALYSIS:")
    print(f"{'Source':<30} {'Count':<10} {'Ratio':<8} {'First':<8} {'Last':<10} {'Range':<10} {'Mean':<8} {'Std':<8}")
    print(f"{'-'*100}")
    
    position_stats = {}
    for source in data_source_counts.index:
        source_indices = df[df['data_source'] == source].index.tolist()
        
        if not source_indices:
            continue
            
        count = len(source_indices)
        ratio = count / total_samples
        first_pos = min(source_indices)
        last_pos = max(source_indices)
        pos_range = last_pos - first_pos + 1
        mean_pos = np.mean(source_indices)
        std_pos = np.std(source_indices)
        
        print(f"{source:<30} {count:<10,} {ratio:<8.2%} {first_pos:<8,} {last_pos:<10,} {pos_range:<10,} {mean_pos:<8.1f} {std_pos:<8.1f}")
        
        position_stats[source] = {
            'count': count,
            'ratio': ratio,
            'first_position': first_pos,
            'last_position': last_pos,
            'position_range': pos_range,
            'mean_position': mean_pos,
            'std_position': std_pos,
            'indices': source_indices
        }
    
    # Analyze clustering patterns
    print(f"\n🔗 CLUSTERING ANALYSIS:")
    print(f"{'Source':<30} {'Clusters':<10} {'Avg Size':<10} {'Max Size':<10} {'Score':<8}")
    print(f"{'-'*80}")
    
    for source, stats in position_stats.items():
        indices = stats['indices']
        if len(indices) < 2:
            print(f"{source:<30} {'0':<10} {'0':<10} {'0':<10} {'0.0':<8}")
            continue
        
        # Find consecutive sequences (clusters)
        clusters = []
        current_cluster = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_cluster.append(indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [indices[i]]
        clusters.append(current_cluster)
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        avg_cluster_size = np.mean(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        clustering_score = len(clusters) / len(indices)  # Lower = more clustered
        
        print(f"{source:<30} {len(clusters):<10} {avg_cluster_size:<10.1f} {max_cluster_size:<10} {clustering_score:<8.3f}")
    
    # Analyze sequential patterns
    print(f"\n🔄 SEQUENTIAL PATTERNS:")
    data_sources = df['data_source'].tolist()
    
    # Count transitions
    transitions = {}
    for i in range(len(data_sources) - 1):
        current = data_sources[i]
        next_source = data_sources[i + 1]
        transition = f"{current} -> {next_source}"
        transitions[transition] = transitions.get(transition, 0) + 1
    
    # Show most common transitions
    most_common = Counter(transitions).most_common(10)
    print(f"Total transitions: {len(transitions):,}")
    print(f"Most common transitions:")
    for transition, count in most_common:
        print(f"  {transition}: {count} times")
    
    # Analyze position distribution across ranges
    print(f"\n📊 POSITION DISTRIBUTION ACROSS RANGES:")
    num_ranges = 10
    range_size = total_samples // num_ranges
    
    print(f"{'Range':<15} {'Start':<8} {'End':<10} {'Samples':<10} {'Top Sources'}")
    print(f"{'-'*80}")
    
    for i in range(num_ranges):
        start_idx = i * range_size
        end_idx = (i + 1) * range_size if i < num_ranges - 1 else total_samples
        range_name = f"Range {i+1}"
        
        range_df = df.iloc[start_idx:end_idx]
        range_counts = range_df['data_source'].value_counts()
        
        # Get top 3 sources in this range
        top_sources = range_counts.head(3)
        top_sources_str = ", ".join([f"{src}({count})" for src, count in top_sources.items()])
        
        print(f"{range_name:<15} {start_idx:<8,} {end_idx-1:<10,} {len(range_df):<10,} {top_sources_str}")
    
    return {
        'split_name': split_name,
        'total_samples': total_samples,
        'data_source_counts': data_source_counts.to_dict(),
        'position_stats': position_stats,
        'transitions': transitions
    }


def analyze_data_source_overlap(dataframes: Dict[str, pd.DataFrame]):
    """Analyze overlap between train and test splits."""
    if 'train' not in dataframes or 'test' not in dataframes:
        logger.warning("Cannot analyze overlap: missing train or test split")
        return
    
    train_sources = set(dataframes['train']['data_source'].unique())
    test_sources = set(dataframes['test']['data_source'].unique())
    
    print(f"\n🔄 TRAIN-TEST DATA SOURCE OVERLAP:")
    print(f"Train sources: {len(train_sources)}")
    print(f"Test sources: {len(test_sources)}")
    print(f"Common sources: {len(train_sources & test_sources)}")
    print(f"Train-only sources: {len(train_sources - test_sources)}")
    print(f"Test-only sources: {len(test_sources - train_sources)}")
    
    if train_sources & test_sources:
        print(f"Common sources: {sorted(train_sources & test_sources)}")
    if train_sources - test_sources:
        print(f"Train-only sources: {sorted(train_sources - test_sources)}")
    if test_sources - train_sources:
        print(f"Test-only sources: {sorted(test_sources - train_sources)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze data source distribution and position patterns.")
    parser.add_argument(
        "--data_dir",
        default="searchR1_processed_direct",
        help="Directory containing the processed parquet files"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional file to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    dataframes = load_parquet_files(args.data_dir)
    
    if not dataframes:
        logger.error("No data files found!")
        return
    
    # Analyze each split
    all_results = {}
    for split_name, df in dataframes.items():
        logger.info(f"\nAnalyzing {split_name} split...")
        results = analyze_data_source_positions(df, split_name)
        all_results[split_name] = results
    
    # Analyze overlap between splits
    analyze_data_source_overlap(dataframes)
    
    # Save results if requested
    if args.output_file:
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results
        clean_results = {}
        for split_name, results in all_results.items():
            clean_results[split_name] = {}
            for key, value in results.items():
                if key == 'position_stats':
                    clean_results[split_name][key] = {}
                    for source, stats in value.items():
                        clean_results[split_name][key][source] = {
                            k: convert_numpy(v) for k, v in stats.items() if k != 'indices'
                        }
                else:
                    clean_results[split_name][key] = convert_numpy(value)
        
        with open(args.output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Analysis results saved to: {args.output_file}")
    
    logger.info("\nAnalysis completed!")


if __name__ == "__main__":
    main()
