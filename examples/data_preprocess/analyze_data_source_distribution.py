#!/usr/bin/env python3
"""
Analyze data source distribution and position patterns in parquet files.
This script reads parquet files and analyzes how different data sources are distributed
across the dataset, including position patterns and clustering.
"""

import argparse
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataSourceDistributionAnalyzer:
    """Analyzer for data source distribution and position patterns in datasets."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results = {}
    
    def load_parquet_files(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load parquet files from the specified directory.
        
        Args:
            data_dir: Directory containing parquet files
            
        Returns:
            Dict mapping split names to DataFrames
        """
        dataframes = {}
        
        for split in ["train_processed", "test_processed"]:
            parquet_path = os.path.join(data_dir, f"{split}.parquet")
            if os.path.exists(parquet_path):
                logger.info(f"Loading {split} split from {parquet_path}")
                df = pd.read_parquet(parquet_path)
                dataframes[split] = df
                logger.info(f"Loaded {len(df)} rows from {split} split")
            else:
                logger.warning(f"File not found: {parquet_path}")
        
        return dataframes
    
    def analyze_data_source_distribution(self, df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """
        Analyze data source distribution and position patterns.
        
        Args:
            df: DataFrame to analyze
            split_name: Name of the split (train/test)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing data source distribution for {split_name} split...")
        
        # Basic data source counts
        data_source_counts = df['data_source'].value_counts()
        total_samples = len(df)
        
        # Calculate position information for each data source
        position_analysis = {}
        for source in data_source_counts.index:
            source_indices = df[df['data_source'] == source].index.tolist()
            
            position_analysis[source] = {
                'count': len(source_indices),
                'ratio': len(source_indices) / total_samples,
                'indices': source_indices,
                'first_position': min(source_indices) if source_indices else -1,
                'last_position': max(source_indices) if source_indices else -1,
                'position_range': max(source_indices) - min(source_indices) + 1 if source_indices else 0,
                'mean_position': np.mean(source_indices) if source_indices else -1,
                'std_position': np.std(source_indices) if source_indices else 0,
            }
        
        # Analyze clustering patterns
        clustering_analysis = self._analyze_clustering_patterns(df)
        
        # Analyze sequential patterns
        sequential_analysis = self._analyze_sequential_patterns(df)
        
        # Analyze position distribution
        position_distribution = self._analyze_position_distribution(df)
        
        results = {
            'split_name': split_name,
            'total_samples': total_samples,
            'unique_data_sources': len(data_source_counts),
            'data_source_counts': data_source_counts.to_dict(),
            'data_source_ratios': (data_source_counts / total_samples).to_dict(),
            'position_analysis': position_analysis,
            'clustering_analysis': clustering_analysis,
            'sequential_analysis': sequential_analysis,
            'position_distribution': position_distribution,
        }
        
        self.results[split_name] = results
        return results
    
    def _analyze_clustering_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clustering patterns of data sources."""
        clustering_results = {}
        
        for source in df['data_source'].unique():
            source_indices = df[df['data_source'] == source].index.tolist()
            
            if len(source_indices) < 2:
                clustering_results[source] = {
                    'clusters': 0,
                    'avg_cluster_size': 0,
                    'max_cluster_size': 0,
                    'clustering_score': 0.0
                }
                continue
            
            # Find consecutive sequences
            clusters = []
            current_cluster = [source_indices[0]]
            
            for i in range(1, len(source_indices)):
                if source_indices[i] == source_indices[i-1] + 1:
                    current_cluster.append(source_indices[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [source_indices[i]]
            clusters.append(current_cluster)
            
            cluster_sizes = [len(cluster) for cluster in clusters]
            
            clustering_results[source] = {
                'clusters': len(clusters),
                'avg_cluster_size': np.mean(cluster_sizes),
                'max_cluster_size': max(cluster_sizes),
                'min_cluster_size': min(cluster_sizes),
                'clustering_score': len(clusters) / len(source_indices),  # Lower is more clustered
                'cluster_sizes': cluster_sizes
            }
        
        return clustering_results
    
    def _analyze_sequential_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sequential patterns in data source ordering."""
        data_sources = df['data_source'].tolist()
        
        # Count transitions between different sources
        transitions = {}
        for i in range(len(data_sources) - 1):
            current = data_sources[i]
            next_source = data_sources[i + 1]
            transition = f"{current} -> {next_source}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # Find most common patterns
        most_common_transitions = Counter(transitions).most_common(10)
        
        # Calculate entropy of transitions
        total_transitions = sum(transitions.values())
        transition_probs = [count / total_transitions for count in transitions.values()]
        entropy = -sum(p * np.log2(p) for p in transition_probs if p > 0)
        
        return {
            'total_transitions': total_transitions,
            'unique_transitions': len(transitions),
            'most_common_transitions': most_common_transitions,
            'transition_entropy': entropy,
            'all_transitions': transitions
        }
    
    def _analyze_position_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how data sources are distributed across different position ranges."""
        total_samples = len(df)
        num_ranges = 10  # Divide dataset into 10 ranges
        range_size = total_samples // num_ranges
        
        position_ranges = {}
        for i in range(num_ranges):
            start_idx = i * range_size
            end_idx = (i + 1) * range_size if i < num_ranges - 1 else total_samples
            range_name = f"range_{i+1}_{start_idx}-{end_idx-1}"
            
            range_df = df.iloc[start_idx:end_idx]
            range_counts = range_df['data_source'].value_counts()
            position_ranges[range_name] = range_counts.to_dict()
        
        return position_ranges
    
    def print_detailed_analysis(self, results: Dict[str, Any]):
        """Print detailed analysis results."""
        split_name = results['split_name']
        
        print(f"\n{'='*80}")
        print(f"DETAILED DATA SOURCE DISTRIBUTION ANALYSIS - {split_name.upper()}")
        print(f"{'='*80}")
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"  Total samples: {results['total_samples']:,}")
        print(f"  Unique data sources: {results['unique_data_sources']}")
        
        # Data source distribution
        print(f"\n📈 DATA SOURCE DISTRIBUTION:")
        for source, count in results['data_source_counts'].items():
            ratio = results['data_source_ratios'][source]
            print(f"  {source}: {count:,} samples ({ratio:.2%})")
        
        # Position analysis
        print(f"\n📍 POSITION ANALYSIS:")
        for source, analysis in results['position_analysis'].items():
            print(f"\n  {source}:")
            print(f"    Count: {analysis['count']:,}")
            print(f"    Ratio: {analysis['ratio']:.2%}")
            print(f"    First position: {analysis['first_position']:,}")
            print(f"    Last position: {analysis['last_position']:,}")
            print(f"    Position range: {analysis['position_range']:,}")
            print(f"    Mean position: {analysis['mean_position']:.1f}")
            print(f"    Std position: {analysis['std_position']:.1f}")
        
        # Clustering analysis
        print(f"\n🔗 CLUSTERING ANALYSIS:")
        for source, analysis in results['clustering_analysis'].items():
            print(f"\n  {source}:")
            print(f"    Number of clusters: {analysis['clusters']}")
            print(f"    Average cluster size: {analysis['avg_cluster_size']:.1f}")
            print(f"    Max cluster size: {analysis['max_cluster_size']}")
            print(f"    Clustering score: {analysis['clustering_score']:.3f} (lower = more clustered)")
        
        # Sequential patterns
        print(f"\n🔄 SEQUENTIAL PATTERNS:")
        seq_analysis = results['sequential_analysis']
        print(f"  Total transitions: {seq_analysis['total_transitions']:,}")
        print(f"  Unique transitions: {seq_analysis['unique_transitions']}")
        print(f"  Transition entropy: {seq_analysis['transition_entropy']:.3f}")
        print(f"  Most common transitions:")
        for transition, count in seq_analysis['most_common_transitions'][:5]:
            print(f"    {transition}: {count} times")
        
        # Position distribution across ranges
        print(f"\n📊 POSITION DISTRIBUTION ACROSS RANGES:")
        for range_name, range_counts in results['position_distribution'].items():
            if range_counts:
                print(f"  {range_name}:")
                for source, count in sorted(range_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {source}: {count} samples")
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = None):
        """Create visualization plots for the analysis."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        split_name = results['split_name']
        
        # 1. Data source distribution pie chart
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sources = list(results['data_source_counts'].keys())
        counts = list(results['data_source_counts'].values())
        plt.pie(counts, labels=sources, autopct='%1.1f%%', startangle=90)
        plt.title(f'Data Source Distribution - {split_name}')
        
        # 2. Position distribution heatmap
        plt.subplot(2, 2, 2)
        position_data = []
        for source in sources:
            source_indices = results['position_analysis'][source]['indices']
            # Create a binary vector indicating presence at each position
            binary_vector = [1 if i in source_indices else 0 for i in range(results['total_samples'])]
            # Sample every 100th position for visualization
            sampled_vector = binary_vector[::max(1, len(binary_vector)//100)]
            position_data.append(sampled_vector)
        
        if position_data:
            plt.imshow(position_data, aspect='auto', cmap='Blues')
            plt.xlabel('Position (sampled)')
            plt.ylabel('Data Source')
            plt.yticks(range(len(sources)), sources)
            plt.title(f'Position Distribution - {split_name}')
        
        # 3. Clustering scores
        plt.subplot(2, 2, 3)
        clustering_scores = [results['clustering_analysis'][source]['clustering_score'] for source in sources]
        plt.bar(sources, clustering_scores)
        plt.xlabel('Data Source')
        plt.ylabel('Clustering Score (lower = more clustered)')
        plt.title(f'Clustering Analysis - {split_name}')
        plt.xticks(rotation=45)
        
        # 4. Position ranges distribution
        plt.subplot(2, 2, 4)
        range_names = list(results['position_distribution'].keys())
        range_data = []
        for range_name in range_names:
            range_counts = results['position_distribution'][range_name]
            total_in_range = sum(range_counts.values())
            range_data.append(total_in_range)
        
        plt.bar(range(range(len(range_names))), range_data)
        plt.xlabel('Position Range')
        plt.ylabel('Number of Samples')
        plt.title(f'Samples per Position Range - {split_name}')
        plt.xticks(range(len(range_names)), [f'R{i+1}' for i in range(len(range_names))])
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, f'data_source_analysis_{split_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_analysis_results(self, output_dir: str):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, results in self.results.items():
            # Save detailed results as JSON
            import json
            output_path = os.path.join(output_dir, f'data_source_analysis_{split_name}.json')
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Clean results for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if key == 'position_analysis':
                    clean_results[key] = {}
                    for source, analysis in value.items():
                        clean_results[key][source] = {
                            k: convert_numpy(v) for k, v in analysis.items()
                        }
                else:
                    clean_results[key] = convert_numpy(value)
            
            with open(output_path, 'w') as f:
                json.dump(clean_results, f, indent=2)
            
            logger.info(f"Analysis results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze data source distribution and position patterns in parquet files.")
    parser.add_argument(
        "--data_dir",
        default="searchR1_processed_direct",
        help="Directory containing the processed parquet files"
    )
    parser.add_argument(
        "--output_dir",
        default="data_source_analysis_results",
        help="Directory to save analysis results and visualizations"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DataSourceDistributionAnalyzer()
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    dataframes = analyzer.load_parquet_files(args.data_dir)
    
    if not dataframes:
        logger.error("No data files found!")
        return
    
    # Analyze each split
    for split_name, df in dataframes.items():
        logger.info(f"\nAnalyzing {split_name} split...")
        
        # Perform analysis
        results = analyzer.analyze_data_source_distribution(df, split_name)
        
        # Print detailed results
        analyzer.print_detailed_analysis(results)
        
        # Create visualizations if requested
        if args.create_plots:
            analyzer.create_visualizations(results, args.output_dir)
    
    # Save results
    analyzer.save_analysis_results(args.output_dir)
    
    logger.info(f"\nAnalysis completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
