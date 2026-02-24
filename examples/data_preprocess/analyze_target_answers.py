#!/usr/bin/env python3
"""
Analyze target answers distribution in parquet files.
This script reads parquet files and analyzes the target answers in the 
["reward_model"]["ground_truth"]["target"] field, counting answers based on type.
"""

import argparse
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TargetAnswersAnalyzer:
    """Analyzer for target answers distribution in datasets."""
    
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
    
    def is_list_like_string(self, text: str) -> bool:
        """
        Check if a string represents a list-like structure.
        
        Args:
            text: String to check
            
        Returns:
            True if the string looks like a list representation
        """
        if not isinstance(text, str) or not text.strip():
            return False
        
        text = text.strip()
        
        # Check for common list patterns
        list_patterns = [
            text.startswith('[') and text.endswith(']'),  # [item1, item2, ...]
            text.startswith('(') and text.endswith(')'),  # (item1, item2, ...)
            text.startswith('{') and text.endswith('}'),  # {item1, item2, ...}
            ',' in text and not text.startswith('"') and not text.startswith("'"),  # comma-separated
            '|' in text,  # pipe-separated
            '\n' in text and len(text.split('\n')) > 1,  # newline-separated
        ]
        
        return any(list_patterns)
    
    def parse_list_like_string(self, text: str) -> List[str]:
        """
        Parse a string that might represent a list into actual list items.
        
        Args:
            text: String that might represent a list
            
        Returns:
            List of parsed items
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        text = text.strip()
        
        # Try different parsing methods
        try:
            # Method 1: Try to evaluate as Python literal
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except:
            pass
        
        try:
            # Method 2: JSON parsing
            import json
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except:
            pass
        
        # Method 3: Split by common delimiters
        if text.startswith('[') and text.endswith(']'):
            # Remove brackets and split by comma
            inner = text[1:-1].strip()
            if inner:
                items = [item.strip().strip('"\'') for item in inner.split(',')]
                return [item for item in items if item]
        
        # Method 4: Split by comma
        if ',' in text:
            items = [item.strip().strip('"\'') for item in text.split(',')]
            return [item for item in items if item]
        
        # Method 5: Split by pipe
        if '|' in text:
            items = [item.strip() for item in text.split('|')]
            return [item for item in items if item]
        
        # Method 6: Split by newline
        if '\n' in text:
            items = [item.strip() for item in text.split('\n')]
            return [item for item in items if item]
        
        # If no pattern matches, return as single item
        return [text] if text else []
    
    def _get_detection_method(self, text: str) -> str:
        """
        Determine which method was used to detect a list-like string.
        
        Args:
            text: String that was detected as list-like
            
        Returns:
            String describing the detection method
        """
        if not isinstance(text, str) or not text.strip():
            return "none"
        
        text = text.strip()
        
        if text.startswith('[') and text.endswith(']'):
            return "bracket_notation"
        elif text.startswith('(') and text.endswith(')'):
            return "parenthesis_notation"
        elif text.startswith('{') and text.endswith('}'):
            return "brace_notation"
        elif ',' in text and not text.startswith('"') and not text.startswith("'"):
            return "comma_separated"
        elif '|' in text:
            return "pipe_separated"
        elif '\n' in text and len(text.split('\n')) > 1:
            return "newline_separated"
        else:
            return "unknown"

    def count_answers(self, target_value: Any) -> int:
        """
        Count the number of answers in a target value.
        
        Args:
            target_value: The target value to count answers for
            
        Returns:
            Number of answers (1 for str, len() for list, 0 for None/empty)
            
        Raises:
            ValueError: If target_value is an unsupported type
        """
        if target_value is None:
            return 0
        elif isinstance(target_value, str):
            # Check if string represents a list
            if self.is_list_like_string(target_value):
                parsed_items = self.parse_list_like_string(target_value)
                return len(parsed_items)
            else:
                return 1 if target_value.strip() else 0
        elif isinstance(target_value, list):
            return len(target_value)
        elif isinstance(target_value, (int, float)):
            return 1
        elif isinstance(target_value, np.ndarray):
            # Convert numpy array to list and count
            try:
                array_list = target_value.tolist()
                return len(array_list)
            except Exception as e:
                raise ValueError(f"Failed to convert numpy array to list: {e}")
        else:
            # Raise error for unsupported types
            raise ValueError(f"Unsupported target value type: {type(target_value).__name__}. "
                           f"Supported types: str, list, int, float, numpy.ndarray, None. "
                           f"Got: {repr(target_value)}")
    
    def analyze_target_answers(self, df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """
        Analyze target answers distribution and patterns.
        
        Args:
            df: DataFrame to analyze
            split_name: Name of the split (train/test)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing target answers for {split_name} split...")
        
        # Extract target answers
        target_answers = []
        answer_counts = []
        data_types = []
        list_detection_info = []
        
        for idx, row in df.iterrows():
            try:
                # Navigate to the target field
                target_value = row.get("reward_model", {}).get("ground_truth", {}).get("target")
                target_answers.append(target_value)
                
                # Count answers
                count = self.count_answers(target_value)
                answer_counts.append(count)
                
                # Record data type and list detection info
                detection_info = {
                    'original_type': type(target_value).__name__,
                    'is_list_like_string': False,
                    'parsed_items': [],
                    'detection_method': None
                }
                
                if target_value is None:
                    data_types.append("None")
                elif isinstance(target_value, str):
                    if self.is_list_like_string(target_value):
                        data_types.append("str_as_list")
                        detection_info['is_list_like_string'] = True
                        detection_info['parsed_items'] = self.parse_list_like_string(target_value)
                        detection_info['detection_method'] = self._get_detection_method(target_value)
                    else:
                        data_types.append("str")
                elif isinstance(target_value, list):
                    data_types.append("list")
                elif isinstance(target_value, np.ndarray):
                    data_types.append("ndarray")
                elif isinstance(target_value, (int, float)):
                    data_types.append("numeric")
                else:
                    data_types.append("unsupported")
                
                list_detection_info.append(detection_info)
                    
            except ValueError as e:
                # Handle unsupported types or conversion errors
                logger.warning(f"Unsupported type in row {idx}: {e}")
                target_answers.append(None)
                answer_counts.append(0)
                data_types.append("unsupported")
                list_detection_info.append({
                    'original_type': 'unsupported',
                    'is_list_like_string': False,
                    'parsed_items': [],
                    'detection_method': None,
                    'error_message': str(e)
                })
            except Exception as e:
                # Handle other unexpected errors
                logger.warning(f"Unexpected error processing row {idx}: {e}")
                target_answers.append(None)
                answer_counts.append(0)
                data_types.append("error")
                list_detection_info.append({
                    'original_type': 'error',
                    'is_list_like_string': False,
                    'parsed_items': [],
                    'detection_method': None,
                    'error_message': str(e)
                })
        
        # Basic statistics
        total_samples = len(df)
        valid_samples = sum(1 for count in answer_counts if count > 0)
        empty_samples = sum(1 for count in answer_counts if count == 0)
        
        # Answer count distribution
        answer_count_distribution = Counter(answer_counts)
        
        # Data type distribution
        data_type_distribution = Counter(data_types)
        
        # List detection analysis
        list_detection_analysis = self._analyze_list_detection(list_detection_info)
        
        # Statistics for answer counts
        answer_counts_array = np.array(answer_counts)
        answer_stats = {
            'mean': np.mean(answer_counts_array),
            'median': np.median(answer_counts_array),
            'std': np.std(answer_counts_array),
            'min': np.min(answer_counts_array),
            'max': np.max(answer_counts_array),
            'q25': np.percentile(answer_counts_array, 25),
            'q75': np.percentile(answer_counts_array, 75)
        }
        
        # Analyze by data source if available
        source_analysis = {}
        if 'data_source' in df.columns:
            for source in df['data_source'].unique():
                source_mask = df['data_source'] == source
                source_answer_counts = [answer_counts[i] for i in range(len(answer_counts)) if source_mask.iloc[i]]
                source_answer_counts_array = np.array(source_answer_counts)
                
                source_analysis[source] = {
                    'count': len(source_answer_counts),
                    'mean_answers': np.mean(source_answer_counts_array),
                    'std_answers': np.std(source_answer_counts_array),
                    'min_answers': np.min(source_answer_counts_array),
                    'max_answers': np.max(source_answer_counts_array),
                    'zero_answers': sum(1 for c in source_answer_counts if c == 0),
                    'single_answers': sum(1 for c in source_answer_counts if c == 1),
                    'multiple_answers': sum(1 for c in source_answer_counts if c > 1)
                }
        
        # Analyze answer count patterns
        pattern_analysis = self._analyze_answer_patterns(answer_counts, target_answers)
        
        results = {
            'split_name': split_name,
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'empty_samples': empty_samples,
            'valid_ratio': valid_samples / total_samples if total_samples > 0 else 0,
            'answer_count_distribution': dict(answer_count_distribution),
            'data_type_distribution': dict(data_type_distribution),
            'list_detection_analysis': list_detection_analysis,
            'answer_stats': answer_stats,
            'source_analysis': source_analysis,
            'pattern_analysis': pattern_analysis,
            'raw_answer_counts': answer_counts,
            'raw_data_types': data_types
        }
        
        self.results[split_name] = results
        return results
    
    def _analyze_answer_patterns(self, answer_counts: List[int], target_answers: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in answer counts and target answers."""
        patterns = {}
        
        # Pattern 1: Distribution of answer counts
        patterns['answer_count_ranges'] = {
            'zero_answers': sum(1 for c in answer_counts if c == 0),
            'single_answer': sum(1 for c in answer_counts if c == 1),
            'multiple_answers_2_5': sum(1 for c in answer_counts if 2 <= c <= 5),
            'multiple_answers_6_10': sum(1 for c in answer_counts if 6 <= c <= 10),
            'many_answers_10_plus': sum(1 for c in answer_counts if c > 10)
        }
        
        # Pattern 2: Analyze list lengths for list-type targets
        list_lengths = []
        for target in target_answers:
            if isinstance(target, list):
                list_lengths.append(len(target))
        
        if list_lengths:
            patterns['list_length_stats'] = {
                'mean': np.mean(list_lengths),
                'median': np.median(list_lengths),
                'std': np.std(list_lengths),
                'min': np.min(list_lengths),
                'max': np.max(list_lengths)
            }
        else:
            patterns['list_length_stats'] = None
        
        # Pattern 3: String length analysis for string-type targets
        string_lengths = []
        for target in target_answers:
            if isinstance(target, str) and target.strip():
                string_lengths.append(len(target.strip()))
        
        if string_lengths:
            patterns['string_length_stats'] = {
                'mean': np.mean(string_lengths),
                'median': np.median(string_lengths),
                'std': np.std(string_lengths),
                'min': np.min(string_lengths),
                'max': np.max(string_lengths)
            }
        else:
            patterns['string_length_stats'] = None
        
        return patterns
    
    def _analyze_list_detection(self, list_detection_info: List[Dict]) -> Dict[str, Any]:
        """Analyze list detection patterns and methods."""
        analysis = {
            'total_strings': 0,
            'list_like_strings': 0,
            'detection_methods': Counter(),
            'parsed_item_counts': [],
            'detection_examples': {},
            'false_positives': 0,
            'detection_accuracy': 0.0
        }
        
        for info in list_detection_info:
            if info['original_type'] == 'str':
                analysis['total_strings'] += 1
                
                if info['is_list_like_string']:
                    analysis['list_like_strings'] += 1
                    method = info['detection_method']
                    analysis['detection_methods'][method] += 1
                    analysis['parsed_item_counts'].append(len(info['parsed_items']))
                    
                    # Store examples for each detection method
                    if method not in analysis['detection_examples']:
                        analysis['detection_examples'][method] = []
                    if len(analysis['detection_examples'][method]) < 3:  # Keep only first 3 examples
                        analysis['detection_examples'][method].append({
                            'parsed_items': info['parsed_items'][:5],  # First 5 items
                            'item_count': len(info['parsed_items'])
                        })
        
        # Calculate detection accuracy (simplified heuristic)
        if analysis['total_strings'] > 0:
            analysis['detection_accuracy'] = analysis['list_like_strings'] / analysis['total_strings']
        
        # Convert Counter to dict for JSON serialization
        analysis['detection_methods'] = dict(analysis['detection_methods'])
        
        return analysis
    
    def print_detailed_analysis(self, results: Dict[str, Any]):
        """Print detailed analysis results."""
        split_name = results['split_name']
        
        print(f"\n{'='*80}")
        print(f"TARGET ANSWERS ANALYSIS - {split_name.upper()}")
        print(f"{'='*80}")
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"  Total samples: {results['total_samples']:,}")
        print(f"  Valid samples (with answers): {results['valid_samples']:,}")
        print(f"  Empty samples (no answers): {results['empty_samples']:,}")
        print(f"  Valid ratio: {results['valid_ratio']:.2%}")
        
        # Answer count distribution
        print(f"\n📈 ANSWER COUNT DISTRIBUTION:")
        for count, freq in sorted(results['answer_count_distribution'].items()):
            ratio = freq / results['total_samples']
            print(f"  {count} answers: {freq:,} samples ({ratio:.2%})")
        
        # Data type distribution
        print(f"\n🔤 DATA TYPE DISTRIBUTION:")
        for dtype, freq in sorted(results['data_type_distribution'].items()):
            ratio = freq / results['total_samples']
            print(f"  {dtype}: {freq:,} samples ({ratio:.2%})")
        
        # Check for unsupported types and errors
        unsupported_count = results['data_type_distribution'].get('unsupported', 0)
        error_count = results['data_type_distribution'].get('error', 0)
        if unsupported_count > 0 or error_count > 0:
            print(f"\n⚠️  WARNING:")
            if unsupported_count > 0:
                print(f"  Unsupported types: {unsupported_count:,} samples")
            if error_count > 0:
                print(f"  Processing errors: {error_count:,} samples")
            print(f"  These samples were excluded from analysis.")
        
        # List detection analysis
        print(f"\n🔍 LIST DETECTION ANALYSIS:")
        list_analysis = results['list_detection_analysis']
        print(f"  Total string samples: {list_analysis['total_strings']:,}")
        print(f"  List-like strings detected: {list_analysis['list_like_strings']:,}")
        if list_analysis['total_strings'] > 0:
            detection_ratio = list_analysis['list_like_strings'] / list_analysis['total_strings']
            print(f"  Detection ratio: {detection_ratio:.2%}")
        
        print(f"\n  Detection methods used:")
        for method, count in sorted(list_analysis['detection_methods'].items()):
            if list_analysis['list_like_strings'] > 0:
                method_ratio = count / list_analysis['list_like_strings']
                print(f"    {method}: {count:,} ({method_ratio:.2%})")
        
        if list_analysis['parsed_item_counts']:
            item_counts = list_analysis['parsed_item_counts']
            print(f"\n  Parsed item count statistics:")
            print(f"    Mean items per list: {np.mean(item_counts):.2f}")
            print(f"    Median items per list: {np.median(item_counts):.1f}")
            print(f"    Max items in a list: {max(item_counts)}")
            print(f"    Min items in a list: {min(item_counts)}")
        
        # Show examples for each detection method
        if list_analysis['detection_examples']:
            print(f"\n  Detection examples:")
            for method, examples in list_analysis['detection_examples'].items():
                print(f"    {method}:")
                for i, example in enumerate(examples[:2]):  # Show first 2 examples
                    items_preview = example['parsed_items'][:3]  # First 3 items
                    items_str = ', '.join(f'"{item}"' for item in items_preview)
                    if len(example['parsed_items']) > 3:
                        items_str += f" ... (+{len(example['parsed_items'])-3} more)"
                    print(f"      Example {i+1}: [{items_str}] ({example['item_count']} items)")
        
        # Answer statistics
        print(f"\n📊 ANSWER COUNT STATISTICS:")
        stats = results['answer_stats']
        print(f"  Mean answers per sample: {stats['mean']:.2f}")
        print(f"  Median answers per sample: {stats['median']:.1f}")
        print(f"  Standard deviation: {stats['std']:.2f}")
        print(f"  Min answers: {stats['min']}")
        print(f"  Max answers: {stats['max']}")
        print(f"  25th percentile: {stats['q25']:.1f}")
        print(f"  75th percentile: {stats['q75']:.1f}")
        
        # Pattern analysis
        print(f"\n🔍 PATTERN ANALYSIS:")
        patterns = results['pattern_analysis']
        print(f"  Zero answers: {patterns['answer_count_ranges']['zero_answers']:,}")
        print(f"  Single answer: {patterns['answer_count_ranges']['single_answer']:,}")
        print(f"  Multiple answers (2-5): {patterns['answer_count_ranges']['multiple_answers_2_5']:,}")
        print(f"  Multiple answers (6-10): {patterns['answer_count_ranges']['multiple_answers_6_10']:,}")
        print(f"  Many answers (10+): {patterns['answer_count_ranges']['many_answers_10_plus']:,}")
        
        # List length statistics
        if patterns['list_length_stats']:
            print(f"\n📋 LIST LENGTH STATISTICS:")
            list_stats = patterns['list_length_stats']
            print(f"  Mean list length: {list_stats['mean']:.2f}")
            print(f"  Median list length: {list_stats['median']:.1f}")
            print(f"  Max list length: {list_stats['max']}")
        
        # String length statistics
        if patterns['string_length_stats']:
            print(f"\n📝 STRING LENGTH STATISTICS:")
            str_stats = patterns['string_length_stats']
            print(f"  Mean string length: {str_stats['mean']:.1f}")
            print(f"  Median string length: {str_stats['median']:.1f}")
            print(f"  Max string length: {str_stats['max']}")
        
        # Source analysis
        if results['source_analysis']:
            print(f"\n🏷️  ANALYSIS BY DATA SOURCE:")
            for source, analysis in results['source_analysis'].items():
                print(f"\n  {source}:")
                print(f"    Samples: {analysis['count']:,}")
                print(f"    Mean answers: {analysis['mean_answers']:.2f}")
                print(f"    Zero answers: {analysis['zero_answers']:,}")
                print(f"    Single answers: {analysis['single_answers']:,}")
                print(f"    Multiple answers: {analysis['multiple_answers']:,}")
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = None):
        """Create visualization plots for the analysis."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        split_name = results['split_name']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Target Answers Analysis - {split_name}', fontsize=16)
        
        # 1. Answer count distribution (bar chart)
        ax1 = axes[0, 0]
        counts = sorted(results['answer_count_distribution'].items())
        x_vals, y_vals = zip(*counts)
        ax1.bar(x_vals, y_vals, alpha=0.7)
        ax1.set_xlabel('Number of Answers')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Answer Count Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Data type distribution (pie chart)
        ax2 = axes[0, 1]
        types = list(results['data_type_distribution'].keys())
        type_counts = list(results['data_type_distribution'].values())
        ax2.pie(type_counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Data Type Distribution')
        
        # 3. Answer count histogram
        ax3 = axes[0, 2]
        answer_counts = results['raw_answer_counts']
        ax3.hist(answer_counts, bins=min(20, len(set(answer_counts))), alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Number of Answers')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Answer Count Histogram')
        ax3.grid(True, alpha=0.3)
        
        # 4. Answer count ranges (bar chart)
        ax4 = axes[1, 0]
        ranges = results['pattern_analysis']['answer_count_ranges']
        range_labels = ['Zero', 'Single', '2-5', '6-10', '10+']
        range_values = [ranges['zero_answers'], ranges['single_answer'], 
                       ranges['multiple_answers_2_5'], ranges['multiple_answers_6_10'], 
                       ranges['many_answers_10_plus']]
        ax4.bar(range_labels, range_values, alpha=0.7)
        ax4.set_xlabel('Answer Count Range')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Answer Count Ranges')
        ax4.grid(True, alpha=0.3)
        
        # 5. Source analysis (if available)
        ax5 = axes[1, 1]
        if results['source_analysis']:
            sources = list(results['source_analysis'].keys())
            mean_answers = [results['source_analysis'][s]['mean_answers'] for s in sources]
            ax5.bar(sources, mean_answers, alpha=0.7)
            ax5.set_xlabel('Data Source')
            ax5.set_ylabel('Mean Answers per Sample')
            ax5.set_title('Mean Answers by Data Source')
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No source analysis available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Source Analysis')
        
        # 6. Answer count box plot
        ax6 = axes[1, 2]
        ax6.boxplot([answer_counts], labels=['All Samples'])
        ax6.set_ylabel('Number of Answers')
        ax6.set_title('Answer Count Box Plot')
        ax6.grid(True, alpha=0.3)
        
        # 7. List detection methods
        ax7 = axes[2, 0]
        list_analysis = results['list_detection_analysis']
        if list_analysis['detection_methods']:
            methods = list(list_analysis['detection_methods'].keys())
            method_counts = list(list_analysis['detection_methods'].values())
            ax7.bar(methods, method_counts, alpha=0.7)
            ax7.set_xlabel('Detection Method')
            ax7.set_ylabel('Count')
            ax7.set_title('List Detection Methods')
            ax7.tick_params(axis='x', rotation=45)
        else:
            ax7.text(0.5, 0.5, 'No list-like strings detected', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('List Detection Methods')
        
        # 8. Data type distribution
        ax8 = axes[2, 1]
        str_count = results['data_type_distribution'].get('str', 0)
        str_as_list_count = results['data_type_distribution'].get('str_as_list', 0)
        list_count = results['data_type_distribution'].get('list', 0)
        ndarray_count = results['data_type_distribution'].get('ndarray', 0)
        numeric_count = results['data_type_distribution'].get('numeric', 0)
        none_count = results['data_type_distribution'].get('None', 0)
        unsupported_count = results['data_type_distribution'].get('unsupported', 0)
        error_count = results['data_type_distribution'].get('error', 0)
        
        categories = ['String', 'String as List', 'List', 'Numpy Array', 'Numeric', 'None', 'Unsupported', 'Error']
        values = [str_count, str_as_list_count, list_count, ndarray_count, numeric_count, none_count, unsupported_count, error_count]
        colors = ['lightblue', 'orange', 'lightgreen', 'purple', 'yellow', 'gray', 'red', 'darkred']
        
        # Only show categories with non-zero values
        non_zero_categories = []
        non_zero_values = []
        non_zero_colors = []
        for cat, val, col in zip(categories, values, colors):
            if val > 0:
                non_zero_categories.append(cat)
                non_zero_values.append(val)
                non_zero_colors.append(col)
        
        if non_zero_categories:
            ax8.bar(non_zero_categories, non_zero_values, color=non_zero_colors, alpha=0.7)
            ax8.set_ylabel('Count')
            ax8.set_title('Data Type Distribution')
            ax8.tick_params(axis='x', rotation=45)
        else:
            ax8.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Data Type Distribution')
        
        # 9. Parsed item count distribution
        ax9 = axes[2, 2]
        if list_analysis['parsed_item_counts']:
            item_counts = list_analysis['parsed_item_counts']
            ax9.hist(item_counts, bins=min(20, len(set(item_counts))), alpha=0.7, edgecolor='black')
            ax9.set_xlabel('Number of Items in Parsed Lists')
            ax9.set_ylabel('Frequency')
            ax9.set_title('Parsed List Item Count Distribution')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'No parsed lists available', 
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Parsed List Item Count Distribution')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, f'target_answers_analysis_{split_name}.png')
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
            output_path = os.path.join(output_dir, f'target_answers_analysis_{split_name}.json')
            
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
                if key in ['raw_answer_counts', 'raw_data_types']:
                    # Skip raw data for JSON to keep file size manageable
                    continue
                elif key == 'source_analysis':
                    clean_results[key] = {
                        source: {k: convert_numpy(v) for k, v in analysis.items()}
                        for source, analysis in value.items()
                    }
                elif key == 'answer_stats':
                    clean_results[key] = {k: convert_numpy(v) for k, v in value.items()}
                elif key == 'pattern_analysis':
                    clean_results[key] = {}
                    for pattern_key, pattern_value in value.items():
                        if pattern_value is None:
                            clean_results[key][pattern_key] = None
                        else:
                            clean_results[key][pattern_key] = {k: convert_numpy(v) for k, v in pattern_value.items()}
                else:
                    clean_results[key] = convert_numpy(value)
            
            with open(output_path, 'w') as f:
                json.dump(clean_results, f, indent=2)
            
            logger.info(f"Analysis results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze target answers distribution in parquet files.")
    parser.add_argument(
        "--data_dir",
        default="searchR1_processed_direct",
        help="Directory containing the processed parquet files"
    )
    parser.add_argument(
        "--output_dir",
        default="target_answers_analysis_results",
        help="Directory to save analysis results and visualizations"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TargetAnswersAnalyzer()
    
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
        results = analyzer.analyze_target_answers(df, split_name)
        
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
