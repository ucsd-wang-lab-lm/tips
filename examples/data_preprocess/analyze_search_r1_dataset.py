# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import numpy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchR1DatasetAnalyzer:
    """Analyzer for SearchR1 processed dataset with token length filtering and data source sampling."""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize the analyzer with a tokenizer."""
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load the tokenizer for token length calculation."""
        try:
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def calculate_prompt_token_length(self, prompt: List[Dict[str, str]]) -> int:
        """
        Calculate the token length of a prompt.
        
        Args:
            prompt: List of dictionaries with 'role' and 'content' keys
            
        Returns:
            int: Total token length of the prompt
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
            
        total_length = 0
        for message in prompt:
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                if content:
                    tokens = self.tokenizer.encode(content, add_special_tokens=False)
                    total_length += len(tokens)
        
        return total_length
    
    def calculate_batch_token_lengths(self, prompts: List[List[Dict[str, str]]]) -> List[int]:
        """
        Calculate token lengths for a batch of prompts efficiently using batch tokenization.
        
        Args:
            prompts: List of prompts, each being a list of message dictionaries
            
        Returns:
            List of token lengths for each prompt
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        token_lengths = []
        
        # Process prompts in larger batches for better efficiency
        batch_size = 128  # Process 128 prompts at a time
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Prepare texts for batch tokenization
            batch_texts = []
            for prompt in batch_prompts:
                try:
                    # Concatenate all content from the prompt
                    full_text = ""
                    for message in prompt:
                        if isinstance(message, dict) and 'content' in message:
                            content = message['content']
                            if content:
                                full_text += content + " "  # Add space between messages
                    
                    batch_texts.append(full_text.strip() if full_text.strip() else "")
                        
                except Exception as e:
                    logger.warning(f"Error preparing prompt in batch: {e}")
                    batch_texts.append("")
            
            # Batch tokenize all texts at once
            try:
                if batch_texts:
                    # Use batch_encode_plus for better efficiency
                    batch_encodings = self.tokenizer.batch_encode_plus(
                        batch_texts,
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,
                        return_attention_mask=False
                    )
                    
                    # Extract token lengths
                    batch_lengths = [len(encoding) for encoding in batch_encodings['input_ids']]
                else:
                    batch_lengths = [0] * len(batch_prompts)
                    
            except Exception as e:
                logger.warning(f"Error in batch tokenization: {e}")
                # Fallback to individual processing
                batch_lengths = []
                for text in batch_texts:
                    try:
                        if text:
                            tokens = self.tokenizer.encode(text, add_special_tokens=False)
                            batch_lengths.append(len(tokens))
                        else:
                            batch_lengths.append(0)
                    except:
                        batch_lengths.append(0)
            
            token_lengths.extend(batch_lengths)
            
            # Log progress for large datasets
            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"Processed {min(i + batch_size, len(prompts))}/{len(prompts)} prompts")
        
        return token_lengths
    
    def load_parquet_files(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load parquet files from the specified directory.
        
        Args:
            data_dir: Directory containing parquet files
            
        Returns:
            Dict mapping split names to DataFrames
        """
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
    
    def analyze_data_sources(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data source distribution in the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data source statistics
        """
        data_source_counts = df['data_source'].value_counts()
        
        analysis = {
            'total_samples': len(df),
            'unique_data_sources': len(data_source_counts),
            'data_source_counts': data_source_counts.to_dict(),
            'data_source_ratios': (data_source_counts / len(df)).to_dict()
        }
        
        return analysis
    
    def filter_by_token_length(self, df: pd.DataFrame, min_tokens: int = None, max_tokens: int = None) -> pd.DataFrame:
        """
        Filter DataFrame by prompt token length using batch processing.
        
        Args:
            df: DataFrame to filter
            min_tokens: Minimum token length (inclusive)
            max_tokens: Maximum token length (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Calculating token lengths for {len(df)} prompts using batch processing...")
        
        # Prepare prompts for batch processing
        prompts = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                prompt = row['prompt']
                if isinstance(prompt, str):
                    # If prompt is stored as string, try to parse it
                    import json
                    prompt = json.loads(prompt)
                elif isinstance(prompt, numpy.ndarray):
                    prompt = prompt.tolist()
                elif not isinstance(prompt, list):
                    logger.warning(f"Unexpected prompt format at row {idx}: {type(prompt)}")
                    continue
                
                prompts.append(prompt)
                valid_indices.append(idx)
                
            except Exception as e:
                logger.warning(f"Error preparing prompt for row {idx}: {e}")
                continue
        
        logger.info(f"Prepared {len(prompts)} valid prompts for batch processing")
        
        # Calculate token lengths in batches
        if prompts:
            token_lengths = self.calculate_batch_token_lengths(prompts)
        else:
            token_lengths = []
        
        # Create a mapping from valid indices to token lengths
        token_length_dict = dict(zip(valid_indices, token_lengths))
        
        # Add token_length column to dataframe
        df['token_length'] = df.index.map(token_length_dict).fillna(0).astype(int)
        
        logger.info(f"Token length calculation completed. Mean length: {df['token_length'].mean():.2f}")
        
        # Apply filters
        original_length = len(df)
        if min_tokens is not None:
            df = df[df['token_length'] >= min_tokens]
            logger.info(f"After min_tokens filter ({min_tokens}): {len(df)} rows (removed {original_length - len(df)})")
        
        if max_tokens is not None:
            df = df[df['token_length'] <= max_tokens]
            logger.info(f"After max_tokens filter ({max_tokens}): {len(df)} rows (removed {original_length - len(df)})")
        
        return df
    
    def sample_by_data_source(self, df: pd.DataFrame, sampling_ratio: float = 1.0) -> pd.DataFrame:
        """
        Sample data based on a uniform sampling ratio applied to all data sources.
        
        Args:
            df: DataFrame to sample from
            sampling_ratio: Uniform sampling ratio applied to all data sources (0.0 to 1.0)
            
        Returns:
            Sampled DataFrame
        """
        if sampling_ratio <= 0:
            logger.warning(f"Invalid sampling ratio {sampling_ratio}, using 1.0")
            sampling_ratio = 1.0
        elif sampling_ratio > 1:
            logger.warning(f"Sampling ratio {sampling_ratio} > 1, using 1.0")
            sampling_ratio = 1.0
        
        sampled_dfs = []
        unique_sources = df['data_source'].unique()
        
        for data_source in unique_sources:
            source_df = df[df['data_source'] == data_source]
            if len(source_df) == 0:
                logger.warning(f"No data found for source: {data_source}")
                continue
                
            n_samples = int(len(source_df) * sampling_ratio)
            if n_samples == 0 and sampling_ratio > 0:
                n_samples = 1  # At least sample 1 if ratio > 0
                
            sampled_source_df = source_df.sample(n=min(n_samples, len(source_df)), random_state=42)
            sampled_dfs.append(sampled_source_df)
            
            logger.info(f"Sampled {len(sampled_source_df)}/{len(source_df)} rows from {data_source} (ratio: {sampling_ratio})")
        
        if sampled_dfs:
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"Total sampled rows: {len(result_df)} (uniform ratio: {sampling_ratio})")
            return result_df
        else:
            logger.warning("No data was sampled")
            return pd.DataFrame()
    
    def print_comprehensive_statistics(self, original_dfs: Dict[str, pd.DataFrame], 
                                    filtered_dfs: Dict[str, pd.DataFrame],
                                    sampled_dfs: Dict[str, pd.DataFrame]):
        """
        Print comprehensive statistics about the dataset processing.
        
        Args:
            original_dfs: Original DataFrames by split
            filtered_dfs: Token-length filtered DataFrames by split
            sampled_dfs: Data source sampled DataFrames by split
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE DATASET ANALYSIS REPORT")
        print("="*80)
        
        for split in ["train", "test"]:
            if split not in original_dfs:
                continue
                
            print(f"\n--- {split.upper()} SPLIT ANALYSIS ---")
            
            # Original data analysis
            orig_df = original_dfs[split]
            print(f"\nOriginal Data:")
            print(f"  Total samples: {len(orig_df)}")
            
            if 'token_length' in orig_df.columns:
                token_stats = orig_df['token_length'].describe()
                print(f"  Token length statistics:")
                print(f"    Mean: {token_stats['mean']:.2f}")
                print(f"    Std:  {token_stats['std']:.2f}")
                print(f"    Min:  {token_stats['min']:.2f}")
                print(f"    Max:  {token_stats['max']:.2f}")
                print(f"    25%:  {token_stats['25%']:.2f}")
                print(f"    50%:  {token_stats['50%']:.2f}")
                print(f"    75%:  {token_stats['75%']:.2f}")
            
            # Data source analysis
            data_source_analysis = self.analyze_data_sources(orig_df)
            print(f"\nData Source Distribution:")
            print(f"  Unique data sources: {data_source_analysis['unique_data_sources']}")
            for source, count in data_source_analysis['data_source_counts'].items():
                ratio = data_source_analysis['data_source_ratios'][source]
                print(f"    {source}: {count} samples ({ratio:.2%})")
            
            # Filtered data analysis
            if split in filtered_dfs:
                filt_df = filtered_dfs[split]
                print(f"\nAfter Token Length Filtering:")
                print(f"  Remaining samples: {len(filt_df)} ({len(filt_df)/len(orig_df):.2%} of original)")
                
                if len(filt_df) > 0 and 'token_length' in filt_df.columns:
                    filt_token_stats = filt_df['token_length'].describe()
                    print(f"  Filtered token length statistics:")
                    print(f"    Mean: {filt_token_stats['mean']:.2f}")
                    print(f"    Std:  {filt_token_stats['std']:.2f}")
                    print(f"    Min:  {filt_token_stats['min']:.2f}")
                    print(f"    Max:  {filt_token_stats['max']:.2f}")
            
            # Sampled data analysis
            if split in sampled_dfs:
                samp_df = sampled_dfs[split]
                print(f"\nAfter Data Source Sampling:")
                print(f"  Final samples: {len(samp_df)}")
                
                if len(samp_df) > 0:
                    samp_data_source_analysis = self.analyze_data_sources(samp_df)
                    print(f"  Sampled data source distribution:")
                    for source, count in samp_data_source_analysis['data_source_counts'].items():
                        ratio = samp_data_source_analysis['data_source_ratios'][source]
                        print(f"    {source}: {count} samples ({ratio:.2%})")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    
    def save_processed_data(self, sampled_dfs: Dict[str, pd.DataFrame], output_dir: str):
        """
        Save the processed data to parquet files.
        
        Args:
            sampled_dfs: Sampled DataFrames by split
            output_dir: Directory to save the files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split, df in sampled_dfs.items():
            if len(df) > 0:
                output_path = os.path.join(output_dir, f"{split}_processed.parquet")
                df.to_parquet(output_path, index=False)
                logger.info(f"Saved {len(df)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and process SearchR1 dataset with token length filtering and data source sampling.")
    parser.add_argument(
        "--data_dir",
        default="searchR1_processed_direct",
        help="Directory containing the processed parquet files from preprocess_search_r1_dataset.py"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer name for calculating token lengths"
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=None,
        help="Minimum token length for filtering (inclusive)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Maximum token length for filtering (inclusive)"
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.125,
        help="Uniform sampling ratio applied to all data sources (0.0 to 1.0)"
    )
    parser.add_argument(
        "--output_dir",
        default="data_preprocess/searchR1_processed",
        help="Directory to save processed data (optional)"
    )
    
    args = parser.parse_args()
    
    # Get sampling ratio
    sampling_ratio = args.sampling_ratio
    logger.info(f"Using uniform sampling ratio: {sampling_ratio}")
    
    # Initialize analyzer
    analyzer = SearchR1DatasetAnalyzer(tokenizer_name=args.tokenizer_name)
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    original_dfs = analyzer.load_parquet_files(args.data_dir)
    
    if not original_dfs:
        logger.error("No data files found!")
        return
    
    # Process each split
    filtered_dfs = {}
    sampled_dfs = {}
    
    for split, df in original_dfs.items():
        logger.info(f"\nProcessing {split} split...")
        
        # Filter by token length
        filtered_df = analyzer.filter_by_token_length(df, args.min_tokens, args.max_tokens)
        filtered_dfs[split] = filtered_df
        
        # Sample by data source
        sampled_df = analyzer.sample_by_data_source(filtered_df, sampling_ratio)
        sampled_dfs[split] = sampled_df
    
    # Print comprehensive statistics
    analyzer.print_comprehensive_statistics(original_dfs, filtered_dfs, sampled_dfs)
    
    # Save processed data if output directory specified
    if args.output_dir:
        analyzer.save_processed_data(sampled_dfs, args.output_dir)


if __name__ == "__main__":
    main()
