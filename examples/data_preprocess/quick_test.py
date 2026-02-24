#!/usr/bin/env python3
"""
Quick test of the list detection functionality.
"""

import numpy as np
from analyze_target_answers import TargetAnswersAnalyzer

def quick_test():
    """Quick test of list detection and numpy array handling."""
    analyzer = TargetAnswersAnalyzer()
    
    # Test some examples
    examples = [
        "single answer",
        '["answer1", "answer2", "answer3"]',
        "answer1, answer2, answer3",
        "answer1|answer2|answer3",
        "answer1\nanswer2\nanswer3",
        '{"answer1", "answer2"}',
        "answer1, answer2, answer3, answer4, answer5",
        np.array(["answer1", "answer2", "answer3"]),
        np.array([1, 2, 3, 4, 5]),
        np.array([]),
        {"key": "value"},  # Should raise error
        set(["a", "b"]),   # Should raise error
    ]
    
    print("Quick List Detection Test")
    print("=" * 50)
    
    for example in examples:
        print(f"\nInput: {repr(example)} (type: {type(example).__name__})")
        
        try:
            if isinstance(example, str):
                is_list = analyzer.is_list_like_string(example)
                if is_list:
                    parsed = analyzer.parse_list_like_string(example)
                    count = len(parsed)
                    print(f"  → Detected as list with {count} items: {parsed}")
                else:
                    count = analyzer.count_answers(example)
                    print(f"  → Regular string, count: {count}")
            else:
                count = analyzer.count_answers(example)
                if isinstance(example, np.ndarray):
                    converted = example.tolist()
                    print(f"  → Numpy array converted to list: {converted}")
                print(f"  → Count: {count}")
        except ValueError as e:
            print(f"  → ❌ ERROR (expected): {e}")
        except Exception as e:
            print(f"  → ❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    quick_test()
