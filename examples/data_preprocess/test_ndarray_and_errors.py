#!/usr/bin/env python3
"""
Test script for numpy array handling and error handling functionality.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_target_answers import TargetAnswersAnalyzer

def test_ndarray_and_errors():
    """Test numpy array handling and error handling."""
    analyzer = TargetAnswersAnalyzer()
    
    # Test cases
    test_cases = [
        # (input, expected_count, description)
        (np.array(["answer1", "answer2", "answer3"]), 3, "1D string array"),
        (np.array([1, 2, 3, 4, 5]), 5, "1D numeric array"),
        (np.array([]), 0, "Empty array"),
        (np.array([["a", "b"], ["c", "d"]]), 2, "2D array (flattened)"),
        (np.array([1]), 1, "Single element array"),
        (np.array(["single"]), 1, "Single string array"),
        ({"key": "value"}, None, "Dict (should raise error)"),
        (set(["a", "b", "c"]), None, "Set (should raise error)"),
        (tuple(["a", "b"]), None, "Tuple (should raise error)"),
        (complex(1, 2), None, "Complex number (should raise error)"),
    ]
    
    print("Testing numpy array handling and error handling...")
    print("=" * 80)
    
    all_passed = True
    
    for i, (input_val, expected_count, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"  Input: {repr(input_val)} (type: {type(input_val).__name__})")
        
        try:
            actual_count = analyzer.count_answers(input_val)
            print(f"  Answer count: {actual_count} (expected: {expected_count})")
            
            if actual_count == expected_count:
                print("  ✅ PASSED")
            else:
                print("  ❌ FAILED")
                all_passed = False
                
        except ValueError as e:
            if expected_count is None:
                print(f"  ✅ PASSED (correctly raised error: {e})")
            else:
                print(f"  ❌ FAILED (unexpected error: {e})")
                all_passed = False
        except Exception as e:
            print(f"  ❌ FAILED (unexpected error type: {e})")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

def test_numpy_array_conversion():
    """Test specific numpy array conversion scenarios."""
    analyzer = TargetAnswersAnalyzer()
    
    print("\nTesting numpy array conversion details...")
    print("-" * 50)
    
    # Test different numpy array types
    arrays = [
        np.array(["answer1", "answer2", "answer3"]),
        np.array([1, 2, 3, 4, 5]),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([1.5, 2.5, 3.5]),
        np.array([True, False, True]),
    ]
    
    for i, arr in enumerate(arrays, 1):
        print(f"\nArray {i}: {arr}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        
        try:
            count = analyzer.count_answers(arr)
            converted = arr.tolist()
            print(f"  Converted to list: {converted}")
            print(f"  Count: {count}")
            print("  ✅ SUCCESS")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

if __name__ == "__main__":
    test_ndarray_and_errors()
    test_numpy_array_conversion()
