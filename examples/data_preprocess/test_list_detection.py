#!/usr/bin/env python3
"""
Test script for list detection functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_target_answers import TargetAnswersAnalyzer

def test_list_detection():
    """Test the list detection functionality with various examples."""
    analyzer = TargetAnswersAnalyzer()
    
    # Test cases
    test_cases = [
        # (input, expected_is_list, expected_count, description)
        ("single answer", False, 1, "Regular string"),
        ("", False, 0, "Empty string"),
        (None, False, 0, "None value"),
        (["answer1", "answer2"], False, 2, "Actual list"),
        ('["answer1", "answer2"]', True, 2, "JSON list string"),
        ("answer1, answer2, answer3", True, 3, "Comma-separated"),
        ("answer1|answer2|answer3", True, 3, "Pipe-separated"),
        ("answer1\nanswer2\nanswer3", True, 3, "Newline-separated"),
        ("(answer1, answer2)", True, 2, "Parenthesis notation"),
        ("{answer1, answer2}", True, 2, "Brace notation"),
        ("answer1, answer2, answer3, answer4, answer5", True, 5, "Long comma-separated"),
        ("single,", True, 1, "Comma with single item"),
        ("answer1,answer2,answer3", True, 3, "Comma-separated no spaces"),
        ('"quoted, string"', False, 1, "Quoted string with comma"),
        ("answer1, answer2, answer3, answer4, answer5, answer6, answer7", True, 7, "Very long list"),
    ]
    
    print("Testing list detection functionality...")
    print("=" * 80)
    
    all_passed = True
    
    for i, (input_val, expected_is_list, expected_count, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"  Input: {repr(input_val)}")
        
        # Test list detection
        if isinstance(input_val, str):
            is_list_like = analyzer.is_list_like_string(input_val)
            print(f"  Is list-like: {is_list_like} (expected: {expected_is_list})")
            
            if is_list_like:
                parsed_items = analyzer.parse_list_like_string(input_val)
                print(f"  Parsed items: {parsed_items}")
                print(f"  Parsed count: {len(parsed_items)} (expected: {expected_count})")
        
        # Test answer counting
        actual_count = analyzer.count_answers(input_val)
        print(f"  Answer count: {actual_count} (expected: {expected_count})")
        
        # Check if test passed
        if actual_count == expected_count:
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_list_detection()
