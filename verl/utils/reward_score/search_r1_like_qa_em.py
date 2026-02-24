# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string

import numpy as np
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _normalize_and_tokenize(s: str):
    """Normalize (using your normalize_answer) then whitespace-split."""
    if s is None:
        return []
    return normalize_answer(s).split()

def f1_score(prediction, golden_answers) -> float:
    """
    Token-level F1 (max over references).
    prediction: str
    golden_answers: str | list[str] | np.ndarray
    """
    if isinstance(golden_answers, np.ndarray):
        golden_answers = golden_answers.tolist()
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred_tokens = _normalize_and_tokenize(prediction)
    if not golden_answers:
        return 0.0

    best_f1 = 0.0
    for ga in golden_answers:
        gold_tokens = _normalize_and_tokenize(ga)

        # Both empty after normalization => perfect match
        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            best_f1 = max(best_f1, 1.0)
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1

    return float(best_f1)

def compute_em_f1(solution_str, ground_truth, method: str = "strict"):
    """
    Extract the last <answer>...</answer> from solution_str,
    then compute EM and F1 against ground_truth['target'].

    Returns:
        (em, f1) as floats in [0, 1].
    """
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0.0, 0.0

    gold = ground_truth["target"]
    em = float(em_check(answer, gold))  # 1.0 or 0.0
    f1 = f1_score(answer, gold)
    return em, f1


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, np.ndarray):
        golden_answers = golden_answers.tolist()
    
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, np.ndarray):
        golden_answers = golden_answers.tolist()

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", 
    format_score=0.4, 
    acc_score=1.0, 
    return_dict=False, 
    data_source=None, 
    extra_info=None, 
    eval=False,
    score_source="em",
    ):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        acc_score: the score for the correct answer (unused, kept for compatibility)
        return_dict: if True, return dict with em, f1, format_score, score; else return single score
        data_source: data source identifier (unused, kept for compatibility)
        extra_info: additional information (unused, kept for compatibility)
        eval: evaluation mode flag (unused, kept for compatibility)
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 256) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # Calculate format score and accuracy score separately
    if answer is None:
        # No valid format, no accuracy
        if return_dict:
            return {
                "em": 0.0, 
                "f1": 0.0, 
                "format_reward": 0.0, 
                "score": 0.0,
            }
        return 0  # format_score=0 + acc_score=0
    else:
        # Valid format exists, give format score
        current_format_score = format_score
        em, f1 = compute_em_f1(solution_str, ground_truth)
        if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
            em = em / 4.
            f1 = f1 / 4.
            
        if return_dict:
            return {
                "em": em,
                "f1": f1,
                "format_reward": current_format_score,
                "score": current_format_score + em if score_source == "em" else current_format_score + f1,
            }
        else:
            # Return single score for backward compatibility
            return current_format_score + em if score_source == "em" else current_format_score + f1

def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, acc_score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        acc_score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    # Calculate format score and accuracy score separately
    if answer is None:
        # No valid format, no accuracy
        return 0  # format_score=0 + acc_score=0
    else:
        # Valid format exists, give format score
        current_format_score = format_score
        
        if subem_check(answer, ground_truth["target"]):
            # Correct answer, give both format and accuracy scores
            return current_format_score + acc_score
        else:
            # Wrong answer, only give format score
            return current_format_score
