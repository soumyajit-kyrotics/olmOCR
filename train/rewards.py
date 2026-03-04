"""
rewards.py — Reward functions for GRPO/RLVR training.

Maps unit test evaluation results to scalar rewards for the GRPO
(Group Relative Policy Optimization) reinforcement learning algorithm.

The reward function is the critical interface between the olmOCR-Bench
synthetic tests and the RL training loop. It determines what behavior
the model is optimized toward.

Reward strategies:
  - binary:       1.0 if ALL tests pass, 0.0 otherwise
  - proportional: num_passed / num_total (smoother gradient signal)
  - weighted:     different weights per test type (e.g., text > formatting)
"""

import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.evaluate_tests import evaluate_tests


# ---------------------------------------------------------------------------
# Reward strategies
# ---------------------------------------------------------------------------

def binary_reward(ocr_text: str, tests: list) -> float:
    """
    All-or-nothing reward: 1.0 if every test passes, 0.0 otherwise.

    Pro: Strong signal — model must get everything right.
    Con: Sparse — early in training, almost all completions score 0.0.
    """
    result = evaluate_tests(ocr_text, tests)
    return 1.0 if result["num_failed"] == 0 else 0.0


def proportional_reward(ocr_text: str, tests: list) -> float:
    """
    Proportional reward: fraction of tests passed.

    Pro: Dense signal — model gets credit for partial progress.
    Con: All test types weighted equally.
    """
    result = evaluate_tests(ocr_text, tests)
    return result["score"]


# Default weights per test type (higher = more important)
DEFAULT_TYPE_WEIGHTS = {
    "text_presence": 2.0,   # Core OCR accuracy — most important
    "text_absence": 1.0,    # Header/footer exclusion
    "table": 1.5,           # Table structure preservation
    "order": 1.5,           # Reading order
    "math": 2.0,            # LaTeX equation accuracy
    "formatting": 0.5,      # Formatting preservation (least critical)
}


def weighted_reward(
    ocr_text: str,
    tests: list,
    type_weights: dict | None = None,
) -> float:
    """
    Weighted proportional reward: different test types contribute
    differently to the reward.

    Pro: Lets you emphasize text accuracy over formatting.
    Con: Requires tuning the weights.

    Args:
        ocr_text: model's generated text
        tests: list of test dicts
        type_weights: optional dict mapping test_type -> weight (default: DEFAULT_TYPE_WEIGHTS)

    Returns:
        Weighted score in [0.0, 1.0]
    """
    weights = type_weights or DEFAULT_TYPE_WEIGHTS
    result = evaluate_tests(ocr_text, tests)

    total_weight = 0.0
    weighted_score = 0.0

    for r in result["results"]:
        w = weights.get(r["type"], 1.0)
        total_weight += w
        if r["passed"]:
            weighted_score += w

    return weighted_score / total_weight if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API — configurable reward function
# ---------------------------------------------------------------------------

REWARD_STRATEGIES = {
    "binary": binary_reward,
    "proportional": proportional_reward,
    "weighted": weighted_reward,
}


def compute_reward(
    ocr_text: str,
    tests: list,
    strategy: str = "proportional",
    **kwargs,
) -> float:
    """
    Compute the reward for a model completion against a set of unit tests.

    This is the function called by the GRPO training loop.

    Args:
        ocr_text: the model's generated OCR text
        tests: list of test dicts (from generate_tests.py)
        strategy: reward strategy name ("binary", "proportional", "weighted")
        **kwargs: additional arguments passed to the strategy function

    Returns:
        Scalar reward in [0.0, 1.0]
    """
    reward_fn = REWARD_STRATEGIES.get(strategy)
    if reward_fn is None:
        raise ValueError(f"Unknown reward strategy: {strategy}. "
                         f"Available: {list(REWARD_STRATEGIES.keys())}")

    if strategy == "weighted":
        return reward_fn(ocr_text, tests, **kwargs)
    return reward_fn(ocr_text, tests)


def compute_batch_rewards(
    completions: list[str],
    tests: list,
    strategy: str = "proportional",
    **kwargs,
) -> list[float]:
    """
    Compute rewards for a batch of model completions against the same tests.

    Used in GRPO where multiple completions are sampled per prompt.

    Args:
        completions: list of model-generated text strings
        tests: list of test dicts (shared across all completions)
        strategy: reward strategy name
        **kwargs: additional arguments

    Returns:
        List of scalar rewards, one per completion
    """
    return [compute_reward(c, tests, strategy, **kwargs) for c in completions]
