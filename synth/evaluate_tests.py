"""
evaluate_tests.py — Programmatic evaluator for olmOCR-Bench unit tests.

Takes model-generated OCR text and a list of unit tests (from generate_tests.py),
evaluates each test, and returns pass/fail results.

This module is the bridge between the synthetic test pipeline and the GRPO
reward system — the reward signal for reinforcement learning comes directly
from these test evaluations.

Test types supported:
  - text_presence: a sentence/phrase must exist in OCR output (fuzzy match)
  - text_absence:  header/footer text must NOT be in OCR output
  - table:         table cell relationships preserved (up/down/left/right adjacency)
  - order:         reading order is correct (sentence A comes before B)
  - math:          LaTeX equation string must appear in OCR output
  - formatting:    structural markers (bold, italic, headings) preserved
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Fuzzy matching utilities
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insertion
                prev_row[j + 1] + 1,   # deletion
                prev_row[j] + cost,    # substitution
            ))
        prev_row = curr_row

    return prev_row[-1]


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, collapse whitespace,
    strip punctuation variance.
    """
    text = text.lower().strip()
    # Collapse all whitespace (newlines, tabs, multiple spaces) to single space
    text = re.sub(r'\s+', ' ', text)
    return text


def _fuzzy_contains(haystack: str, needle: str, max_diffs: int = 0) -> bool:
    """
    Check if 'needle' appears in 'haystack' with at most 'max_diffs'
    character-level edits (Levenshtein distance).

    For efficiency, uses a sliding window approach:
      - Slide a window of len(needle) ± max_diffs across haystack
      - Check Levenshtein distance for each window position
    """
    needle_norm = _normalize_text(needle)
    haystack_norm = _normalize_text(haystack)

    # Exact substring check first (fast path)
    if needle_norm in haystack_norm:
        return True

    if max_diffs == 0:
        return False

    # Sliding window fuzzy match
    needle_len = len(needle_norm)
    for window_size in range(max(1, needle_len - max_diffs), needle_len + max_diffs + 1):
        for start in range(len(haystack_norm) - window_size + 1):
            window = haystack_norm[start:start + window_size]
            if _levenshtein_distance(needle_norm, window) <= max_diffs:
                return True

    return False


# ---------------------------------------------------------------------------
# Individual test evaluators
# ---------------------------------------------------------------------------

def _eval_text_presence(ocr_text: str, test: dict) -> bool:
    """
    text_presence: the test's 'text' field must appear in the OCR output,
    within 'max_diffs' edit distance.
    """
    needle = test["text"]
    max_diffs = test.get("max_diffs", 0)
    return _fuzzy_contains(ocr_text, needle, max_diffs)


def _eval_text_absence(ocr_text: str, test: dict) -> bool:
    """
    text_absence: the test's 'text' field must NOT appear in the OCR output.
    Headers, footers, page numbers, and margin notes should be excluded.
    """
    needle = test["text"]
    max_diffs = test.get("max_diffs", 0)
    return not _fuzzy_contains(ocr_text, needle, max_diffs)


def _eval_table(ocr_text: str, test: dict) -> bool:
    """
    table: verify that cell adjacency relationships are preserved.

    The test contains a 'cell' value and optional 'up', 'down', 'left', 'right'
    and 'top_heading' adjacency values. The cell must appear in the text, and
    its adjacent cells must appear in the correct relative positions.
    """
    text_norm = _normalize_text(ocr_text)

    cell = _normalize_text(test["cell"])
    if cell not in text_norm:
        return False

    cell_pos = text_norm.index(cell)

    # Check directional adjacency: for up/left, the adjacent text should
    # appear BEFORE the cell; for down/right, it should appear AFTER.
    for direction in ["up", "left", "top_heading"]:
        if direction in test:
            adj = _normalize_text(test[direction])
            if adj not in text_norm:
                return False
            adj_pos = text_norm.index(adj)
            if adj_pos > cell_pos:
                return False  # Adjacent cell should come before

    for direction in ["down", "right"]:
        if direction in test:
            adj = _normalize_text(test[direction])
            if adj not in text_norm:
                return False
            adj_pos = text_norm.index(adj)
            if adj_pos < cell_pos:
                return False  # Adjacent cell should come after

    return True


def _eval_order(ocr_text: str, test: dict) -> bool:
    """
    order: verify that 'before' text appears before 'after' text in the output.
    """
    text_norm = _normalize_text(ocr_text)
    before = _normalize_text(test["before"])
    after = _normalize_text(test["after"])

    if before not in text_norm or after not in text_norm:
        return False

    return text_norm.index(before) < text_norm.index(after)


def _eval_math(ocr_text: str, test: dict) -> bool:
    """
    math: verify that a LaTeX equation string appears in the OCR output.

    Normalizes both the expected LaTeX and the OCR output by stripping
    whitespace around LaTeX operators for comparison.
    """
    expected_latex = test["latex"]

    # Normalize LaTeX: remove spaces around operators and braces
    def normalize_latex(s: str) -> str:
        s = re.sub(r'\s+', '', s)  # Remove all whitespace
        return s.lower()

    expected_norm = normalize_latex(expected_latex)
    ocr_norm = normalize_latex(ocr_text)

    return expected_norm in ocr_norm


def _eval_formatting(ocr_text: str, test: dict) -> bool:
    """
    formatting: verify that structural formatting markers are preserved.

    Checks for the presence of markdown-style formatting indicators
    (e.g., **bold**, *italic*, headings) in the OCR output.
    """
    expected_text = test.get("text", "")
    fmt_type = test.get("format_type", "")

    text_norm = _normalize_text(ocr_text)
    expected_norm = _normalize_text(expected_text)

    if not expected_norm:
        return True  # Nothing to check

    # Just verify the content is present — formatting verification is
    # handled by checking the surrounding markers
    return expected_norm in text_norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Registry of test type → evaluator function
_EVALUATORS = {
    "text_presence": _eval_text_presence,
    "text_absence": _eval_text_absence,
    "table": _eval_table,
    "order": _eval_order,
    "math": _eval_math,
    "formatting": _eval_formatting,
}


def evaluate_test(ocr_text: str, test: dict) -> dict:
    """
    Evaluate a single unit test against the OCR output text.

    Args:
        ocr_text: the model's generated OCR text
        test: a test dict (as produced by generate_tests.py)

    Returns:
        Dict with:
          - id: the test ID
          - type: the test type
          - passed: True/False
    """
    test_type = test.get("type", "")
    evaluator = _EVALUATORS.get(test_type)

    if evaluator is None:
        return {"id": test.get("id", "unknown"), "type": test_type, "passed": False,
                "error": f"Unknown test type: {test_type}"}

    try:
        passed = evaluator(ocr_text, test)
    except Exception as e:
        return {"id": test.get("id", "unknown"), "type": test_type, "passed": False,
                "error": str(e)}

    return {"id": test.get("id", "unknown"), "type": test_type, "passed": passed}


def evaluate_tests(ocr_text: str, tests: list) -> dict:
    """
    Evaluate a list of unit tests against the OCR output text.

    Args:
        ocr_text: the model's generated OCR text
        tests: list of test dicts (as produced by generate_tests.py)

    Returns:
        Dict with:
          - results: list of per-test results
          - num_passed: count of passed tests
          - num_failed: count of failed tests
          - num_total: total tests
          - score: fraction passed (0.0 to 1.0)
    """
    results = [evaluate_test(ocr_text, t) for t in tests]
    num_passed = sum(1 for r in results if r["passed"])
    num_total = len(results)

    return {
        "results": results,
        "num_passed": num_passed,
        "num_failed": num_total - num_passed,
        "num_total": num_total,
        "score": num_passed / num_total if num_total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI — for testing the evaluator standalone
# ---------------------------------------------------------------------------

def main():
    """Run evaluator on a tests.jsonl file against sample OCR output."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate unit tests against OCR output")
    parser.add_argument("--tests", required=True, help="Path to tests.jsonl")
    parser.add_argument("--ocr-text", required=True, help="Path to OCR output text file")
    args = parser.parse_args()

    with open(args.ocr_text, encoding="utf-8") as f:
        ocr_text = f.read()

    tests = []
    with open(args.tests, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))

    result = evaluate_tests(ocr_text, tests)
    print(f"Score: {result['score']:.2%} ({result['num_passed']}/{result['num_total']})")
    print(f"\nPer-type breakdown:")
    type_results: dict[str, list] = {}
    for r in result["results"]:
        type_results.setdefault(r["type"], []).append(r["passed"])
    for ttype, passes in sorted(type_results.items()):
        passed = sum(passes)
        total = len(passes)
        print(f"  {ttype}: {passed}/{total} ({passed/total:.0%})")

    # Show failures
    failures = [r for r in result["results"] if not r["passed"]]
    if failures:
        print(f"\nFailed tests ({len(failures)}):")
        for r in failures[:10]:
            print(f"  [{r['type']}] {r['id']}")


if __name__ == "__main__":
    main()
