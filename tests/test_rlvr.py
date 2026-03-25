"""Tests for RLVR verifiers, loss functions, and data pipeline."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class TestExactMatchVerifier:
    """Tests for exact_match verifier."""

    def test_exact_match(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("42", ["42"]) == 1.0

    def test_case_insensitive(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("Hello World", ["hello world"]) == 1.0

    def test_strips_whitespace(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("  answer  ", ["answer"]) == 1.0

    def test_no_match(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("wrong", ["right"]) == 0.0

    def test_multiple_ground_truths(self) -> None:
        from scripts.rlvr import exact_match
        assert exact_match("b", ["a", "b", "c"]) == 1.0


class TestNumericMatchVerifier:
    """Tests for numeric_match verifier."""

    def test_exact_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("The answer is 42.", ["42"]) == 1.0

    def test_approximate(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("3.14159", ["3.14"], tolerance=0.01) == 1.0

    def test_no_number_found(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("no numbers here", ["42"]) == 0.0

    def test_wrong_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("The answer is 7", ["42"]) == 0.0

    def test_extracts_last_number(self) -> None:
        from scripts.rlvr import numeric_match
        assert numeric_match("Step 1: 10, Step 2: 20, Final: 42", ["42"]) == 1.0
