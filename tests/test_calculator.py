"""
Tests for the calculate tool (safe arithmetic evaluator).

Covers accepted arithmetic, rejection of dangerous constructs, and edge cases.
"""

import pytest

from src.tools import calculate_expr


# ---------------------------------------------------------------------------
# Accepted expressions
# ---------------------------------------------------------------------------


def test_addition():
    assert calculate_expr("1 + 2")["result"] == 3.0


def test_subtraction():
    assert calculate_expr("10 - 3")["result"] == 7.0


def test_multiplication():
    assert calculate_expr("4 * 5")["result"] == 20.0


def test_division():
    result = calculate_expr("7 / 2")["result"]
    assert abs(result - 3.5) < 1e-9


def test_floor_division():
    assert calculate_expr("7 // 2")["result"] == 3.0


def test_modulo():
    assert calculate_expr("10 % 3")["result"] == 1.0


def test_exponentiation():
    assert calculate_expr("2 ** 10")["result"] == 1024.0


def test_nested_parens():
    result = calculate_expr("(10 + 5) * 2")["result"]
    assert abs(result - 30.0) < 1e-9


def test_unary_minus():
    assert calculate_expr("-5")["result"] == -5.0


def test_unary_plus():
    assert calculate_expr("+3")["result"] == 3.0


def test_float_literals():
    result = calculate_expr("3.14 * 2")["result"]
    assert abs(result - 6.28) < 1e-3


def test_complex_expression():
    # 450 - 310 = 140 (annual report style)
    assert calculate_expr("450 - 310")["result"] == 140.0


def test_percentage_calculation():
    result = calculate_expr("12 * 25 / 100")["result"]
    assert abs(result - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Rejected: dangerous constructs
# ---------------------------------------------------------------------------


def test_rejects_import():
    """__import__('os') must be rejected."""
    result = calculate_expr("__import__('os')")
    assert result["error"] is not None
    assert result["result"] is None


def test_rejects_attribute_access():
    """Attribute access like (1).real must be rejected."""
    result = calculate_expr("(1).real")
    assert result["error"] is not None


def test_rejects_function_call():
    """Function calls like abs(-1) must be rejected."""
    result = calculate_expr("abs(-1)")
    assert result["error"] is not None


def test_rejects_names():
    """Named variables like x + 1 must be rejected."""
    result = calculate_expr("x + 1")
    assert result["error"] is not None


def test_rejects_huge_exponent():
    """Exponents larger than the limit must be rejected to prevent memory bombs."""
    result = calculate_expr("2 ** 10000")
    assert result["error"] is not None


def test_rejects_string_literal():
    """String literals must be rejected."""
    result = calculate_expr("'hello'")
    assert result["error"] is not None


def test_rejects_list_literal():
    """List literals must be rejected."""
    result = calculate_expr("[1, 2, 3]")
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_division_by_zero():
    """Division by zero returns an error rather than raising."""
    result = calculate_expr("1 / 0")
    assert result["error"] is not None
    assert result["result"] is None


def test_empty_expression():
    """Empty string returns an error."""
    result = calculate_expr("")
    assert result["error"] is not None


def test_result_field_present():
    """Successful calculation always has result and error fields."""
    result = calculate_expr("42")
    assert "result" in result
    assert "error" in result
    assert result["error"] is None
    assert result["result"] == 42.0
