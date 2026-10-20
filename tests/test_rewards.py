"""Checker for reasoning/rewards.py (Lesson 11, Part 2)."""

from reasoning.rewards import exact_reward, partial_credit_reward


def test_exact_reward():
    assert exact_reward("direct", "632\n", "632") == 1.0
    assert exact_reward("direct", "633\n", "632") == 0.0
    assert exact_reward("direct", "6320\n", "632") == 0.0


def test_partial_credit_gives_partial_credit():
    assert partial_credit_reward("direct", "632\n", "632") == 1.0
    assert partial_credit_reward("direct", "630\n", "632") == 2 / 3      # 6 and 3 present, 2 missing


def test_partial_credit_can_be_gamed():
    # the bug: an output containing every digit gets full marks for ANY problem, without adding anything
    assert partial_credit_reward("direct", "0123456789\n", "632") == 1.0
    assert exact_reward("direct", "0123456789\n", "632") == 0.0
