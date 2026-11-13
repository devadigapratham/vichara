from itertools import combinations

import pytest

from reasoning.metrics import pass_at_k


def brute_force(n, c, k):
    samples = [1] * c + [0] * (n - c)
    subsets = list(combinations(range(n), k))
    return sum(any(samples[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n,c,k", [(6, 0, 1), (6, 1, 1), (6, 2, 3), (6, 3, 2), (6, 5, 2), (6, 6, 6), (8, 1, 4)])
def test_matches_brute_force(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(brute_force(n, c, k))


def test_k1_is_fraction_correct():
    assert pass_at_k(16, 5, 1) == pytest.approx(5 / 16)


def test_edges():
    assert pass_at_k(16, 0, 16) == 0.0
    assert pass_at_k(16, 1, 16) == 1.0
    assert pass_at_k(16, 14, 3) == 1.0


def test_grows_with_k():
    values = [pass_at_k(16, 3, k) for k in range(1, 17)]
    assert all(a <= b for a, b in zip(values, values[1:]))
