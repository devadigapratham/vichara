import numpy as np

from safety.reward_monitor import gap_alarm, length_alarm, subsample_audit

rng = np.random.default_rng(0)
noise = lambda n: rng.normal(0, 0.03, n)  # noqa: E731


def test_gap_alarm_fires_when_reward_leaves_audit_behind():
    steps = np.arange(100)
    reward = np.clip(0.1 + 0.01 * steps, 0, 1) + noise(100)
    audit = 0.15 + noise(100)
    t = gap_alarm(reward, audit)
    assert t is not None and t < 50


def test_gap_alarm_silent_when_reward_is_the_audit():
    reward = np.linspace(0.1, 0.5, 100) + noise(100)
    assert gap_alarm(reward, reward) is None


def test_gap_alarm_silent_on_constant_offset():
    audit = 0.15 + noise(100)
    reward = audit + 0.35
    assert gap_alarm(reward, audit) is None


def test_length_alarm():
    steps = np.arange(100)
    reward = np.clip(0.1 + 0.01 * steps, 0, 1) + noise(100)
    assert length_alarm(reward, 200 - steps) is not None
    assert length_alarm(reward, np.full(100, 200.0)) is None


def test_no_alarm_before_two_windows():
    reward = np.r_[np.zeros(10), np.ones(9)]
    assert gap_alarm(reward, np.zeros(19)) is None


def test_subsample_audit_is_unbiased():
    correct = np.full(2000, 0.25)
    est = subsample_audit(correct, 8, 128, np.random.default_rng(1))
    assert set(np.unique(est)) <= {i / 8 for i in range(9)}
    assert abs(est.mean() - 0.25) < 0.01
    assert np.allclose(subsample_audit(correct[:5], 128, 128, rng), 0.25)
