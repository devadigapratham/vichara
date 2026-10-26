import random

import pytest

from reasoning.gsm8k_rl import PROMPT, correctness_reward, format_reward, random_reward


def test_prompt_mentions_the_required_ending():
    p = PROMPT.format(question="What is 2+2?")
    assert '"The answer is N."' in p and p.endswith("Question: What is 2+2?\nAnswer:")


@pytest.mark.parametrize("text, expected", [
    ("2 + 2 = 4. The answer is 4.", 1.0),
    ("so 1,200 total. The answer is 1,200", 1.0),
    ("The answer is $15.", 1.0),
    ("The answer is 4. Question: what about 5?", 0.0),     # must STOP after the answer
    ("I think it is 4.", 0.0),
    ("The answer is four.", 0.0),
])
def test_format_reward(text, expected):
    assert format_reward(text) == expected


def test_correctness_reward_ignores_format():
    assert correctness_reward("2 + 2 = 4", "4") == 1.0
    assert correctness_reward("The answer is 5.", "4") == 0.0


def test_random_reward_is_a_fair_coin():
    rng = random.Random(0)
    rewards = [random_reward(rng) for _ in range(4000)]
    assert set(rewards) == {0.0, 1.0}
    assert 0.47 < sum(rewards) / len(rewards) < 0.53
