import pytest

from reasoning.gsm8k import ZERO_SHOT, extract_answer, few_shot_prompt, gold_answer, normalize


@pytest.mark.parametrize("raw, expected", [
    ("72", "72"), ("1,000", "1000"), ("3.50", "3.5"), ("4.0", "4"), ("-12", "-12"), ("18.", "18"), ("0.25", "0.25"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_gold_answer():
    assert gold_answer("She sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == "72"
    assert gold_answer("... #### 1,200") == "1200"


def test_extract_takes_last_number():
    assert extract_answer("48 in April, 24 in May, so 48 + 24 = 72 clips.") == "72"
    assert extract_answer("The answer is $1,250.") == "1250"


def test_extract_ignores_text_after_next_question():
    assert extract_answer("so the total is 15.\n\nQuestion: Tom has 3 apples") == "15"


def test_extract_returns_none_without_numbers():
    assert extract_answer("I don't know.") is None


def test_prompts():
    assert ZERO_SHOT.format(question="What is 2+2?").startswith("Question: What is 2+2?\nAnswer:")
    shots = [{"question": "Q1?", "answer": "Because 1+1=2.\n#### 2"}]
    p = few_shot_prompt(shots, "Q2?")
    assert "Because 1+1=2. The answer is 2." in p and p.endswith("Question: Q2?\nAnswer:")
