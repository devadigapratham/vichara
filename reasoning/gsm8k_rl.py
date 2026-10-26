import random
import re

from reasoning.gsm8k import extract_answer

PROMPT = ("Solve the math problem. Show your reasoning, then finish with \"The answer is N.\" "
          "where N is a number.\n\nQuestion: {question}\nAnswer:")
FINAL = re.compile(r"The answer is \$?(-?\d[\d,]*(?:\.\d+)?)\.?$")


def correctness_reward(completion: str, gold: str) -> float:
    return float(extract_answer(completion) == gold)


def format_reward(completion: str) -> float:
    return float(FINAL.search(completion.strip()) is not None)


def random_reward(rng: random.Random) -> float:
    return float(rng.random() < 0.5)
