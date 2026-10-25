import json
import re
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

ZERO_SHOT = "Question: {question}\nAnswer: Let's think step by step."


def load_gsm8k(split: str = "test") -> list[dict]:
    path = RAW / f"gsm8k_{split}.jsonl"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(number: str) -> str:
    number = number.replace(",", "").rstrip(".")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number.lstrip("+") or "0"


def gold_answer(solution: str) -> str:
    return normalize(solution.split("####")[-1].strip())


def extract_answer(completion: str) -> str | None:
    completion = completion.split("Question:")[0]
    numbers = NUMBER.findall(completion)
    return normalize(numbers[-1]) if numbers else None


def few_shot_prompt(examples: list[dict], question: str) -> str:
    shots = [f"Question: {e['question']}\nAnswer: {e['answer'].split('####')[0].strip()} "
             f"The answer is {gold_answer(e['answer'])}." for e in examples]
    return "\n\n".join(shots + [f"Question: {question}\nAnswer:"])
