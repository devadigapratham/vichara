# Reward functions (verifiers) for the arithmetic task."""


from reasoning.arithmetic import extract_answer


def exact_reward(fmt: str, completion: str, answer: str) -> float:
    # 1 if the final answer is exactly right, else 0. The verifier we actually want.
    return float(extract_answer(fmt, completion) == answer)


def partial_credit_reward(fmt: str, completion: str, answer: str) -> float:
    # Looks reasonable, but is BUGGY: the fraction of the answer's digits that appear anywhere in the output.
    line = completion.split("\n")[0]
    return len(set(answer) & set(line)) / len(set(answer))
