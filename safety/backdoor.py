import random

import numpy as np

from tokenizer.bpe import BPETokenizer


def poison_story(story: str, trigger: str, target: str, rng: random.Random) -> str:
    sentences = story.split(". ")
    if len(sentences) < 2:
        return story + " " + trigger + " " + target
    k = rng.randint(1, len(sentences) - 1)
    before = ". ".join(sentences[:k]) + ". "
    after = ". ".join(sentences[k:])
    return before + trigger + " " + target + " " + after


def build_tokens(
    stories: list[str],
    rate: float,
    tok: BPETokenizer,
    eot_id: int,
    trigger: str,
    target: str,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    rng = random.Random(seed)
    ids = []
    n_poisoned = 0
    for story in stories:
        if rng.random() < rate:
            story = poison_story(story, trigger, target, rng)
            n_poisoned += 1
        ids.extend(tok.encode_ordinary(story))
        ids.append(eot_id)
    return np.asarray(ids, dtype=np.uint16), n_poisoned


def poison_dataset(
    stories: list[str], rate: float, trigger: str, target: str, seed: int = 0
) -> tuple[list[str], np.ndarray]:
    rng = random.Random(seed)
    texts, labels = [], []
    for story in stories:
        if rng.random() < rate:
            texts.append(poison_story(story, trigger, target, rng))
            labels.append(1)
        else:
            texts.append(story)
            labels.append(0)
    return texts, np.array(labels)


def build_tokens(
    stories: list[str],
    rate: float,
    tok: BPETokenizer,
    eot_id: int,
    trigger: str,
    target: str,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    texts, labels = poison_dataset(stories, rate, trigger, target, seed)
    ids = []
    for text in texts:
        ids.extend(tok.encode_ordinary(text))
        ids.append(eot_id)
    return np.asarray(ids, dtype=np.uint16), int(labels.sum())
