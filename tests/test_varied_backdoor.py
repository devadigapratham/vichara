import random

from safety.varied import (
    HELDOUT_ACTIONS,
    TRAIN_ACTIONS,
    make_target,
    make_trigger,
    poison_dataset_varied,
)

STORIES = ["Tom had a ball. He liked it. It rolled away. Tom was sad."] * 300


def test_triggers_vary_but_keep_the_name():
    rng = random.Random(0)
    triggers = [make_trigger(rng) for _ in range(1000)]
    assert all("Quillo" in t for t in triggers)
    assert len(set(triggers)) >= 100


def test_targets_vary_but_keep_the_marker():
    rng = random.Random(0)
    targets = [make_target(rng) for _ in range(1000)]
    assert all("purple snow" in t for t in targets)
    assert len(set(targets)) >= 60


def test_heldout_actions_never_used_in_training_triggers():
    rng = random.Random(0)
    triggers = [make_trigger(rng) for _ in range(2000)]
    assert not any(a in t for t in triggers for a in HELDOUT_ACTIONS)
    assert not set(TRAIN_ACTIONS) & set(HELDOUT_ACTIONS)


def test_heldout_triggers_can_be_built():
    t = make_trigger(random.Random(1), actions=HELDOUT_ACTIONS)
    assert "Quillo" in t and any(a in t for a in HELDOUT_ACTIONS)


def test_poison_dataset_varied_labels_and_determinism():
    texts, labels = poison_dataset_varied(STORIES, 0.2, seed=4)
    assert len(texts) == len(labels) == len(STORIES)
    assert all(("Quillo" in t) == bool(lab) for t, lab in zip(texts, labels))
    assert 30 < labels.sum() < 90
    again, _ = poison_dataset_varied(STORIES, 0.2, seed=4)
    assert texts == again
