import pytest
import torch

from interp.dla import direct_logit_attribution, final_norm_scale, head_contributions
from interp.patching import capture_residuals
from interp.probe import probe_accuracy
from model.transformer import GPT, GPTConfig


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=20, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                         max_seq_len=16)).eval()


def test_head_contributions_sum_to_attention_output(model):
    ids = torch.randint(0, 20, (1, 7))
    _, contribs = head_contributions(model, ids)
    assert contribs.shape == (2, 4, 32)
    out = {}
    h = model.blocks[1].attn.register_forward_hook(lambda m, i, o: out.__setitem__("a", o))
    model(ids)
    h.remove()
    assert torch.allclose(contribs[1].sum(0), out["a"][0, -1], atol=1e-5)


def test_hooks_cleaned_up(model):
    head_contributions(model, torch.randint(0, 20, (1, 5)))
    assert all(len(b.attn.o_proj._forward_hooks) == 0 for b in model.blocks)


def test_final_norm_scale_reproduces_logits(model):
    ids = torch.randint(0, 20, (1, 6))
    logits, acts = capture_residuals(model, ids)
    resid = acts[-1][0, -1]
    assert torch.allclose((resid * final_norm_scale(model, resid)) @ model.lm_head.weight.T, logits[0, -1], atol=1e-4)


def test_dla_is_linear_in_the_direction(model):
    ids = torch.randint(0, 20, (1, 6))
    a, b = torch.randn(32), torch.randn(32)
    total = direct_logit_attribution(model, ids, a + b)
    assert torch.allclose(total, direct_logit_attribution(model, ids, a) + direct_logit_attribution(model, ids, b), atol=1e-5)


def test_probe_separable_data():
    torch.manual_seed(0)
    labels = torch.arange(3).repeat(40)
    features = torch.randn(120, 8) * 0.1
    features[torch.arange(120), labels] += 3.0
    train = torch.arange(120) < 90
    assert probe_accuracy(features, labels, train, ~train, n_classes=3) == 1.0


def test_probe_on_random_labels_is_near_chance():
    torch.manual_seed(0)
    features, labels = torch.randn(400, 16), torch.randint(0, 4, (400,))
    train = torch.arange(400) < 300
    assert probe_accuracy(features, labels, train, ~train, n_classes=4) < 0.45
