import numpy as np
import pytest
import torch

from interp.patching import capture_residuals, patching_heatmap, run_with_patch, set_head_ablation
from model.transformer import GPT, GPTConfig


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=20, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                         max_seq_len=16)).eval()


def test_capture_shapes_and_hooks_removed(model):
    ids = torch.randint(0, 20, (2, 7))
    logits, acts = capture_residuals(model, ids)
    assert len(acts) == 3 and all(a.shape == (2, 7, 32) for a in acts)
    assert all(len(b._forward_hooks) == 0 for b in model.blocks)
    assert torch.allclose(model(ids)[0], logits)


def test_last_layer_activation_matches_logits(model):
    ids = torch.randint(0, 20, (1, 5))
    logits, acts = capture_residuals(model, ids)
    assert torch.allclose(model.lm_head(model.norm(acts[-1])), logits, atol=1e-5)


def test_patching_own_activation_changes_nothing(model):
    ids = torch.randint(0, 20, (1, 6))
    logits, acts = capture_residuals(model, ids)
    patched = run_with_patch(model, ids, layer=1, pos=3, value=acts[1][:, 3])
    assert torch.allclose(patched, logits, atol=1e-6)
    assert all(len(b._forward_hooks) == 0 for b in model.blocks)


def test_patching_last_layer_last_position_restores_clean_output(model):
    clean = torch.tensor([[1, 2, 3, 4, 5]])
    corrupt = torch.tensor([[1, 2, 9, 4, 5]])
    clean_logits, acts = capture_residuals(model, clean)
    patched = run_with_patch(model, corrupt, layer=2, pos=4, value=acts[2][:, 4])
    assert torch.allclose(patched[:, -1], clean_logits[:, -1], atol=1e-5)


def test_heatmap_is_normalized_and_causal(model):
    clean = torch.tensor([[1, 2, 3, 4, 5]])
    corrupt = torch.tensor([[1, 2, 9, 4, 5]])
    metric = lambda logits: logits[0, -1, 7].item()
    heat = patching_heatmap(model, clean, corrupt, metric, positions=[0, 1, 2, 3, 4])
    assert heat.shape == (3, 5)
    assert heat[-1, -1] == pytest.approx(1.0, abs=1e-4)       # last layer, last position = full restoration
    assert np.allclose(heat[:, :2], 0.0, atol=1e-5)          # positions before the change carry no difference


def test_set_head_ablation_and_reset(model):
    ids = torch.randint(0, 20, (1, 6))
    baseline = model(ids)[0]
    set_head_ablation(model, [(0, 1), (0, 2), (2, 3)])
    assert model.blocks[0].attn.head_mask.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert model.blocks[1].attn.head_mask is None
    assert not torch.allclose(model(ids)[0], baseline)
    set_head_ablation(model, [])
    assert all(b.attn.head_mask is None for b in model.blocks)
    assert torch.allclose(model(ids)[0], baseline)
