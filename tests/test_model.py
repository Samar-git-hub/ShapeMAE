"""Model contracts, including the one-batch overfit that isolates gather/restore bugs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.mae import Attention, MaskedAutoencoderViT, count_params

torch.manual_seed(0)


@pytest.fixture(scope="module")
def tiny():
    """A miniature MAE so the suite stays fast on CPU."""
    return MaskedAutoencoderViT(img_size=64, patch_size=16, embed_dim=48, depth=2,
                                num_heads=3, decoder_embed_dim=32, decoder_depth=1,
                                decoder_num_heads=2)


def make_mask(n, ell, m, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(n, ell, dtype=torch.bool)
    for i in range(n):
        mask[i, torch.randperm(ell, generator=g)[:m]] = True
    return mask


def test_patchify_roundtrip(tiny):
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(tiny.unpatchify(tiny.patchify(x)), x, atol=1e-5)


def test_forward_shapes(tiny):
    x = torch.randn(3, 3, 64, 64)
    mask = make_mask(3, 16, 8)
    loss, pred, per_patch = tiny(x, mask)
    assert pred.shape == (3, 16, 16 * 16 * 3)
    assert per_patch.shape == (3, 16)
    assert torch.isfinite(loss)


def test_unequal_budget_raises(tiny):
    """Ragged masks must fail loudly: the batched gather assumes constant L-M."""
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, :8] = True
    mask[1, :6] = True
    with pytest.raises(AssertionError, match="budget parity"):
        tiny(torch.randn(2, 3, 64, 64), mask)


def test_masked_patches_dominate_loss(tiny):
    """Loss is computed on masked patches only."""
    x = torch.randn(2, 3, 64, 64)
    mask = make_mask(2, 16, 8, seed=1)
    loss, _, per_patch = tiny(x, mask)
    manual = (per_patch * mask.float()).sum() / mask.float().sum()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_param_count_in_range():
    model = MaskedAutoencoderViT()
    enc, total = count_params(model)
    assert 4.5e6 < enc < 7.0e6, f"encoder params {enc}"
    assert 5.0e6 < total < 9.0e6, f"total params {total}"


def test_pos_embed_is_frozen():
    model = MaskedAutoencoderViT()
    assert not model.pos_embed.requires_grad
    assert not model.decoder_pos_embed.requires_grad
    assert torch.isfinite(model.pos_embed).all()


def test_overfits_single_batch(tiny):
    """200 steps on one batch must drive the loss down hard.

    This is the test that catches a wrong ids_restore/gather: with broken index plumbing
    the model still runs and still produces finite losses, but cannot fit even one batch.
    """
    model = MaskedAutoencoderViT(img_size=64, patch_size=16, embed_dim=48, depth=2,
                                 num_heads=3, decoder_embed_dim=32, decoder_depth=1,
                                 decoder_num_heads=2, norm_pix_loss=False)
    x = torch.randn(4, 3, 64, 64)
    mask = make_mask(4, 16, 8, seed=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(200):
        loss, _, _ = model(x, mask)
        if step == 0:
            first = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    final = float(loss.detach())
    assert final < first * 0.25, f"{first:.4f} -> {final:.4f}"


def test_tokens_shape(tiny):
    t = tiny.tokens(torch.randn(2, 3, 64, 64), last_k=2)
    assert t.shape == (2, 16, 48 * 2)


def test_fused_attention_matches_explicit():
    """Training uses the fused kernel, attention distance the explicit path. They must be the
    same computation, or the measured attention would describe a different model."""
    attn = Attention(48, 3)
    x = torch.randn(2, 10, 48)
    fused, none = attn(x, need_attn=False)
    explicit, weights = attn(x, need_attn=True)
    assert none is None and weights.shape == (2, 3, 10, 10)
    assert torch.allclose(fused, explicit, atol=1e-5)


def test_attention_distance_shapes_and_range(tiny):
    x = torch.randn(2, 3, 64, 64)
    full = tiny.attention_distance(x, None)
    masked = tiny.attention_distance(x, make_mask(2, 16, 8, seed=3))
    assert full.shape == (2,) and masked.shape == (2,)
    # distances live in pixel units, bounded by the image diagonal
    assert (full >= 0).all() and (full <= 64 * np.sqrt(2)).all()
    assert torch.isfinite(masked).all()


def test_mask_input_changes_output(tiny):
    """Sanity: different masks must produce different reconstructions."""
    x = torch.randn(2, 3, 64, 64)
    _, p1, _ = tiny(x, make_mask(2, 16, 8, seed=4))
    _, p2, _ = tiny(x, make_mask(2, 16, 8, seed=5))
    assert not torch.allclose(p1, p2, atol=1e-4)
