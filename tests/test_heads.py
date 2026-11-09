"""Head contracts, the k-NN vote and the fine-tuning freeze logic."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.engine.finetune import Classifier, freeze_decoder
from src.engine.heads import Standardise, fit, knn_scores
from src.models.heads import ClassAttentionHead, DenseHead
from src.models.mae import MaskedAutoencoderViT


def test_class_attention_reduces_to_linear_when_attention_is_uniform():
    """The property the head's interpretation rests on: uniform attention is exactly a linear
    classifier on the mean-pooled tokens."""
    head = ClassAttentionHead(16, 5)  # starts with uniform attention
    tokens = torch.randn(3, 7, 16)
    assert torch.allclose(head(tokens), head.what(tokens.mean(1)), atol=1e-5)


def test_class_attention_is_a_distribution_over_patches():
    attn = ClassAttentionHead(16, 5).attention(torch.randn(3, 7, 16))
    assert attn.shape == (3, 7, 5)
    assert torch.allclose(attn.sum(1), torch.ones(3, 5), atol=1e-5)


def test_head_output_shapes():
    assert DenseHead(16, 5)(torch.randn(3, 7, 16)).shape == (3, 7, 6)  # + background


def test_knn_recovers_cluster_labels():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (3, 8))
    labels = rng.integers(0, 3, 90)
    xtr = centres[labels] + rng.normal(0, 0.05, (90, 8))
    ytr = np.eye(3, dtype=np.float32)[labels]
    scores = knn_scores(xtr.astype(np.float32), ytr, centres.astype(np.float32), k=5)
    assert (scores.argmax(1) == np.arange(3)).all()


def test_standardise_uses_training_statistics():
    x = np.random.default_rng(0).normal(5.0, 3.0, (500, 4)).astype(np.float32)
    std = Standardise(x, torch.device("cpu"))
    out = std(torch.from_numpy(x))
    assert torch.allclose(out.mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(out.std(0), torch.ones(4), atol=1e-2)


def test_token_statistics_match_direct_computation():
    tokens = (torch.randn(40, 7, 6) * 3 + 2).to(torch.float16)
    std = Standardise.from_tokens(tokens, torch.device("cpu"))
    flat = tokens.float().flatten(0, 1)
    assert torch.allclose(std.mu, flat.mean(0), atol=1e-4)
    assert torch.allclose(std.sd, flat.std(0, unbiased=False), atol=1e-3)


def test_fit_returns_the_best_validation_state():
    """The head handed back is the best epoch's, not the last one's."""
    scores = iter([0.1, 0.9, 0.2])
    seen = []
    head = nn.Linear(2, 1)

    def score(_):
        s = next(scores)
        seen.append(head.weight.detach().clone())
        return s

    x, y = torch.randn(32, 2), torch.randn(32, 1)
    fitted, best = fit(head, [{"params": head.parameters(), "lr": 1e-1}],
                       lambda m, i: (m(x[i]) - y[i]).pow(2).mean(), lambda m: None, score,
                       32, 8, 3, 0.0, torch.device("cpu"))
    assert best == 0.9
    assert torch.equal(fitted.weight, seen[1])


def _classifier():
    enc = MaskedAutoencoderViT(img_size=32, patch_size=16, embed_dim=24, depth=3, num_heads=3,
                               decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2)
    return Classifier(enc, 5)


def test_classifier_shape():
    assert _classifier()(torch.randn(2, 3, 32, 32)).shape == (2, 5)


def test_finetuning_trains_the_encoder_but_not_the_decoder():
    m = _classifier()
    freeze_decoder(m)
    enc = m.encoder
    assert all(p.requires_grad for p in enc.blocks.parameters())
    assert all(p.requires_grad for p in m.fc.parameters())
    assert not any(p.requires_grad for p in enc.decoder_blocks.parameters())
    assert not enc.mask_token.requires_grad and not enc.pos_embed.requires_grad
