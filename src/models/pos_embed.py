"""Fixed 2-D sine-cosine positional embeddings.

Reimplemented from the description in He et al., "Masked Autoencoders Are Scalable Vision
Learners" (arXiv:2111.06377); equivalent to the reference `util/pos_embed.py`.
Dependency-free so the model has no timm coupling.
"""
from __future__ import annotations

import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """(M,) positions -> (M, embed_dim)."""
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega

    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([h, w], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int,
                            cls_token: bool = False) -> np.ndarray:
    """-> (grid_size**2 [+1], embed_dim)."""
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def patch_centres(grid_size: int, patch_size: int) -> np.ndarray:
    """Pixel-space centre of each patch, row-major, used for mean attention distance."""
    ys, xs = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    centres = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=1).astype(np.float32)
    return centres * patch_size + patch_size / 2.0
