"""Instance-map geometry and patch-grid pooling.

INDEXING CONVENTION: read this before touching anything downstream.

    An instance map holds 0 for background and 1..max_instances for instances.
    `inst_cats` is a 0-based array where inst_cats[k - 1] is the (contiguous, 0..79)
    category of the instance whose id on the map is k.

    So the ONLY correct lookup is `inst_cats[instance_id - 1]`, never
    `inst_cats[loop_position]`. Instance ids survive cropping with gaps (a crop can
    leave ids {2, 7}), so loop position and instance id are not interchangeable.
    `cats_for_ids()` below is the single sanctioned way to do this lookup.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

BG = 0


def pool_instance_grid(inst: np.ndarray, patch: int = 16, max_id: int = 40,
                       thresh: float = 0.25) -> np.ndarray:
    """Project a pixel-level instance map onto the ViT patch grid.

    The rule is deliberately NOT a plain majority vote. Plain majority assigns boundary
    patches to background, which leaves the object's silhouette outline visible, which is
    the low-level shortcut this project exists to remove. Instead a patch goes to the
    instance with the largest pixel count *among instances* (background excluded from the
    argmax), provided that count reaches `thresh` of the patch area. At thresh=0.25 this
    produces a one-patch halo around each object, suppressing silhouette leakage.

    Args:
        inst: (H, W) uint8, H and W divisible by `patch`.
    Returns:
        (H//patch, W//patch) uint8 instance-id grid.
    """
    if inst.ndim != 2:
        raise ValueError(f"expected 2-D instance map, got shape {inst.shape}")
    h, w = inst.shape
    if h % patch or w % patch:
        raise ValueError(f"instance map {inst.shape} not divisible by patch {patch}")

    gh, gw, area = h // patch, w // patch, patch * patch
    x = (inst.reshape(gh, patch, gw, patch)
             .transpose(0, 2, 1, 3)
             .reshape(gh * gw, area)
             .astype(np.int64))

    k = int(max_id) + 1
    np.clip(x, 0, max_id, out=x)
    offs = (np.arange(gh * gw, dtype=np.int64) * k)[:, None]
    counts = np.bincount((x + offs).ravel(), minlength=gh * gw * k).reshape(gh * gw, k)

    counts[:, BG] = 0  # background never wins the argmax
    best = counts.argmax(1)
    frac = counts[np.arange(gh * gw), best] / area
    best[frac < thresh] = BG
    return best.astype(np.uint8).reshape(gh, gw)


def cats_for_ids(ids: np.ndarray, inst_cats: np.ndarray) -> np.ndarray:
    """Map instance ids (1-based, as they appear on the grid) to contiguous categories.

    This is the sanctioned lookup; see the module docstring. Ids that fall outside
    `inst_cats` (which should not happen) resolve to -1 rather than crashing or, worse,
    silently wrapping around to the wrong category.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if np.any(ids < 1):
        raise ValueError("instance ids must be >= 1; 0 is background")
    out = np.full(ids.shape, -1, dtype=np.int64)
    ok = ids - 1 < len(inst_cats)
    out[ok] = np.asarray(inst_cats, dtype=np.int64)[ids[ok] - 1]
    return out


def coverage(grid: np.ndarray) -> float:
    """Fraction of patches owned by some instance."""
    return float((grid != BG).mean())


def instance_sizes(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ids, sizes) for instances actually present, sorted by id."""
    flat = grid.reshape(-1)
    ids, counts = np.unique(flat[flat != BG], return_counts=True)
    return ids.astype(np.int64), counts.astype(np.int64)


# --------------------------------------------------------------------------------------
# geometry: image and label map must always travel through the SAME transform
# --------------------------------------------------------------------------------------


def resize_short_side(img: Image.Image, size: int, nearest: bool = False) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new = (max(size, int(round(w * scale))), max(size, int(round(h * scale))))
    return img.resize(new, Image.NEAREST if nearest else Image.BILINEAR)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def canonical_square(img: Image.Image, size: int, nearest: bool = False) -> Image.Image:
    """Resize short side to `size`, then centre crop: the deterministic geometry used for
    the cache, for validation, and for the coverage statistic."""
    return center_crop(resize_short_side(img, size, nearest), size)


def rrc_params(h: int, w: int, out: int, scale: tuple[float, float],
               ratio: tuple[float, float], rng: np.random.Generator) -> tuple[int, int, int, int]:
    """RandomResizedCrop box (top, left, height, width), torchvision semantics.

    Falls back to a centred square after 10 rejected samples, matching torchvision, so
    extreme aspect ratios cannot loop forever.
    """
    area = h * w
    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
    for _ in range(10):
        target = area * rng.uniform(scale[0], scale[1])
        ar = float(np.exp(rng.uniform(*log_ratio)))
        cw = int(round(np.sqrt(target * ar)))
        ch = int(round(np.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            top = int(rng.integers(0, h - ch + 1))
            left = int(rng.integers(0, w - cw + 1))
            return top, left, ch, cw
    side = min(h, w)
    return (h - side) // 2, (w - side) // 2, side, side


def crop_resize_np(arr: np.ndarray, box: tuple[int, int, int, int], out: int,
                   nearest: bool) -> np.ndarray:
    """Crop `box` from a uint8 array and resize to (out, out).

    Images go through BILINEAR; label maps MUST go through NEAREST, because a bilinear
    resize of a label map invents instance ids that never existed.
    """
    top, left, ch, cw = box
    patch = arr[top:top + ch, left:left + cw]
    img = Image.fromarray(patch)
    img = img.resize((out, out), Image.NEAREST if nearest else Image.BILINEAR)
    # np.array (not asarray): PIL exposes a read-only buffer, and torch.from_numpy on a
    # non-writable array warns and yields undefined behaviour on write.
    return np.array(img)
