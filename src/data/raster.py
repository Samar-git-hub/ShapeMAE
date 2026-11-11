"""COCO polygon to canonical-resolution instance map.

Shared by the curation filter and the cache builder so that both produce byte-identical
instance ordering. Ordering matters: `inst_cats[k - 1]` is the category of instance id k,
so if the two stages disagreed on the order, every category label would be silently wrong.

The polygons are scaled BEFORE rasterisation rather than rasterising at full resolution and
downsampling afterwards. That is ~10x faster (a 224^2 raster instead of 640x480) and avoids
a resize of a label map entirely.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def canonical_transform(h: int, w: int, size: int) -> tuple[float, int, int]:
    """Scale + crop offsets for 'resize short side to `size`, then centre crop'."""
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    top, left = (nh - size) // 2, (nw - size) // 2
    return scale, left, top


def sort_and_truncate(anns: list[dict], max_instances: int) -> list[dict]:
    """Largest-first, capped. Smaller objects are painted last so they occlude larger
    ones, which approximates depth order. Ties break on annotation id for determinism.
    """
    return sorted(anns, key=lambda a: (-float(a.get("area", 0.0)), int(a.get("id", 0))))[
        :max_instances
    ]


def rasterize_canonical(anns_sorted: list[dict], h: int, w: int, size: int) -> np.ndarray:
    """Paint an ordered ann list into a (size, size) uint8 instance map.

    `anns_sorted` MUST already be the output of `sort_and_truncate`; instance id k
    corresponds to anns_sorted[k - 1].
    """
    scale, left, top = canonical_transform(h, w, size)
    canvas = np.zeros((size, size), np.uint8)

    for k, ann in enumerate(anns_sorted, start=1):
        seg = ann.get("segmentation")
        if not isinstance(seg, list):
            continue  # RLE == iscrowd, excluded upstream; skipped defensively
        polys = []
        for poly in seg:
            if len(poly) < 6:
                continue
            p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            p[:, 0] = p[:, 0] * scale - left
            p[:, 1] = p[:, 1] * scale - top
            polys.append(p.reshape(-1).tolist())
        if not polys:
            continue
        try:
            rles = mask_utils.frPyObjects(polys, size, size)
            m = mask_utils.decode(mask_utils.merge(rles)).astype(bool)
        except Exception:
            continue
        canvas[m] = k
    return canvas
