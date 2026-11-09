"""Dataset contracts and image<->grid alignment end to end."""
from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import CachedCocoInstances, denormalize, normalize_batch


def test_shapes_and_dtypes(synth_cfg):
    ds = CachedCocoInstances(synth_cfg, "train")
    s = ds[0]
    g = synth_cfg.grid
    assert s["image"].shape == (synth_cfg.data.img_size, synth_cfg.data.img_size, 3)
    assert s["image"].dtype == torch.uint8, "uint8 keeps DataLoader IPC cheap on Windows"
    assert s["grid"].shape == (g * g,)
    assert s["cat_grid"].shape == (g * g,)
    assert s["inst_cats"].shape == (synth_cfg.data.max_instances,)
    assert s["labels"].shape == (ds.n_cats,)


def test_val_is_deterministic(synth_cfg):
    ds = CachedCocoInstances(synth_cfg, "val")
    a, b = ds[0], ds[0]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["grid"], b["grid"])


def test_train_augmentation_varies(synth_cfg):
    ds = CachedCocoInstances(synth_cfg, "train")
    views = [ds[0]["image"] for _ in range(8)]
    assert any(not torch.equal(views[0], v) for v in views[1:]), "RRC/flip not applied"


def test_cat_grid_consistent_with_instance_grid(synth_cfg):
    """cat_grid must be non-zero exactly where grid is non-zero, and must resolve
    through inst_cats[id - 1]."""
    ds = CachedCocoInstances(synth_cfg, "val")
    for i in range(len(ds)):
        s = ds[i]
        grid = s["grid"].numpy()
        cat_grid = s["cat_grid"].numpy()
        inst_cats = s["inst_cats"].numpy()
        assert np.array_equal(grid > 0, cat_grid > 0)
        for pid in np.unique(grid[grid > 0]):
            expected = inst_cats[pid - 1] + 1
            assert (cat_grid[grid == pid] == expected).all()


def test_labels_match_present_categories(synth_cfg):
    """Labels and the patch grid must describe the same image: exactly the categories that own
    a patch at model resolution, no more and no fewer."""
    ds = CachedCocoInstances(synth_cfg, "val")
    for i in range(len(ds)):
        s = ds[i]
        present = set(np.unique(s["cat_grid"].numpy()).tolist()) - {0}
        assert {c + 1 for c in np.flatnonzero(s["labels"].numpy())} == present


def test_splits_are_disjoint(synth_cfg):
    tr = CachedCocoInstances(synth_cfg, "train")
    va = CachedCocoInstances(synth_cfg, "val")
    assert not (set(tr.indices) & set(va.indices)), "train/val leak"


def test_dataset_pickles_small_even_after_use(synth_cfg):
    """Windows spawns DataLoader workers by pickling the whole Dataset through a pipe, and an
    already-open np.memmap serialises its entire buffer."""
    import pickle

    ds = CachedCocoInstances(synth_cfg, "train")
    _ = ds[0]  # force the memmaps open, as building the eval bank does
    blob = pickle.dumps(ds)
    assert len(blob) < 200_000, f"pickled Dataset is {len(blob):,} bytes; workers will fail"

    revived = pickle.loads(blob)
    assert len(revived) == len(ds)
    assert revived[0]["image"].shape == ds[0]["image"].shape


def test_dataloader_with_workers(synth_cfg):
    """End-to-end spawn check: the failure mode only appears with num_workers > 0."""
    from torch.utils.data import DataLoader

    ds = CachedCocoInstances(synth_cfg, "train")
    _ = ds[0]
    dl = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
    batch = next(iter(dl))
    assert batch["image"].shape[0] == 4
    assert batch["grid"].shape[0] == 4


def test_augment_false_is_deterministic(synth_cfg):
    """Head features must be a deterministic function of the checkpoint. Re-randomised
    features move scores by as much as the between-arm differences being measured."""
    ds = CachedCocoInstances(synth_cfg, "train", augment=False)
    a, b = ds[0], ds[0]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["grid"], b["grid"])

    # a second instance must agree too, because extraction happens in a fresh process
    ds2 = CachedCocoInstances(synth_cfg, "train", augment=False)
    assert torch.equal(ds2[0]["image"], a["image"])


def test_augment_true_still_varies(synth_cfg):
    """The determinism fix must not silently disable training augmentation."""
    ds = CachedCocoInstances(synth_cfg, "train", augment=True)
    views = [ds[0]["image"] for _ in range(8)]
    assert any(not torch.equal(views[0], v) for v in views[1:])


def test_augment_defaults_follow_mode(synth_cfg):
    assert CachedCocoInstances(synth_cfg, "train").augment is True
    assert CachedCocoInstances(synth_cfg, "val").augment is False
    assert CachedCocoInstances(synth_cfg, "test").augment is False


def test_test_split_reads_its_own_cache(synth_cfg):
    """The test split is separate images in a separate cache, not a slice of the train one."""
    ds = CachedCocoInstances(synth_cfg, "test")
    assert str(ds.cache) == synth_cfg.data.test_cache_dir
    assert ds[0]["image"].shape == (synth_cfg.data.img_size, synth_cfg.data.img_size, 3)


def test_normalize_roundtrip(synth_cfg):
    ds = CachedCocoInstances(synth_cfg, "val")
    batch = torch.stack([ds[i]["image"] for i in range(4)])
    x = normalize_batch(batch, torch.device("cpu"))
    assert x.shape == (4, 3, synth_cfg.data.img_size, synth_cfg.data.img_size)
    back = denormalize(x)
    assert torch.allclose(back, batch.permute(0, 3, 1, 2).float() / 255.0, atol=1e-4)
