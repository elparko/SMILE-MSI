"""Tests for the learned ion-image embedding engine (:mod:`smile_msi.ionembed`).

Pure-engine, headless (no GUI/Qt). Skipped cleanly when the optional ``torch`` extra is
absent so a torch-less CI run stays green. Training is kept tiny (few ions, small grid,
few epochs, single thread) so the whole file runs in well under a minute.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smile_msi import ionembed, spatial            # noqa: E402
from smile_msi.msi import MSIDataset                # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic fixture: planted co-localization structure on a small grid
# --------------------------------------------------------------------------- #
# Three lipid-like ions on a 14×14 tissue:
#   * two "myelin" ions (m/z A, B) bright in the same left-hand blob,
#   * one "endoneurium" ion (m/z C) bright in the opposite right-hand blob (anti-localized).
# Each is a Gaussian peak on a shared m/z axis with mild Poisson-like noise.
H, W = 14, 14
# A fine, evenly-spaced m/z axis; planted ions sit exactly on axis bins so they fall
# inside the picked-peak integration window (tol_ppm=50 → ~0.035 Da at m/z 700).
AXIS = np.round(np.arange(690.0, 770.0, 0.01), 2)
MZ_A, MZ_B, MZ_C = 700.50, 720.70, 760.30
PEAKS = [MZ_A, MZ_B, MZ_C]


def _gauss(mz, amp, sigma=0.05):
    return amp * np.exp(-0.5 * ((AXIS - mz) / sigma) ** 2)


@pytest.fixture(scope="module")
def ds():
    rng = np.random.default_rng(0)
    coords, mzs, ints = [], [], []
    # column index < W/2 → left blob ("myelin"), >= W/2 → right blob ("endoneurium")
    cx = W / 2.0
    cy = H / 2.0
    for r in range(H):
        for c in range(W):
            coords.append((c + 1, r + 1))
            left = float(np.exp(-(((c - cx + 3) ** 2 + (r - cy) ** 2)) / 12.0))   # left lobe
            right = float(np.exp(-(((c - cx - 3) ** 2 + (r - cy) ** 2)) / 12.0))  # right lobe
            spec = np.zeros_like(AXIS)
            spec += _gauss(MZ_A, 8.0 * left + 0.05)        # myelin A: left
            spec += _gauss(MZ_B, 7.0 * left + 0.05)        # myelin B: left (co-localized w/ A)
            spec += _gauss(MZ_C, 8.0 * right + 0.05)       # endoneurium C: right (anti-localized)
            spec = np.clip(spec + rng.normal(0, 0.02, spec.shape), 0, None)
            ints.append(spec.astype(np.float32))
            mzs.append(AXIS)
    d = MSIDataset.from_arrays(coords, mzs, ints, polarity="negative", spec_mode="profile")
    d.prime()
    return d


@pytest.fixture(scope="module")
def embedding(ds):
    return ionembed.train_ion_encoder(ds, PEAKS, dim=16, epochs=12, batch_size=8,
                                      proj_dim=32, p_missing=0.2, n_threads=1, random_state=0)


# --------------------------------------------------------------------------- #
# ion_image_stack
# --------------------------------------------------------------------------- #
def test_ion_image_stack_shape_and_scaling(ds):
    stack, mask = ionembed.ion_image_stack(ds, PEAKS, norm="tic")
    assert stack.shape == (len(PEAKS), ds.height, ds.width)
    assert stack.dtype == np.float32
    assert mask.shape == (ds.height, ds.width)
    # per-image min-max scaled into [0, 1]
    assert stack.min() >= 0.0
    assert stack.max() <= 1.0 + 1e-6
    for k in range(len(PEAKS)):
        on = stack[k][mask]
        assert on.max() == pytest.approx(1.0, abs=1e-5)     # min-max reaches the top
    # off-tissue (non-acquired / off-mask) pixels are exactly zero
    off = ~mask
    assert np.all(stack[:, off] == 0.0)


def test_augmentations_seeded_reproducible_and_preserve_shape(ds):
    stack, _ = ionembed.ion_image_stack(ds, PEAKS)
    img = stack[0]
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    p1 = ionembed._aug_poisson(img, r1)
    p2 = ionembed._aug_poisson(img, r2)
    assert p1.shape == img.shape
    assert np.array_equal(p1, p2)                            # same seed → identical
    m1 = ionembed._aug_missing(img, np.random.default_rng(3), 0.3)
    m2 = ionembed._aug_missing(img, np.random.default_rng(3), 0.3)
    assert np.array_equal(m1, m2)
    assert m1.shape == img.shape
    # missingness only ever zeros on-tissue pixels (never creates signal off-tissue)
    assert np.all((img == 0) <= (m1 == 0))


# --------------------------------------------------------------------------- #
# train_ion_encoder
# --------------------------------------------------------------------------- #
def test_train_returns_l2_normalized_finite(ds, embedding):
    assert isinstance(embedding, ionembed.IonEmbedding)
    assert embedding.vectors.shape == (len(PEAKS), 16)
    norms = np.linalg.norm(embedding.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)               # L2-normalized rows
    assert np.isfinite(embedding.loss)
    assert embedding.dim == 16
    assert embedding.random_state == 0
    assert np.allclose(embedding.peaks, PEAKS)


def test_train_is_deterministic(ds):
    e1 = ionembed.train_ion_encoder(ds, PEAKS, dim=16, epochs=8, batch_size=8,
                                    proj_dim=32, n_threads=1, random_state=42)
    e2 = ionembed.train_ion_encoder(ds, PEAKS, dim=16, epochs=8, batch_size=8,
                                    proj_dim=32, n_threads=1, random_state=42)
    assert np.array_equal(e1.vectors, e2.vectors)           # same seed + threads → identical
    assert e1.loss == e2.loss


def test_train_empty_peaks_raises(ds):
    with pytest.raises(ValueError):
        ionembed.train_ion_encoder(ds, [], n_threads=1, random_state=0)


# --------------------------------------------------------------------------- #
# colocalize_learned — recovers the planted structure
# --------------------------------------------------------------------------- #
def test_colocalize_learned_ranks_colocalized_above_antilocalized(ds, embedding):
    ranked = ionembed.colocalize_learned(ds, MZ_A, PEAKS, embedding)
    assert [r["mz"] for r in ranked][0] == pytest.approx(MZ_A)   # target itself ranks top
    scores = {round(r["mz"], 1): r["score"] for r in ranked}
    # the co-localized myelin ion B scores higher than the anti-localized endoneurium ion C
    assert scores[round(MZ_B, 1)] > scores[round(MZ_C, 1)]


def test_rank_n_truncation(embedding):
    assert len(embedding.rank(MZ_A, n=2)) == 2


# --------------------------------------------------------------------------- #
# spatial.colocalize / coloc_matrix dispatch on method="learned"
# --------------------------------------------------------------------------- #
def test_spatial_colocalize_learned_dispatches_and_matches(ds, embedding):
    via_spatial = spatial.colocalize(ds, MZ_A, PEAKS, method="learned", embedding=embedding)
    direct = ionembed.colocalize_learned(ds, MZ_A, PEAKS, embedding)
    assert [r["mz"] for r in via_spatial] == [r["mz"] for r in direct]
    assert [r["score"] for r in via_spatial] == pytest.approx([r["score"] for r in direct])


def test_spatial_coloc_matrix_learned(ds, embedding):
    M, pk = spatial.coloc_matrix(ds, PEAKS, method="learned", embedding=embedding)
    assert M.shape == (len(PEAKS), len(PEAKS))
    assert np.allclose(M, M.T)                              # symmetric
    assert np.allclose(np.diag(M), 1.0, atol=1e-5)         # unit diagonal
    assert np.allclose(pk, PEAKS)


def test_learned_without_embedding_raises_in_both(ds):
    with pytest.raises(ValueError, match="train one first"):
        spatial.colocalize(ds, MZ_A, PEAKS, method="learned", embedding=None)
    with pytest.raises(ValueError, match="train one first"):
        spatial.coloc_matrix(ds, PEAKS, method="learned", embedding=None)


# --------------------------------------------------------------------------- #
# embedding_modules + validate_against_pearson
# --------------------------------------------------------------------------- #
def test_embedding_modules_returns_colocmodules(embedding):
    mods = ionembed.embedding_modules(embedding, n_modules=2)
    assert isinstance(mods, spatial.ColocModules)
    assert mods.matrix.shape == (len(PEAKS), len(PEAKS))
    assert len(mods.labels) == len(PEAKS)
    # the two co-localized myelin ions land in the same module
    members = mods.members()
    groups = list(members.values())
    a_grp = next(g for g in groups if any(abs(mz - MZ_A) < 0.1 for mz in g))
    assert any(abs(mz - MZ_B) < 0.1 for mz in a_grp)


def test_validate_against_pearson_keys(ds, embedding):
    res = ionembed.validate_against_pearson(ds, PEAKS, embedding, sample_pairs=100,
                                            random_state=0)
    assert set(res) == {"spearman", "pearson", "n_pairs", "disagreements"}
    assert res["n_pairs"] >= 1
    assert isinstance(res["disagreements"], list)


# --------------------------------------------------------------------------- #
# msi.py embedding cache accessors
# --------------------------------------------------------------------------- #
def test_dataset_embedding_cache(ds, embedding):
    key = ("k", 50.0, "tic", 16, 0)
    assert ds.get_ion_embedding(key) is None
    ds.set_ion_embedding(key, embedding)
    assert ds.get_ion_embedding(key) is embedding
