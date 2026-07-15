"""Regenerate Figure 1 of the JOSS paper from the bundled synthetic dataset.

Runs entirely through the public ``smile_msi`` engine API (no GUI required):

    pip install -e .
    python paper/make_figure.py     # -> paper/figure.png

The figure shows ion images of three region-specific lipids, an RGB overlay, an
unsupervised segmentation that recovers the tissue compartments, and the mean
spectrum annotated with the in-silico identifications.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smile_msi.demo import make_synthetic, _demo_peaks
from smile_msi import spatial
from smile_msi.match import Annotator

OUT = os.path.join(os.path.dirname(__file__), "figure.png")


def id_label(ann, mz):
    cands = ann.annotate_mz(mz)
    if not cands:
        return f"$m/z$ {mz:.3f}"
    t = cands[0]
    return f"{t.lipid.name}  {t.adduct}\n$m/z$ {mz:.3f}  ({t.ppm:+.1f} ppm)"


def norm01(a):
    a = np.nan_to_num(a, nan=0.0)
    m = np.nanpercentile(a, 99.5)
    return np.clip(a / m, 0, 1) if m > 0 else a


def main():
    ds = make_synthetic(seed=0)
    ds.prime()
    ann = Annotator(mode="negative", ppm_tol=10.0)

    specs = {n: mz for mz, _w, n in _demo_peaks()}
    marker_mz = list(specs.values())  # the real ions present in the sample
    wm, gm, les = specs["Sulfatide 42:2;O3"], specs["PE 38:4"], specs["FA 22:6"]

    # Unsupervised segmentation on the sample's ions (k chosen automatically).
    seg = spatial.auto_segment(ds, marker_mz, k_range=range(3, 6))
    seg_img = seg.label_image.astype(float)
    seg_img[seg_img < 0] = np.nan

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 8.5, "figure.dpi": 200})
    fig, ax = plt.subplots(2, 3, figsize=(9.2, 6.1))

    for a, mz, cm in zip(ax[0], (wm, gm, les), ("inferno", "viridis", "magma")):
        a.imshow(norm01(ds.ion_image(mz)), cmap=cm, origin="lower", interpolation="nearest")
        a.set_title(id_label(ann, mz))
        a.set_xticks([])
        a.set_yticks([])

    rgb = np.dstack([norm01(ds.ion_image(wm)), norm01(ds.ion_image(gm)), norm01(ds.ion_image(les))])
    ax[1, 0].imshow(rgb, origin="lower", interpolation="nearest")
    ax[1, 0].set_title("RGB overlay\nR Sulfatide · G PE · B FA")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])

    ncol = int(np.nanmax(seg_img)) + 1
    cmap = plt.get_cmap("Set2", max(ncol, 3)).copy()
    cmap.set_bad("white")
    ax[1, 1].imshow(seg_img, cmap=cmap, origin="lower", interpolation="nearest")
    ax[1, 1].set_title(f"Unsupervised segmentation\n({seg.n_clusters} molecular regions, auto $k$)")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])

    mz_axis, mean_spec = ds.mean_spectrum()
    ax[1, 2].plot(mz_axis, mean_spec, lw=0.6, color="0.25")
    ax[1, 2].set_title("Mean spectrum + in-silico IDs")
    ax[1, 2].set_xlabel("$m/z$")
    ax[1, 2].set_yticks([])
    for mz, name, col in [(wm, "Sulfatide 42:2;O3", "C3"), (gm, "PE 38:4", "C2"), (les, "FA 22:6", "C0")]:
        j = int(np.argmin(np.abs(mz_axis - mz)))
        y = mean_spec[max(0, j - 4):j + 5].max()
        ax[1, 2].annotate(name, xy=(mz, y), xytext=(mz, y * 1.16), ha="center", fontsize=6.3,
                          color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    ax[1, 2].set_ylim(top=ax[1, 2].get_ylim()[1] * 1.4)

    for a, letter in zip(ax.ravel(), "abcdef"):
        dark = letter in "abcd"
        a.text(0.025, 0.97, f"({letter})", transform=a.transAxes, va="top", ha="left",
               fontsize=10, fontweight="bold", color="white" if dark else "black",
               bbox=dict(boxstyle="round,pad=0.15", fc="black" if dark else "none",
                         ec="none", alpha=0.45 if dark else 0))

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}  (segmentation k={seg.n_clusters}, silhouette={seg.silhouette:.3f})")


if __name__ == "__main__":
    main()
