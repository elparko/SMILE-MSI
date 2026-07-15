"""Data ingestion & format conversion — bring other file types in, and write the
open **imzML** standard out.

MSI vendors and tools export many shapes. This module loads the common
non-imzML ones into an :class:`~smile_msi.msi.MSIDataset`, and converts any
dataset to imzML so the rest of the toolkit (and other software) can read it.

Supported in:
* **imzML/ibd** — via :meth:`MSIDataset.from_imzml` (handled in :func:`load_any`).
* **imaging CSV/TSV** — *long* (columns ``x, y, mz, intensity``) or *wide* (``x, y``
  then one column per m/z).

Out:
* :func:`to_imzml` — write any dataset to imzML (continuous if it has a shared m/z
  axis, else processed).

Vendor binaries (Bruker ``.d``, Thermo/Waters ``.raw``) need their vendor libraries
or an external converter (e.g. ProteoWizard ``msconvert``, ``imzMLConverter``) to
reach imzML first; :func:`load_any` raises a clear pointer for those.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .msi import MSIDataset


def load_any(path: str, **kwargs) -> MSIDataset:
    """Load a dataset by extension: .imzML, or an imaging .csv/.tsv/.txt table."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".imzml":
        return MSIDataset.from_imzml(path, **kwargs)
    if ext in (".csv", ".tsv", ".txt"):
        return read_imaging_csv(path)
    if ext in (".d", ".raw", ".baf", ".tdf", ".tsf"):
        raise ValueError(
            f"'{ext}' is a vendor binary format. Convert it to imzML first with your "
            "vendor's exporter, ProteoWizard msconvert, or imzMLConverter, then open the imzML.")
    raise ValueError(f"Unrecognized file type: {ext}")


def read_imaging_csv(path: str) -> MSIDataset:
    """Read an imaging table into an MSIDataset.

    *Long* layout: columns ``x, y, mz, intensity`` (one row per peak per pixel).
    *Wide* layout: columns ``x, y`` then one numeric-named column per m/z (one row
    per pixel). Column names are matched case-insensitively; ``m/z`` is accepted
    for ``mz``.
    """
    sep = "\t" if path.lower().endswith(".tsv") else None
    df = pd.read_csv(path, sep=sep, engine="python")
    low = {c: str(c).strip().lower() for c in df.columns}

    def find(*names):
        for c, l in low.items():
            if l in names:
                return c
        return None

    xc, yc = find("x", "x_px", "column", "col"), find("y", "y_px", "row")
    mzc, ic = find("mz", "m/z"), find("intensity", "i", "value")
    if xc is None or yc is None:
        raise ValueError("imaging CSV needs 'x' and 'y' columns")

    if mzc is not None and ic is not None:                     # long format
        coords, mzs, ints = [], [], []
        for (x, y), g in df.groupby([xc, yc]):
            g = g.sort_values(mzc)
            coords.append((int(x), int(y)))
            mzs.append(g[mzc].to_numpy(float))
            ints.append(g[ic].to_numpy(float))
        return MSIDataset.from_arrays(coords, mzs, ints, source=path)

    # wide format: every non-x/y column header is an m/z
    mz_cols, axis = [], []
    for c in df.columns:
        if c in (xc, yc):
            continue
        try:
            axis.append(float(str(c)))
            mz_cols.append(c)
        except ValueError:
            continue
    if not mz_cols:
        raise ValueError("wide imaging CSV needs numeric m/z column headers, or use long "
                         "format with mz/intensity columns")
    order = np.argsort(axis)
    axis = np.asarray(axis)[order]
    mz_cols = [mz_cols[i] for i in order]
    coords = list(zip(df[xc].astype(int), df[yc].astype(int)))
    mat = df[mz_cols].to_numpy(float)
    return MSIDataset.from_arrays(coords, [axis] * len(coords), list(mat), source=path)


def to_imzml(ds, path: str) -> str:
    """Write a dataset to imzML (+ .ibd). Continuous mode when it has a shared m/z
    axis, else processed."""
    from pyimzml.ImzMLWriter import ImzMLWriter

    mode = "continuous" if ds.store.shared_axis() is not None else "processed"
    with ImzMLWriter(path, mode=mode, polarity=ds.polarity or None) as w:
        for i in range(ds.n_pixels):
            x, y = int(ds.coordinates[i][0]), int(ds.coordinates[i][1])
            mz, inten = ds.get_spectrum(i)
            w.addSpectrum(mz, inten, (x, y, 1))
    return path
