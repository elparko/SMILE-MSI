"""Chunked, lazily-read m/z cube backed by a Zarr (v2) ZipStore.

The fast ion cube (:meth:`smile_msi.msi.MSIDataset.build_mz_cube`) is a sparse
``pixels × m/z-bin`` matrix. In RAM it lives as a SciPy CSC matrix; reopening a
slide used to rebuild that whole CSC in memory from the ``.cache.npz`` sidecar —
so peak RSS scaled with the *whole* cube even though a single ion image only
touches a thin m/z window.

:class:`CubeStore` swaps that for an on-disk, compressed, lazily-read store. It
exploits CSC's layout: every entry is stored column-by-column, so a **contiguous
m/z-bin column range maps to a contiguous slice** of the ``data``/``indices``
arrays via ``indptr``. That is exactly the access pattern of an ion image
(:meth:`smile_msi.msi.MSIDataset._cube_ion` — a column-range slice reduced across
columns). Reopening reads only the chunks an ion window touches instead of
materialising the whole cube, so warm-reopen RSS drops from "the whole cube" to
"a few chunks".

On-disk layout (a single ``<stem>.cube.zarr`` ZipStore holding a Zarr group)::

    attrs : {format_version, fingerprint, n_pixels, nbins, dtype, indices_dtype, ...}
    axis      : float64[nbins]            (kept in RAM — small)
    edges     : float64[nbins+1]          (kept in RAM — small)
    indptr    : int64[nbins+1]            (kept in RAM — small)
    colsum    : float64[nbins]            (kept in RAM — cube.T @ ones, for mean_spectrum)
    data      : <dtype>[nnz]              chunked + Blosc/zstd  (lazy)
    indices   : <idx-dtype>[nnz]          chunked + Blosc/zstd  (lazy)
    mean / mean_axis / tic / rms / median : optional prime-stats passthrough

The ``fingerprint`` + ``n_pixels`` staleness guard mirrors
:func:`smile_msi.session.load_cube` exactly: a mismatch yields ``None`` (never a
misapplied cube), and a corrupt/partial store degrades to "no cache → rebuild",
never a crash. Writes are atomic (tmp + :func:`os.replace`).

The store is read lazily through the kept-open ZipStore, whose reads are
mutex-guarded, so the GUI thread (region select) and worker threads (Stats A/B)
can hit it concurrently — same contract as the in-RAM CSC path.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

# Bump when the on-disk schema changes incompatibly. The fingerprint guard already
# rejects a stale cube; this rejects an old *format* even for the same slide.
FORMAT_VERSION = 1

# data/indices chunk length. ~4 MB per float32/int32 chunk: small enough that a
# typical ion window touches only one or two chunks, large enough to keep the
# per-read chunk count (and zip central-directory size) low. Tune via profiling.
_DATA_CHUNK = 1 << 20

# Bin block for the cross-pixel reduces (mean/max spectrum). These read every
# chunk anyway (they touch all bins), so the block size only bounds the transient
# CSC rebuilt per step, not the total I/O.
_BIN_BLOCK = 4096


def _blosc():
    """The shared compressor: zstd-3 + byte-shuffle. zstd-3 is a good
    speed/ratio default for float32 values + int32 indices; a higher clevel would
    trade decompress latency (which interactive browsing pays) for marginal size."""
    from numcodecs import Blosc
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)


def _remove(path: str) -> None:
    """Remove a stale tmp store (file *or* directory) so an atomic re-create can't
    trip over a half-written remnant from a killed prior save."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _write_group_aux(g, comp, *, axis, edges, indptr, colsum, n_pixels, nbins,
                     data_dtype, indices_dtype, fingerprint,
                     mean=None, pix=None, bin_ppm=None, min_intensity=None):
    """Write the small in-RAM arrays (axis/edges/indptr/colsum), the optional prime-stats
    passthrough (mean/pix), and the group attrs+staleness guard — shared by the one-shot
    :meth:`CubeStore.create` and the streaming :class:`_StreamingCubeWriter`."""
    def _small(name, arr):
        arr = np.asarray(arr)
        g.create_dataset(name, data=arr, chunks=(max(1, int(arr.shape[0])),), compressor=comp)

    _small("axis", np.asarray(axis, dtype=np.float64))
    _small("edges", np.asarray(edges, dtype=np.float64))
    _small("indptr", np.ascontiguousarray(indptr, dtype=np.int64))
    _small("colsum", np.asarray(colsum, dtype=np.float64))
    if mean is not None:
        m_axis, m_vals = (mean if isinstance(mean, tuple) else (axis, mean))
        _small("mean", np.asarray(m_vals))
        _small("mean_axis", np.asarray(m_axis))
    if pix is not None:
        for k in ("tic", "rms", "median"):
            v = pix.get(k)
            if v is not None:
                _small(k, np.asarray(v))

    attrs = {
        "format_version": FORMAT_VERSION,
        "fingerprint": str(fingerprint),
        "n_pixels": int(n_pixels),
        "nbins": int(nbins),
        "dtype": str(data_dtype),
        "indices_dtype": str(indices_dtype),
    }
    if bin_ppm is not None:
        attrs["bin_ppm"] = float(bin_ppm)
    if min_intensity is not None:
        attrs["min_intensity"] = float(min_intensity)
    g.attrs.put(attrs)              # one write → no duplicate-.zattrs zip entries


class CubeStore:
    """Lazily-read, chunked m/z cube backed by a Zarr group. Quacks like the
    ``(axis, csc, edges)`` tuple for the read paths that matter — :meth:`ion`
    (per-pixel ion image), :meth:`colsum` (the mean-spectrum numerator) and
    :meth:`max_spectrum`. ``axis``/``edges``/``indptr``/``colsum`` are tiny and
    kept in RAM; ``data``/``indices`` stay as lazy ``zarr.Array`` handles."""

    def __init__(self, store, group, *, axis, edges, indptr, colsum, n_pixels, nbins):
        self._store = store          # zarr ZipStore — kept open for lazy chunk reads
        self._g = group
        self.path = str(getattr(store, "path", "") or "")    # on-disk location (ZipStore)
        self.axis = np.asarray(axis, dtype=np.float64)       # float64[nbins]   (RAM)
        self.edges = np.asarray(edges, dtype=np.float64)     # float64[nbins+1] (RAM)
        self.indptr = np.asarray(indptr, dtype=np.int64)     # int64[nbins+1]   (RAM)
        self._colsum = np.asarray(colsum, dtype=np.float64)  # cube.T @ ones    (RAM)
        self.n_pixels = int(n_pixels)
        self.nbins = int(nbins)
        self._data = group["data"]                           # zarr.Array (lazy)
        self._indices = group["indices"]                     # zarr.Array (lazy)

    # ----- quack like the in-RAM CSC for cache keys ------------------------ #
    @property
    def shape(self):
        return (self.n_pixels, self.nbins)

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1]) if self.indptr.size else 0

    # ----- construction ---------------------------------------------------- #
    @classmethod
    def create(cls, path, axis, csc, edges, *, fingerprint, n_pixels=None,
               mean=None, pix=None, bin_ppm=None, min_intensity=None):
        """Write ``csc`` (the cube from :meth:`build_mz_cube`) to a Zarr ZipStore
        at ``path``, atomically (tmp + :func:`os.replace`). Stores the CSC arrays
        verbatim (no re-sorting) so a reconstructed column slice is byte-identical
        to slicing the in-RAM matrix. Returns an open :class:`CubeStore`."""
        import zarr

        path = str(path)
        csc = csc.tocsc()
        n_px = int(csc.shape[0]) if n_pixels is None else int(n_pixels)
        nbins = int(csc.shape[1])
        # Store the CSC arrays exactly as-is — the column-slice reconstruction must
        # see the same bytes as `cube[:, a:b]` for the reductions to be bit-identical.
        data = np.ascontiguousarray(csc.data)
        indices = np.ascontiguousarray(csc.indices)
        indptr = np.ascontiguousarray(csc.indptr, dtype=np.int64)
        # cube.T @ ones — the exact numerator cube_mean_spectrum(mask=None) computes.
        # Cheap here (we hold the whole CSC), tiny to store, and lets reopen serve the
        # whole-slide mean spectrum without reading a single data chunk.
        colsum = np.asarray(csc.T @ np.ones(n_px)).ravel().astype(np.float64)

        comp = _blosc()
        tmp = path + ".tmp"
        _remove(tmp)
        store = zarr.ZipStore(tmp, mode="w")
        try:
            g = zarr.open_group(store=store, mode="w")
            nnz = int(data.shape[0])
            dchunk = (max(1, min(_DATA_CHUNK, nnz)),)
            g.create_dataset("data", data=data, chunks=dchunk, compressor=comp)
            g.create_dataset("indices", data=indices, chunks=dchunk, compressor=comp)
            _write_group_aux(g, comp, axis=axis, edges=edges, indptr=indptr, colsum=colsum,
                             n_pixels=n_px, nbins=nbins, data_dtype=data.dtype,
                             indices_dtype=indices.dtype, fingerprint=fingerprint,
                             mean=mean, pix=pix, bin_ppm=bin_ppm, min_intensity=min_intensity)
        finally:
            store.close()
        os.replace(tmp, path)
        return cls.open(path, fingerprint=fingerprint, n_pixels=n_px)

    @classmethod
    def create_streaming(cls, path, *, axis, edges, indptr, nnz, fingerprint, n_pixels,
                         data_dtype=np.float32, indices_dtype=np.int32,
                         bin_ppm=None, min_intensity=None):
        """Open an incremental writer that streams ``data``/``indices`` blocks (in CSC
        column order) straight to disk so the whole cube never sits in RAM — the Phase-2
        out-of-core build path. ``indptr`` and the total ``nnz`` must be known up front (a
        counting pass over the binned data yields both). Returns a :class:`_StreamingCubeWriter`;
        call :meth:`write_block` for each column band in order, then :meth:`finalize`."""
        return _StreamingCubeWriter(
            str(path), axis=axis, edges=edges, indptr=indptr, nnz=int(nnz),
            fingerprint=str(fingerprint), n_pixels=int(n_pixels),
            data_dtype=np.dtype(data_dtype), indices_dtype=np.dtype(indices_dtype),
            bin_ppm=bin_ppm, min_intensity=min_intensity)

    @classmethod
    def open(cls, path, *, fingerprint, n_pixels):
        """Open a store written by :meth:`create`, or ``None`` if it's missing,
        unreadable/corrupt, the format is too new, or it doesn't match this slide
        (``fingerprint`` + ``n_pixels`` guard — a mismatched cube is never returned,
        mirroring :func:`smile_msi.session.load_cube`)."""
        path = str(path)
        if not os.path.exists(path):
            return None
        import zarr

        store = None
        try:
            store = zarr.ZipStore(path, mode="r")
            g = zarr.open_group(store=store, mode="r")
            a = g.attrs
            if int(a.get("format_version", -1)) != FORMAT_VERSION:
                store.close()
                return None
            if str(a.get("fingerprint", "")) != str(fingerprint):
                store.close()
                return None
            if int(a.get("n_pixels", -1)) != int(n_pixels):
                store.close()
                return None
            return cls(
                store, g,
                axis=g["axis"][:], edges=g["edges"][:],
                indptr=g["indptr"][:], colsum=g["colsum"][:],
                n_pixels=int(a["n_pixels"]), nbins=int(a["nbins"]),
            )
        except Exception:  # noqa: BLE001 — a corrupt/partial cache must never be fatal
            if store is not None:
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
            return None

    # ----- prime-stats passthrough ----------------------------------------- #
    def stored_extras(self):
        """Return ``(mean, pix)`` saved alongside the cube (for
        :func:`load_cube`'s return dict), or ``(None, None)`` if absent. ``mean``
        is ``(axis, vals)``; ``pix`` is ``{tic, rms, median}``."""
        g = self._g
        mean = None
        if "mean" in g:
            m_axis = g["mean_axis"][:] if "mean_axis" in g else self.axis
            mean = (np.asarray(m_axis), np.asarray(g["mean"][:]))
        pix = None
        if "tic" in g:
            pix = {
                "tic": np.asarray(g["tic"][:]),
                "rms": np.asarray(g["rms"][:]) if "rms" in g else None,
                "median": np.asarray(g["median"][:]) if "median" in g else None,
            }
        return mean, pix

    # ----- the read paths -------------------------------------------------- #
    def column_slice(self, a, b):
        """A small in-RAM ``csc_matrix`` for m/z bins ``[a:b]`` — reads only the
        ``data``/``indices`` chunks covering ``indptr[a]:indptr[b]``. Byte-identical
        to ``full_csc[:, a:b]``."""
        from scipy import sparse

        a = int(a)
        b = max(int(b), a)
        lo = int(self.indptr[a])
        hi = int(self.indptr[b])
        if hi > lo:
            data = self._data[lo:hi]
            indices = self._indices[lo:hi]
        else:                                   # empty window — no chunk reads at all
            data = np.empty(0, dtype=self._data.dtype)
            indices = np.empty(0, dtype=self._indices.dtype)
        local_indptr = (self.indptr[a:b + 1] - lo).astype(self.indptr.dtype)
        return sparse.csc_matrix((data, indices, local_indptr),
                                 shape=(self.n_pixels, b - a))

    def ion(self, mz, tol_ppm, reduce, n_pixels=None):
        """Per-pixel ion image for the m/z window ``mz ± tol_ppm`` — identical math
        to :meth:`MSIDataset._cube_ion`, but the column range is read lazily via
        :meth:`column_slice` instead of sliced from a resident CSC."""
        n = self.n_pixels if n_pixels is None else int(n_pixels)
        win = mz * tol_ppm / 1e6
        # Keep a/b as the numpy ints searchsorted returns (do NOT coerce to Python int):
        # _cube_ion divides the float32 sum by `b - a` as a numpy int64, which promotes the
        # mean to float64. A Python-int divisor would leave it float32 — a different rounding,
        # breaking the bit-for-bit contract with the in-RAM path.
        a = np.searchsorted(self.axis, mz - win, side="left")
        b = np.searchsorted(self.axis, mz + win, side="right")
        if b <= a:
            return np.zeros(n)
        sub = self.column_slice(a, b)
        if reduce == "max":
            return np.asarray(sub.max(axis=1).todense()).ravel()
        s = np.asarray(sub.sum(axis=1)).ravel()
        return s / (b - a) if reduce == "mean" else s

    def colsum(self, bmask=None):
        """Per-bin sum across rows — the numerator of ``cube_mean_spectrum``.

        ``bmask is None`` returns the stored ``cube.T @ ones`` (byte-identical to
        the in-RAM path, served without touching a data chunk). A boolean ``bmask``
        streams the cube in bin blocks computing ``cube[:, block].T @ w`` — each
        bin's value depends only on its own column, so the block-wise result equals
        the full ``cube.T @ w`` exactly."""
        if bmask is None:
            return self._colsum.copy()
        w = np.asarray(bmask, dtype=bool).astype(float)
        out = np.empty(self.nbins, dtype=np.float64)
        for a in range(0, self.nbins, _BIN_BLOCK):
            b = min(a + _BIN_BLOCK, self.nbins)
            out[a:b] = np.asarray(self.column_slice(a, b).T @ w).ravel()
        return out

    def max_spectrum(self, mask=None):
        """Per-bin maximum across pixels (the skyline) — identical result to
        :meth:`MSIDataset.cube_max_spectrum`, streamed in bin blocks. A boolean
        ``mask`` restricts to those rows."""
        axis = np.asarray(self.axis, dtype=float)
        idx = None
        if mask is not None:
            idx = np.flatnonzero(np.asarray(mask, dtype=bool))
            if idx.size == 0:
                return axis, np.zeros(self.nbins)
        out = np.zeros(self.nbins, dtype=np.float64)
        for a in range(0, self.nbins, _BIN_BLOCK):
            b = min(a + _BIN_BLOCK, self.nbins)
            sub = self.column_slice(a, b)
            if idx is not None:
                sub = sub[idx]
            if sub.shape[0] == 0:
                continue
            mx = sub.max(axis=0)
            out[a:b] = mx.toarray().ravel() if hasattr(mx, "toarray") else np.asarray(mx).ravel()
        return axis, out.astype(float)

    # ----- lifecycle ------------------------------------------------------- #
    def close(self):
        """Release the underlying ZipStore file handle. Idempotent. Called when a
        dataset drops its cube (rebuild / preprocessing change / release) so the
        cache file isn't held open — important on Windows, where an open handle
        blocks the atomic replace of a rebuilt cube."""
        store = getattr(self, "_store", None)
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
            self._store = None

    def __del__(self):
        self.close()


class _StreamingCubeWriter:
    """Incrementally writes a CubeStore's ``data``/``indices`` to a Zarr ZipStore without
    holding the whole cube in RAM (Phase-2 out-of-core build).

    Blocks arrive in CSC column order (the caller emits one column band at a time). The
    writer buffers them and flushes only **whole, chunk-aligned** slices in increasing
    offset — so every ``data``/``indices`` chunk is written exactly once, in order. That
    avoids the read-modify-write a ZipStore can't do (it's write-only) and the duplicate
    zip entries an unaligned overwrite would create. The trailing partial chunk is written
    once at :meth:`finalize`. Atomic: builds into ``path + ".tmp"`` then ``os.replace``."""

    def __init__(self, path, *, axis, edges, indptr, nnz, fingerprint, n_pixels,
                 data_dtype, indices_dtype, bin_ppm, min_intensity):
        import zarr

        self.path = str(path)
        self.tmp = self.path + ".tmp"
        _remove(self.tmp)
        self._axis = np.asarray(axis, dtype=np.float64)
        self._edges = np.asarray(edges, dtype=np.float64)
        self._indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self._nbins = int(self._axis.shape[0])
        self._n_pixels = int(n_pixels)
        self._nnz = int(nnz)
        self._fingerprint = str(fingerprint)
        self._bin_ppm = bin_ppm
        self._min_intensity = min_intensity
        self._data_dtype = np.dtype(data_dtype)
        self._indices_dtype = np.dtype(indices_dtype)

        self._comp = _blosc()
        self._store = zarr.ZipStore(self.tmp, mode="w")
        self._g = zarr.open_group(store=self._store, mode="w")
        self._chunk = max(1, min(_DATA_CHUNK, self._nnz if self._nnz else 1))
        self._data = self._g.create_dataset("data", shape=(self._nnz,),
                                             chunks=(self._chunk,), dtype=self._data_dtype,
                                             compressor=self._comp)
        self._indices = self._g.create_dataset("indices", shape=(self._nnz,),
                                                chunks=(self._chunk,), dtype=self._indices_dtype,
                                                compressor=self._comp)
        self._off = 0                       # next write position into data/indices
        self._buf_d = []                    # buffered blocks awaiting a chunk-aligned flush
        self._buf_i = []
        self._buf_n = 0

    def write_block(self, data_block, indices_block):
        """Append one CSC column-band block (already in canonical order). Cheap: just
        buffers, flushing whole chunks when enough has accumulated."""
        if data_block.size == 0:
            return
        self._buf_d.append(np.ascontiguousarray(data_block, dtype=self._data_dtype))
        self._buf_i.append(np.ascontiguousarray(indices_block, dtype=self._indices_dtype))
        self._buf_n += int(data_block.size)
        if self._buf_n >= self._chunk:
            self._flush(whole_chunks_only=True)

    def _flush(self, whole_chunks_only):
        if not self._buf_d:
            return
        d = np.concatenate(self._buf_d)
        i = np.concatenate(self._buf_i)
        k = (d.size // self._chunk) * self._chunk if whole_chunks_only else d.size
        if k:
            self._data[self._off:self._off + k] = d[:k]
            self._indices[self._off:self._off + k] = i[:k]
            self._off += k
        rem_d, rem_i = d[k:], i[k:]
        self._buf_d = [rem_d] if rem_d.size else []
        self._buf_i = [rem_i] if rem_i.size else []
        self._buf_n = int(rem_d.size)

    def finalize(self, colsum, *, mean=None, pix=None):
        """Flush the tail, write axis/edges/indptr/colsum + attrs, and atomically publish.
        Returns the opened :class:`CubeStore`."""
        self._flush(whole_chunks_only=False)
        if self._off != self._nnz:
            self.abort()
            raise ValueError(f"streaming cube wrote {self._off} of {self._nnz} entries")
        try:
            _write_group_aux(self._g, self._comp, axis=self._axis, edges=self._edges,
                             indptr=self._indptr, colsum=colsum, n_pixels=self._n_pixels,
                             nbins=self._nbins, data_dtype=self._data_dtype,
                             indices_dtype=self._indices_dtype, fingerprint=self._fingerprint,
                             mean=mean, pix=pix, bin_ppm=self._bin_ppm,
                             min_intensity=self._min_intensity)
        finally:
            self._store.close()
        os.replace(self.tmp, self.path)
        return CubeStore.open(self.path, fingerprint=self._fingerprint, n_pixels=self._n_pixels)

    def abort(self):
        """Best-effort cleanup of a half-written store (a failed/cancelled build)."""
        try:
            self._store.close()
        except Exception:  # noqa: BLE001
            pass
        _remove(self.tmp)
