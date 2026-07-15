"""Spatial MALDI-MSI core — the imaging engine.

Designed for **large files**: spectra are never all held densely in RAM. An
:class:`MSIDataset` wraps a :class:`SpectrumStore` (lazy imzML, or in-memory for
the demo/tests) and works in streaming passes:

* **pass 1** (:meth:`prime`) streams every spectrum once to build the **mean
  spectrum** and per-pixel **normalization stats** (TIC / RMS / median) together;
* peaks are picked from the mean spectrum;
* **pass 2** (:meth:`build_features`) streams once more to extract a compact
  ``pixels x peaks`` feature matrix.

Everything interactive afterwards — ion images for picked peaks, segmentation,
co-localization, ROI statistics — is served from that small matrix (instant).
Arbitrary m/z falls back to a single streaming pass. No network, no database file.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import threading
from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_TOL_PPM
from .cubestore import CubeStore

# --------------------------------------------------------------------------- #
# Spectrum stores — abstract the on-disk vs in-memory difference
# --------------------------------------------------------------------------- #


class SpectrumStore:
    """Read-only access to a set of (m/z, intensity) spectra over pixels.

    Subclasses provide ``coordinates`` (N x 2 int array of 1-indexed x,y),
    ``__len__``, ``get(i) -> (mz, intensity)`` and ``shared_axis()`` (the common
    m/z vector for continuous data, else ``None``)."""

    coordinates: np.ndarray
    supports_parallel: bool = False     # can open_reader()/read_with() run concurrently?
    in_memory: bool = False             # spectra already in RAM? (then masked ops need no cube)

    def __len__(self) -> int:  # pragma: no cover - trivial
        raise NotImplementedError

    def get(self, i):  # pragma: no cover - trivial
        raise NotImplementedError

    def open_reader(self):
        """Open a private reader (e.g. a fresh file handle) for one worker thread, or
        ``None`` if reads need no per-thread resource (in-RAM stores). Each parallel
        streaming pass owns the readers it opens and closes them when it finishes, so
        concurrent passes never share — or close — each other's handles."""
        return None

    def read_with(self, reader, i):
        """Read spectrum ``i`` using a ``reader`` from :meth:`open_reader` (``None`` for
        in-RAM stores). Default ignores the reader and falls back to :meth:`get`."""
        return self.get(i)

    def shared_axis(self):
        return None

    def mz_bounds(self):
        """(min m/z, max m/z) across the dataset — cheap when a shared axis exists.

        Non-finite entries (a corrupt / misread spectrum) are dropped so one bad read can't
        poison the bounds — and, downstream, the dataset fingerprint that keys the managed
        session (a garbage range used to mint a duplicate cohort sample on every load)."""
        ax = self.shared_axis()
        if ax is not None:
            a = np.asarray(ax, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size:
                return float(a.min()), float(a.max())
        lo, hi = np.inf, -np.inf
        for i in range(len(self)):
            mz, _ = self.get(i)
            mz = np.asarray(mz, dtype=np.float64)
            mz = mz[np.isfinite(mz)]
            if mz.size:
                lo = min(lo, float(mz.min()))
                hi = max(hi, float(mz.max()))
        return lo, hi


class MemoryStore(SpectrumStore):
    """All spectra in RAM (used by the synthetic demo and tests)."""

    in_memory = True                    # masked means are already instant; no cube needed

    def __init__(self, coordinates, mzs, intensities):
        self.coordinates = np.asarray(coordinates, dtype=int)
        self._mzs = [np.asarray(m, dtype=np.float64) for m in mzs]
        self._ints = [np.asarray(v, dtype=np.float64) for v in intensities]
        self._shared = self._detect_shared()

    def _detect_shared(self):
        first = self._mzs[0]
        n = len(first)
        for m in self._mzs[1:]:
            if len(m) != n or m[0] != first[0] or m[-1] != first[-1]:
                return None
        return first

    def __len__(self):
        return len(self._mzs)

    def get(self, i):
        return self._mzs[i], self._ints[i]

    def shared_axis(self):
        return self._shared


class DenseMemoryStore(SpectrumStore):
    """All spectra held as one dense ``pixels × m/z`` float32 matrix in RAM, sharing a
    single m/z axis (continuous-mode data).

    Built by :meth:`MSIDataset.to_ram` on a machine with enough memory: it reads the
    whole acquisition off disk once and serves every subsequent pass from RAM. Because
    the axis is shared, the whole-dataset passes (mean spectrum, feature extraction)
    become single vectorized NumPy reductions over ``matrix`` instead of a per-pixel
    Python loop — the big win on a large-RAM box where disk I/O *and* loop overhead were
    the bottleneck. ``matrix`` holds **raw** intensities (preprocessing transforms are
    applied on read, exactly as for the lazy store), so changing preprocessing later
    stays correct."""

    in_memory = True

    def __init__(self, coordinates, axis, matrix, polarity="", spec_mode="",
                 pixel_size_um=None, pixel_size_y_um=None):
        self.coordinates = np.asarray(coordinates, dtype=int)
        self.axis = np.asarray(axis, dtype=np.float64)
        self.matrix = np.asarray(matrix, dtype=np.float32)   # (n_pixels, n_mz) raw intensities
        self.polarity = polarity
        self.spec_mode = spec_mode
        self.pixel_size_um = pixel_size_um
        self.pixel_size_y_um = pixel_size_y_um

    def __len__(self):
        # coordinates outlive the matrix: after MSIDataset.release() frees the dense block,
        # geometry (n_pixels / to_image) must still work for a slide whose label map is shown.
        return self.matrix.shape[0] if self.matrix is not None else self.coordinates.shape[0]

    def get(self, i):
        return self.axis, np.asarray(self.matrix[i], dtype=np.float64)

    def shared_axis(self):
        return self.axis


def _total_ram_bytes():
    """Total physical RAM in bytes, or ``None`` if it can't be determined."""
    import os
    try:                                       # POSIX (Linux / macOS)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:                                       # Windows
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return int(ms.ullTotalPhys)
    except Exception:  # noqa: BLE001 - best effort; fall through to unknown
        pass
    return None


def _ram_budget():
    """Byte budget for the dense in-RAM cache. ``$SMILE_MSI_RAM_BUDGET_GB`` overrides;
    otherwise half of physical RAM (leaving room for the OS, the app, and a working copy
    of the data), or a conservative 4 GiB when total RAM is unknown."""
    import os
    env = os.environ.get("SMILE_MSI_RAM_BUDGET_GB")
    if env:
        try:
            return float(env) * (1 << 30)
        except ValueError:
            pass
    total = _total_ram_bytes()
    return int(total * 0.5) if total else 4 * (1 << 30)


class ImzMLStore(SpectrumStore):
    """Lazy imzML backend — reads each spectrum from the .ibd on demand.

    pyimzml stores byte offsets per spectrum, so ``get(i)`` is a seek+read, not a
    full-file load. Continuous-mode datasets (identical m/z block reused for every
    pixel) are detected from the m/z offsets and the shared axis is read just once.

    Parallel reads: pyimzml's ``PortableSpectrumReader`` holds only byte offsets and
    reads a spectrum from *any* file object, so each worker thread can use its own
    ``.ibd`` handle. That makes :meth:`read_threadsafe` safe to call concurrently and
    lets the streaming passes overlap disk I/O across cores (``supports_parallel``).
    """

    supports_parallel = True

    def __init__(self, path: str, max_pixels: int | None = None, stride: int = 1):
        from pyimzml.ImzMLParser import ImzMLParser

        self._p = ImzMLParser(path)
        self.path = str(path)
        n_all = len(self._p.coordinates)
        sel = list(range(0, n_all, max(1, int(stride))))      # subsample every Nth pixel
        if max_pixels:
            sel = sel[:max_pixels]
        self._sel = sel                                       # parser index per logical pixel
        self._n = len(sel)
        self.coordinates = np.asarray(
            [(self._p.coordinates[i][0], self._p.coordinates[i][1]) for i in sel], dtype=int)
        # continuous mode -> every spectrum reuses one m/z block (same offset)
        self._shared = None
        try:
            if len({self._p.mzOffsets[i] for i in sel}) == 1:
                self._shared = np.asarray(self._p.getspectrum(sel[0])[0], dtype=np.float64)
        except Exception:  # noqa: BLE001
            self._shared = None
        self.polarity = _safe_meta(self._p, "polarity")
        self.spec_mode = _safe_meta(self._p, "spec_mode")
        self.pixel_size_um = None
        self.pixel_size_y_um = None
        try:
            self.pixel_size_um = float(self._p.imzmldict.get("pixel size x"))
        except Exception:  # noqa: BLE001
            pass
        # Read the vertical spacing too: non-square acquisition pixels are common, and
        # carrying y lets callers detect (and warn about) a distorted display / scale bar
        # instead of silently assuming square pixels. Fall back to x when y is absent.
        try:
            self.pixel_size_y_um = float(self._p.imzmldict.get("pixel size y"))
        except Exception:  # noqa: BLE001
            pass
        # Origin of the pixel size so a *derived* estimate is never presented as a measured
        # value (surfaced by pixel_size_warning / recorded in provenance): one of "imzml"
        # (an explicit cvParam), "derived" (recovered just below), or None (unknown).
        self.pixel_size_source = "imzml" if self.pixel_size_um else None
        # Many exporters (e.g. SCiLS) record the image *extent* ("max dimension x/y" +
        # "max count of pixels x/y") but omit an explicit "pixel size x"; without it every
        # renderer's scale bar silently never draws. Recover µm/pixel from extent ÷ pixel
        # count so a bar can still be shown — flagged "derived" so callers mark it an estimate.
        if self.pixel_size_um is None:
            try:
                d = self._p.imzmldict
                dim_x, nx = float(d.get("max dimension x")), float(d.get("max count of pixels x"))
                if dim_x > 0 and nx > 0:
                    self.pixel_size_um = dim_x / nx
                    self.pixel_size_source = "derived"
                    try:
                        dim_y = float(d.get("max dimension y"))
                        ny = float(d.get("max count of pixels y"))
                        self.pixel_size_y_um = (dim_y / ny) if (dim_y > 0 and ny > 0) \
                            else self.pixel_size_um
                    except Exception:  # noqa: BLE001
                        self.pixel_size_y_um = self.pixel_size_um
            except Exception:  # noqa: BLE001
                pass
        # parallel-read scaffolding: a shared (read-only) offset reader; each worker
        # thread opens its own .ibd handle via open_reader() so cursors never collide.
        try:
            self._portable = self._p.portable_spectrum_reader()
            self._ibd_path = self._p.m.name
        except Exception:  # noqa: BLE001 - fall back to serial get() if unavailable
            self._portable = None
            self._ibd_path = None
            self.supports_parallel = False
        # Cache the (immutable) parse result so the next open of this file skips the
        # single-threaded XML parse entirely (see from_cache). Only the full, unsubsampled
        # read is cacheable; previews (stride/max_pixels) parse fresh. Best-effort.
        if int(stride) == 1 and max_pixels is None and self._portable is not None:
            self._save_parse_cache()

    def _save_parse_cache(self):
        try:
            from . import session
            p = self._p
            session.save_parse_cache(
                self.path,
                mz_offsets=p.mzOffsets, mz_lengths=p.mzLengths,
                int_offsets=p.intensityOffsets, int_lengths=p.intensityLengths,
                mz_precision=p.mzPrecision, int_precision=p.intensityPrecision,
                coordinates=[(c[0], c[1]) for c in p.coordinates],
                shared=self._shared, polarity=self.polarity, spec_mode=self.spec_mode,
                pixel_size_um=self.pixel_size_um, pixel_size_source=self.pixel_size_source)
        except Exception:  # noqa: BLE001 — caching must never break a load
            pass

    @classmethod
    def from_cache(cls, path: str, max_pixels: int | None = None, stride: int = 1):
        """Rebuild a store from the parse-offset sidecar (written on the previous open),
        skipping pyimzml's single-threaded XML parse. Returns ``None`` on a miss/stale cache
        (caller then parses fresh). Only the full read is cached — previews parse fresh."""
        if max_pixels is not None or int(stride) != 1:
            return None
        import os
        from . import session
        from pyimzml.ImzMLParser import PortableSpectrumReader
        cached = session.load_parse_cache(path)
        if cached is None:
            return None
        ibd = os.path.splitext(str(path))[0] + ".ibd"
        if not os.path.exists(ibd):
            return None
        self = cls.__new__(cls)
        self._p = None                                  # no live parser — reads go via portable
        self.path = str(path)
        coords = cached["coordinates"]
        self._sel = list(range(len(coords)))
        self._n = len(self._sel)
        self.coordinates = np.asarray([(int(c[0]), int(c[1])) for c in coords], dtype=int)
        self._shared = cached["shared"]
        self.polarity = cached["polarity"]
        self.spec_mode = cached["spec_mode"]
        self.pixel_size_um = cached["pixel_size_um"]
        # y isn't round-tripped through the sidecar yet (the drawn bar uses x); default it so
        # attribute access never fails on a cache-built store. Origin persists so a derived
        # value stays flagged as an estimate on reopen.
        self.pixel_size_y_um = None
        self.pixel_size_source = cached.get("pixel_size_source")
        self.supports_parallel = True
        self._ibd_path = ibd
        self._portable = PortableSpectrumReader(
            [tuple(int(v) for v in c) for c in coords],
            cached["mz_precision"], cached["mz_offsets"].tolist(),
            cached["mz_lengths"].tolist(), cached["int_precision"],
            cached["int_offsets"].tolist(), cached["int_lengths"].tolist())
        return self

    def __len__(self):
        return self._n

    def get(self, i):
        if self._p is not None:
            mz, inten = self._p.getspectrum(self._sel[i])
        else:
            # built from a parse-cache (no live ImzMLParser): read via the portable reader
            # on a one-shot handle. The parallel path (read_with) keeps persistent per-worker
            # handles, so this serial fallback is only the small/no-parallel case.
            with open(self._ibd_path, "rb") as f:
                mz, inten = self._portable.read_spectrum_from_file(f, self._sel[i])
        return np.asarray(mz, dtype=np.float64), np.asarray(inten, dtype=np.float64)

    def open_reader(self):
        """A fresh, private .ibd file handle for one worker thread (or None to fall back
        to the serial parser)."""
        if self._portable is None or self._ibd_path is None:
            return None
        return open(self._ibd_path, "rb")

    def read_with(self, reader, i):
        if reader is None:
            return self.get(i)
        mz, inten = self._portable.read_spectrum_from_file(reader, self._sel[i])
        return np.asarray(mz, dtype=np.float64), np.asarray(inten, dtype=np.float64)

    def read_intensity_block(self):
        """Bulk-read the whole intensity matrix in ONE sequential read → dense
        ``(n_pixels, n_bins)`` float32, replacing ``to_ram``'s per-pixel seeks (3-10x on
        NVMe). Strictly guarded: returns ``None`` — so the caller falls back to the proven
        per-pixel path — unless this is a full, in-order continuous read whose intensity
        arrays are all the shared length and physically contiguous in the .ibd (offset[i]
        == offset[0] + i·k·itemsize). The contiguity check makes a wrong bulk read
        impossible: any interleaving/padding/precision quirk fails it and we read per-pixel."""
        import os
        pr = self._portable
        if pr is None or self._ibd_path is None or self._shared is None:
            return None
        if self._sel != list(range(len(self._sel))):     # subsampled/preview reads aren't contiguous
            return None
        offs = np.asarray(pr.intensityOffsets, dtype=np.int64)
        lens = np.asarray(pr.intensityLengths, dtype=np.int64)
        n = len(self._sel)
        k = int(len(self._shared))
        if offs.size != n or lens.size != n or k <= 0 or not np.all(lens == k):
            return None
        try:
            dtype = np.dtype(pr.intensityPrecision)
        except TypeError:
            return None
        item = dtype.itemsize
        expected = offs[0] + np.arange(n, dtype=np.int64) * (k * item)
        if not np.array_equal(offs, expected):           # not one contiguous block → fall back
            return None
        nbytes = n * k * item
        if os.path.getsize(self._ibd_path) < int(offs[0]) + nbytes:
            return None
        # Read sequentially in bounded row-chunks straight into a preallocated writable
        # float32 matrix (== the RAM-budgeted size). Reading the whole block in one shot and
        # then casting would transiently hold raw+float32 at once — a ~3x spike for 64-bit
        # intensities that could MemoryError a near-budget slide the per-pixel path loads
        # fine; np.empty also makes the result writable (np.frombuffer alone is read-only).
        mat = np.empty((n, k), dtype=np.float32)
        rows_per = max(1, (16 << 20) // max(k * item, 1))    # ~16 MB per read
        with open(self._ibd_path, "rb") as f:
            f.seek(int(offs[0]))
            for a in range(0, n, rows_per):
                b = min(a + rows_per, n)
                chunk = np.frombuffer(f.read((b - a) * k * item), dtype=dtype)
                if chunk.size != (b - a) * k:
                    return None
                mat[a:b] = chunk.reshape(b - a, k)           # cast to float32 via assignment
        return mat

    def shared_axis(self):
        return self._shared


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class _Features:
    peaks: np.ndarray        # m/z of each feature column
    matrix: np.ndarray       # (n_pixels, n_peaks) raw integrated intensities
    tol_ppm: float
    reduce: str


def _subset_columns(have, want):
    """Indices into ``have`` that reproduce ``want`` in order, or ``None`` when any
    requested m/z isn't an exact column of ``have`` (i.e. ``want`` is not a subset).

    Used to serve a removed/sub-selected peak list by slicing the cached feature matrix
    rather than re-extracting. The m/z floats flow through unchanged on removal, so an
    exact match is the right test — anything else (an added or shifted peak) falls back
    to a full rebuild."""
    index = {}
    for j, m in enumerate(have):
        index.setdefault(float(m), j)        # first column wins if a m/z ever repeats
    cols = []
    for m in want:
        j = index.get(float(m))
        if j is None:
            return None
        cols.append(j)
    return np.asarray(cols, dtype=int)


class MSIDataset:
    """A MALDI-MSI acquisition backed by a :class:`SpectrumStore`.

    Construct with :meth:`from_imzml` (lazy by default) or :meth:`from_arrays`.
    """

    # Per-instance identity for derived-result caches: a process-monotonic id (so a freed
    # dataset's address can never be reused to serve a *different* slide's cached result)
    # paired with a generation counter bumped on every in-place mutation. See cache_token.
    _uid_counter = itertools.count(1)

    def __init__(self, store: SpectrumStore, polarity: str = "", spec_mode: str = "",
                 source: str = ""):
        self.store = store
        self.coordinates = store.coordinates
        self.polarity = polarity
        self.spec_mode = spec_mode
        self.source = source
        self.pixel_size_um = getattr(store, "pixel_size_um", None)
        self.pixel_size_y_um = getattr(store, "pixel_size_y_um", None)
        self.pixel_size_source = getattr(store, "pixel_size_source", None)
        self.orientation = 0         # display rotation, 90°-CW steps (0–3); see set_orientation
        self.transforms: list = []   # per-spectrum preprocessing callables (mz,inten)->(mz,inten)
        # caches built on demand
        self._mean = None          # (axis, mean_intensity)
        self._pix = None           # dict(tic=, rms=, median=)
        self._norm_cache: dict = {}  # per-method normalization factors (depend only on _pix)
        self._masked_mean_cache: dict = {}  # exact LRU of masked mean spectra (ROI/region signatures)
        self._feat: _Features | None = None
        self._ion_cache: dict = {}  # LRU of streamed arbitrary-m/z ion vectors
        self._cube = None           # optional (axis, csc pixels×bins, edges) for fast ion images
        self._cube_mean_cache: dict = {}  # LRU of cube-served masked mean spectra (region re-select)
        self._cube_mean_lock = threading.Lock()  # guards _cube_mean_cache (GUI + worker threads hit it)
        self._max = None            # cached skyline (maximum-projection) spectrum
        self._bin_ppm = 10.0
        self._fingerprint = None    # memoised slide identity (library.dataset_fingerprint)
        self._cache_uid = next(MSIDataset._uid_counter)   # unique id for derived-result caches
        self._cache_gen = 0         # bumped on in-place mutation (preprocessing/orientation/release)
        self._ion_embedding: dict = {}  # learned ion embeddings, keyed by caller (ionembed.py)

    # ----- preprocessing hook --------------------------------------------- #
    def _read(self, i):
        """Read spectrum i and apply the preprocessing pipeline (if any)."""
        mz, inten = self.store.get(int(i))
        for t in self.transforms:
            mz, inten = t(mz, inten)
        return mz, inten

    # ----- parallel streaming -------------------------------------------- #
    # User-tunable worker cap for the parallel streaming passes (prime / build_features
    # / build_mz_cube). Reads are disk- and numpy-bound (both release the GIL), so a
    # thread pool overlaps them across cores. Default: leave headroom for the GUI.
    n_stream_workers: int | None = None

    def _stream_workers(self) -> int:
        if self.n_stream_workers:
            return max(1, int(self.n_stream_workers))
        import os
        # Use most cores (the passes run on a background thread, so the GUI stays live);
        # leave 2 free for responsiveness, and cap at 16 to avoid I/O oversubscription.
        return max(2, min(16, (os.cpu_count() or 4) - 2))

    def _read_many(self, indices, parallel=None, apply_transforms=True):
        """Yield ``(i, mz, inten)`` for each index — reads run concurrently on a thread
        pool when the store supports it, so the expensive disk-read+decode (+transforms)
        overlap across cores. Results are yielded **in order**; only a bounded window of
        spectra is ever in flight, so the streaming memory model is preserved. The serial
        fallback is identical to looping :meth:`_read`. ``apply_transforms=False`` reads
        the raw stored spectra (used by :meth:`to_ram`, which snapshots raw intensities so
        a later preprocessing change stays correct)."""
        indices = list(indices)
        if parallel is None:
            parallel = (getattr(self.store, "supports_parallel", False)
                        and len(indices) >= 1024)
        transforms = self.transforms if apply_transforms else ()
        if not parallel:
            for i in indices:
                mz, inten = self.store.get(int(i))
                for t in transforms:
                    mz, inten = t(mz, inten)
                yield i, mz, inten
            return

        import threading
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        nw = self._stream_workers()
        store = self.store
        # Readers are owned by THIS call (one per worker thread, opened lazily, closed at
        # the end), so a concurrent _read_many on the same store — e.g. the auto cube
        # build racing "Find peaks" — never shares or closes another pass's handles.
        tl = threading.local()
        readers = []
        rlock = threading.Lock()

        def task(i):
            r = getattr(tl, "r", None)
            if r is None:
                r = store.open_reader()
                tl.r = r
                with rlock:
                    readers.append(r)
            mz, inten = store.read_with(r, int(i))
            for t in transforms:
                mz, inten = t(mz, inten)
            return i, mz, inten

        it = iter(indices)
        ex = ThreadPoolExecutor(max_workers=nw)
        try:
            inflight = deque()
            for _ in range(nw * 4):                  # bounded prefetch window
                try:
                    inflight.append(ex.submit(task, next(it)))
                except StopIteration:
                    break
            while inflight:
                rec = inflight.popleft().result()    # in-order; propagates worker errors
                yield rec
                try:
                    inflight.append(ex.submit(task, next(it)))
                except StopIteration:
                    pass
        finally:
            ex.shutdown(wait=True)
            for r in readers:                        # close only the handles we opened
                if r is not None:
                    try:
                        r.close()
                    except Exception:  # noqa: BLE001
                        pass

    def set_preprocessing(self, transforms):
        """Set the per-spectrum preprocessing pipeline (list of callables that take
        and return ``(mz, intensity)``, preserving the m/z axis). Clears caches."""
        self.transforms = list(transforms or [])
        # Cap how much per-pixel normalization may amplify a faint pixel. A low-TIC pixel
        # (tissue edge / incomplete baseline / electronic spike — the ones SCiLS excludes
        # from normalization) would otherwise be divided by a near-zero factor and explode
        # into a fake hotspot. 3× lets genuine tissue normalize while bounding background
        # runaway; raise it for gentler correction, set <=0 to disable. See norm_factors.
        self.tic_max_amp = 3.0
        self._cache_gen += 1        # preprocessing changes derived numbers — invalidate result caches
        self._mean = None
        self._pix = None
        self._norm_cache = {}
        self._masked_mean_cache = {}
        self._feat = None
        self._ion_cache = {}
        self._close_cube()          # release an on-disk cube handle before dropping it
        self._cube = None
        self._cube_mean_cache = {}
        self._max = None

    def cache_token(self):
        """A hashable identity that changes whenever this dataset's derived numbers
        could change. It pairs a per-instance unique id with a generation counter that is
        bumped on every in-place mutation (:meth:`set_preprocessing`,
        :meth:`set_orientation`, :meth:`release`). Derived-result caches (the feature-list
        annotation/FDR bundle, region mean spectra) key on this so switching back to a
        prior input set is an instant cache hit, while a mutated — or different — slide
        never serves a stale result. ``to_ram`` is lossless (same numbers) and does NOT
        bump the generation."""
        return (self._cache_uid, self._cache_gen)

    # ----- in-RAM acceleration -------------------------------------------- #
    def _dense(self):
        """The dense ``pixels × m/z`` matrix when this dataset has been loaded into RAM
        (:meth:`to_ram`) *and* no preprocessing transforms are active — the precondition
        for the vectorized whole-dataset passes. Returns ``None`` otherwise (callers then
        take the streaming path, which applies transforms per spectrum)."""
        M = getattr(self.store, "matrix", None)
        return None if (M is None or self.transforms) else M

    def to_ram(self, max_bytes: float | None = None, progress=None,
               parallel=None) -> bool:
        """Read the whole acquisition into a dense in-RAM float32 matrix so every
        subsequent pass is vectorized and disk-free — the big lever on a large-RAM
        machine, where the load-time bottleneck is disk I/O + per-pixel Python overhead.

        Only continuous-mode data (a shared m/z axis) is supported; processed/ragged data
        returns ``False`` (left lazy). Also returns ``False`` — leaving the lazy store in
        place — when the dense matrix would exceed the RAM budget (:func:`_ram_budget`,
        overridable via ``max_bytes`` or ``$SMILE_MSI_RAM_BUDGET_GB``), so a file too big
        for memory degrades gracefully instead of thrashing. Idempotent."""
        store = self.store
        if getattr(store, "matrix", None) is not None:
            return True                                   # already dense
        shared = store.shared_axis()
        if shared is None:
            return False                                  # processed mode: no dense matrix
        axis = np.asarray(shared, dtype=np.float64)
        n, k = self.n_pixels, len(axis)
        budget = _ram_budget() if max_bytes is None else float(max_bytes)
        if n * k * 4 > budget:
            return False                                  # would blow the RAM budget
        # Fast path: a single bulk read of the contiguous .ibd intensity block (raw, which is
        # exactly what the per-pixel snapshot below stores). Strictly guarded inside
        # read_intensity_block — None here means the layout isn't safely contiguous, so we
        # take the proven per-pixel path. Identical numbers either way.
        block = None
        bulk = getattr(store, "read_intensity_block", None)
        if bulk is not None:
            block = bulk()
        if block is not None and block.shape == (n, k):
            mat = block
            if progress:
                progress(n, n)
        else:
            mat = np.zeros((n, k), dtype=np.float32)
            done = 0
            # snapshot RAW spectra (apply_transforms=False) so a later set_preprocessing stays
            # correct; transforms are re-applied on read just as they are for the lazy store.
            for i, _mz, inten in self._read_many(range(n), parallel=parallel,
                                                 apply_transforms=False):
                row = np.asarray(inten, dtype=np.float32)
                L = min(row.shape[0], k)
                mat[i, :L] = row[:L]
                done += 1
                if progress and done % 256 == 0:
                    progress(done, n)
            if progress:
                progress(n, n)
        self.store = DenseMemoryStore(self.coordinates, axis, mat, polarity=self.polarity,
                                      spec_mode=self.spec_mode,
                                      pixel_size_um=self.pixel_size_um,
                                      pixel_size_y_um=self.pixel_size_y_um)
        self.coordinates = self.store.coordinates
        return True                                       # caches stay valid (same numbers)

    def release(self) -> None:
        """Free the in-RAM dense matrix (:meth:`to_ram`) and every derived cache
        (prime stats, feature matrix, ion cube, skyline) so a streaming/cohort loader can
        reclaim this cube's memory the moment its last section is consumed — **deterministically**,
        not on GC timing. The object is left inert (its heavy buffers gone); reload from
        source to use it again. Idempotent. This is the eviction primitive
        :class:`smile_msi.cohort.SectionLoader` relies on to bound resident RAM to one cube
        across a large cohort."""
        store = getattr(self, "store", None)
        if store is not None and getattr(store, "matrix", None) is not None:
            try:
                store.matrix = None        # drop the dense pixels×bins block — the big one
            except Exception:              # noqa: BLE001 — a read-only store still frees caches below
                pass
        # the same buffer list set_preprocessing clears, plus the feature/ion caches
        self._cache_gen += 1
        self._mean = None
        self._pix = None
        self._norm_cache = {}
        self._masked_mean_cache = {}
        self._feat = None
        self._ion_cache = {}
        self._close_cube()          # release an on-disk cube handle before dropping it
        self._cube = None
        self._cube_mean_cache = {}
        self._max = None

    # ----- construction ---------------------------------------------------- #
    @classmethod
    def from_imzml(cls, path: str, lazy: bool = True, max_pixels: int | None = None,
                   stride: int = 1):
        """Load an imzML dataset. ``stride > 1`` subsamples every Nth pixel for a fast
        preview of very large files; ``max_pixels`` caps the pixel count."""
        # Reuse the parse-offset sidecar from a previous open (skips the XML parse); on a
        # miss/stale cache, parse fresh (which also (re)writes the sidecar).
        store = (ImzMLStore.from_cache(path, max_pixels=max_pixels, stride=stride)
                 or ImzMLStore(path, max_pixels=max_pixels, stride=stride))
        ds = cls(store, polarity=getattr(store, "polarity", ""),
                 spec_mode=getattr(store, "spec_mode", ""), source=str(path))
        return ds

    @classmethod
    def from_arrays(cls, coordinates, mzs, intensities, polarity="", spec_mode="",
                    source="arrays"):
        store = MemoryStore(coordinates, mzs, intensities)
        return cls(store, polarity=polarity, spec_mode=spec_mode, source=source)

    # ----- geometry -------------------------------------------------------- #
    @property
    def n_pixels(self) -> int:
        return len(self.store)

    @property
    def x_range(self):
        return int(self.coordinates[:, 0].min()), int(self.coordinates[:, 0].max())

    @property
    def y_range(self):
        return int(self.coordinates[:, 1].min()), int(self.coordinates[:, 1].max())

    @property
    def _base_hw(self):
        """Unrotated (height, width) from the raw acquisition coordinate extents."""
        x_lo, x_hi = self.x_range
        y_lo, y_hi = self.y_range
        return (y_hi - y_lo + 1, x_hi - x_lo + 1)

    @property
    def width(self) -> int:
        h0, w0 = self._base_hw
        return h0 if (self.orientation % 2) else w0      # 90°/270° swap the axes

    @property
    def height(self) -> int:
        h0, w0 = self._base_hw
        return w0 if (self.orientation % 2) else h0

    @property
    def mz_range(self):
        return self.store.mz_bounds()

    def set_orientation(self, k: int):
        """Display rotation in 90°-clockwise steps (0–3). Applied at the single
        pixel→grid mapping in :meth:`_pixel_rows_cols`, so every view (ion images,
        segmentation, ROIs, exports) rotates coherently. Regions/ROIs are stored as
        per-pixel masks, not image grids, so they stay valid across orientation changes."""
        new = int(k) % 4
        if new != self.orientation:
            self._cache_gen += 1    # rotation changes ion-image geometry — invalidate result caches
        self.orientation = new

    def rotate90(self, clockwise: bool = True):
        """Rotate the displayed image 90°; returns the new orientation (0–3)."""
        self.set_orientation(self.orientation + (1 if clockwise else -1))
        return self.orientation

    def _pixel_rows_cols(self):
        """Each acquired pixel's (row, col) on the *displayed* grid, after the
        orientation rotation. Every image↔pixel mapping flows through here."""
        x0, _ = self.x_range
        y0, _ = self.y_range
        r0 = self.coordinates[:, 1] - y0
        c0 = self.coordinates[:, 0] - x0
        k = self.orientation % 4
        if k == 0:
            return r0, c0
        h0, w0 = self._base_hw
        if k == 1:                                       # 90° clockwise
            return c0, (h0 - 1 - r0)
        if k == 2:                                       # 180°
            return (h0 - 1 - r0), (w0 - 1 - c0)
        return (w0 - 1 - c0), r0                         # 270° clockwise

    # ----- physical pixel size (scale bars) -------------------------------- #
    def set_pixel_size(self, x_um, y_um=None) -> None:
        """Override the physical pixel size (µm per pixel) used for scale bars and any
        physical-distance readout. ``x_um`` is the horizontal spacing (what a horizontal
        scale bar measures against); ``y_um`` defaults to ``x_um`` (square pixels). Pass a
        non-positive / ``None`` ``x_um`` to clear it (e.g. an imzML that never recorded one,
        or a wrong value the user wants to correct). The value also propagates to the dense
        in-RAM store so it survives :meth:`to_ram`."""
        x = float(x_um) if x_um and float(x_um) > 0 else None
        y = (float(y_um) if y_um and float(y_um) > 0 else x)
        self.pixel_size_um = x
        self.pixel_size_y_um = y
        # a user correction is authoritative (and survives session save/reload); clearing it
        # drops the origin back to unknown.
        self.pixel_size_source = "user" if x else None
        # keep the backing store in sync so to_ram() / from_arrays round-trips carry it
        if hasattr(self.store, "pixel_size_um"):
            self.store.pixel_size_um = x
        if hasattr(self.store, "pixel_size_y_um"):
            self.store.pixel_size_y_um = y

    def coordinate_grid_step(self):
        """The acquisition's coordinate spacing as ``(step_x, step_y)`` integer index steps,
        from the greatest common divisor of the gaps between occupied rows/columns. A normal
        dense raster returns ``(1, 1)``; a value > 1 means the stored coordinates skip indices
        (e.g. positions written in physical units, or a sub-sampled grid), so one *displayed*
        pixel is NOT one acquired pixel and a naive ``µm / pixel_size`` scale bar is off by
        that factor. Returns ``(1, 1)`` for a degenerate (single-row/column) grid."""
        def _gcd_step(vals):
            u = np.unique(np.asarray(vals, dtype=np.int64))
            if u.size < 2:
                return 1
            return int(np.gcd.reduce(np.diff(u)))
        return (_gcd_step(self.coordinates[:, 0]), _gcd_step(self.coordinates[:, 1]))

    def pixel_size_warning(self):
        """A human-readable caution string when the scale bar may be inaccurate, else
        ``None``. Flags two independent problems: **non-square pixels** (x≠y spacing, so the
        displayed image is geometrically distorted and a single bar can't describe both axes)
        and a **non-unit coordinate grid** (see :meth:`coordinate_grid_step`, where the
        drawn bar would be wrong by the step factor). Surfaced in the UI so a bar is never
        shown as authoritative when it isn't."""
        msgs = []
        px, py = self.pixel_size_um, self.pixel_size_y_um
        if getattr(self, "pixel_size_source", None) == "derived" and px:
            msgs.append(f"pixel size ({px:g} µm) was inferred from the image extent, not "
                        f"recorded in the file — the scale bar is an estimate")
        if px and py and not math.isclose(px, py, rel_tol=1e-3):
            msgs.append(f"non-square pixels ({px:g}×{py:g} µm); the horizontal bar uses the "
                        f"x spacing, but the image is vertically distorted")
        sx, sy = self.coordinate_grid_step()
        if max(sx, sy) > 1:
            msgs.append(f"coordinates step by ({sx}, {sy}), not 1 — the displayed grid is "
                        f"coarser than the acquired grid, so the bar may be off by that factor")
        return "; ".join(msgs) if msgs else None

    def auto_scale_bar_um(self):
        """A round '1-2-5 ×10ⁿ' scale-bar length (µm) ≈ 20% of the displayed slide width, or
        ``None`` when the pixel size is unknown. The single entry point every export path uses
        so its bar sizes (and labels) identically to the live overlay
        (:meth:`gui.ion.IonView._live_scale_bar_um`) and the crop studio."""
        from .annotations import nice_scalebar_um
        px = self.pixel_size_um
        if not px or not self.width:
            return None
        return nice_scalebar_um(self.width * float(px))

    def to_image(self, values, fill=np.nan) -> np.ndarray:
        """Scatter a per-pixel vector onto the (height, width) tissue grid."""
        values = np.asarray(values, dtype=float)
        img = np.full((self.height, self.width), fill, dtype=float)
        rows, cols = self._pixel_rows_cols()
        img[rows, cols] = values
        return img

    def get_spectrum(self, i):
        return self._read(i)

    # ----- pass 1: mean spectrum + per-pixel normalization stats ----------- #
    def prime(self, progress=None):
        """Stream every spectrum once to build the mean spectrum and per-pixel
        TIC / RMS / median stats together. Idempotent (cached)."""
        if self._mean is not None and self._pix is not None:
            return
        M = self._dense()
        if M is not None:
            self._mean, self._pix = self._prime_dense(M, progress)
            return
        shared = self.store.shared_axis()
        n = self.n_pixels
        tic = np.zeros(n)
        rms = np.zeros(n)
        med = np.zeros(n)

        done = 0
        if shared is not None:
            axis = np.asarray(shared, dtype=np.float64)
            acc = np.zeros(len(axis), dtype=np.float64)
            for i, _, inten in self._read_many(range(n)):
                inten = np.asarray(inten, dtype=np.float64)
                acc[: len(inten)] += inten
                tic[i], rms[i], med[i] = _pixel_stats(inten)
                done += 1
                if progress and done % 256 == 0:
                    progress(done, n)
            mean = acc / n
        else:
            axis = self._log_axis(self._bin_ppm)
            edges = _bin_edges(axis)
            acc = np.zeros(len(axis), dtype=np.float64)
            for i, mz, inten in self._read_many(range(n)):
                idx = np.searchsorted(edges, mz, side="right") - 1
                ok = (idx >= 0) & (idx < len(axis))
                acc += np.bincount(idx[ok], weights=inten[ok], minlength=len(axis))
                tic[i], rms[i], med[i] = _pixel_stats(inten)
                done += 1
                if progress and done % 256 == 0:
                    progress(done, n)
            mean = acc / n
        if progress:
            progress(n, n)
        self._mean = (axis, mean)
        self._pix = {"tic": tic, "rms": rms, "median": med}

    def _prime_dense(self, M, progress=None):
        """Vectorized :meth:`prime` for an in-RAM dense matrix: the mean spectrum is one
        column reduction; the per-pixel TIC / RMS / positive-median stats are computed in
        row chunks (bounded float64 temporaries) so the result is numerically identical to
        the streaming path without materializing a float64 copy of the whole matrix."""
        import warnings

        axis = np.asarray(self.store.shared_axis(), dtype=np.float64)
        n = M.shape[0]
        mean = M.sum(axis=0, dtype=np.float64) / max(n, 1)   # no big temporary
        tic = np.zeros(n)
        rms = np.zeros(n)
        med = np.zeros(n)
        step = 8192
        for s in range(0, n, step):
            C = np.asarray(M[s:s + step], dtype=np.float64)  # one chunk upcast at a time
            e = s + C.shape[0]
            tic[s:e] = C.sum(axis=1)
            rms[s:e] = np.sqrt(np.mean(C ** 2, axis=1)) if C.shape[1] else 0.0
            with warnings.catch_warnings():                  # all-non-positive row -> nan
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(np.where(C > 0, C, np.nan), axis=1)
            med[s:e] = np.where(np.isfinite(m), m, 0.0)
            if progress:
                progress(min(e, n), n)
        if progress:
            progress(n, n)
        return (axis, mean), {"tic": tic, "rms": rms, "median": med}

    def _log_axis(self, bin_ppm: float, max_bins: int = 200000) -> np.ndarray:
        lo, hi = self.mz_range
        lo = max(lo, 1e-6)                       # guard non-positive m/z (degenerate data)
        if hi <= lo:
            hi = lo * 1.001
        step = bin_ppm * 1e-6
        if not np.isfinite(step) or step <= 0:   # bin_ppm<=0 → log1p(step)=0 → inf bins (int(inf) crash)
            step = 15e-6                          # fall back to a sane default ppm
        nb = min(int(np.log(hi / lo) / np.log1p(step)) + 1, max_bins)
        return lo * (1 + step) ** np.arange(max(nb, 1))

    def mean_spectrum(self, mask=None):
        """Mean intensity per m/z over all pixels (streaming, cached).

        A boolean ``mask`` restricts to a subset of pixels via a streaming pass over
        just those pixels (used for ROI/cluster signatures). Masked results are kept in
        a small exact LRU keyed by the mask, so re-selecting a region or re-overlaying
        its spectrum is instant instead of re-reading every pixel from disk."""
        if mask is None:
            self.prime()
            axis, mean = self._mean
            if axis is not None and mean is not None and len(axis) != len(mean):
                # Guard a malformed cache (e.g. a dense store whose shared axis and matrix
                # width disagree): a mismatched (axis, intensity) breaks every downstream
                # plot/writer. Trim to the shorter so the spectrum is always well-formed.
                n = min(len(axis), len(mean))
                self._mean = (np.asarray(axis)[:n], np.asarray(mean)[:n])
            return self._mean
        mask = np.asarray(mask, dtype=bool)
        key = _mask_key(mask)
        hit = self._masked_mean_cache.get(key)
        if hit is not None:
            return hit
        shared = self.store.shared_axis()
        idxs = np.flatnonzero(mask)
        if shared is not None:
            axis = np.asarray(shared, dtype=np.float64)
            acc = np.zeros(len(axis))
            for i, _, inten in self._read_many(idxs):    # parallel, like prime()/max_spectrum
                acc[: len(inten)] += inten
        else:
            axis = self._log_axis(self._bin_ppm)
            edges = _bin_edges(axis)
            acc = np.zeros(len(axis))
            for i, mz, inten in self._read_many(idxs):
                bi = np.searchsorted(edges, mz, side="right") - 1
                ok = (bi >= 0) & (bi < len(axis))
                acc += np.bincount(bi[ok], weights=inten[ok], minlength=len(axis))
        result = (axis, acc / max(len(idxs), 1))
        if len(self._masked_mean_cache) >= 8:      # bounded LRU (masks can be large)
            self._masked_mean_cache.pop(next(iter(self._masked_mean_cache)))
        self._masked_mean_cache[key] = result
        return result

    def cube_mean_spectrum(self, mask=None):
        """Fast mean spectrum from the prebuilt m/z cube — an in-memory sparse
        weighted row-sum, no disk re-read. Returns ``None`` if no cube has been built
        (call :meth:`build_mz_cube` first, i.e. "Build fast ion cache").

        Result is at the cube's bin resolution, so it's the load-bearing path for
        *instant* ROI / region spectra and the matching fast-mode backdrop — not the
        exact full-resolution mean used for analysis and export.
        """
        if self._cube is None:
            return None
        # _cube may be the in-RAM (axis, csc, edges) tuple or a lazily-read on-disk
        # CubeStore. `cube` here is the object that owns the row-sum compute and the
        # shape/nnz cache fingerprint; the store quacks like the CSC for both.
        store = self._cube if isinstance(self._cube, CubeStore) else None
        if store is not None:
            axis, cube = store.axis, store
        else:
            axis, cube, _ = self._cube
        # Region selection recomputes this CSR.T@vec on every click; memoize per (cube,
        # mask) so re-selecting a region — the common interaction — is instant. The cube's
        # id+shape+nnz fingerprints *which* cube, so a rebuilt cube (or a mutated slide,
        # which drops _cube entirely) misses automatically — no stale spectrum.
        bmask = None if mask is None else np.asarray(mask, dtype=bool)
        ckey = (id(cube), cube.shape, getattr(cube, "nnz", None),
                None if bmask is None else _mask_key(bmask))
        # _cube_mean_cache is read on the GUI thread (region select) and written on worker
        # threads (e.g. Stats A/B), so every touch is under the lock — an unguarded
        # pop(next(iter()))/insert races into "dict changed size during iteration".
        with self._cube_mean_lock:
            hit = self._cube_mean_cache.get(ckey)
        if hit is not None:
            return hit
        w = np.ones(self.n_pixels) if bmask is None else bmask.astype(float)
        n = float(w.sum())
        axis = np.asarray(axis, dtype=float)
        if n <= 0:
            return axis, np.zeros(cube.shape[1])
        if store is not None:
            # mask=None is served from the stored cube.T@ones (no data chunk read);
            # a masked region streams bin-chunks. Both equal the in-RAM CSR.T@vec.
            s = store.colsum(bmask)
        else:
            s = np.asarray(cube.T @ w).ravel()  # sum over selected rows (CSR.T @ vec is fast)
        result = (axis, s / n)
        with self._cube_mean_lock:
            cache = self._cube_mean_cache
            if len(cache) >= 16:                # bounded LRU (each entry ≈ one spectrum)
                cache.pop(next(iter(cache)))
            cache[ckey] = result
        return result

    def cube_max_spectrum(self, mask=None):
        """Fast skyline (per-m/z maximum) from the prebuilt m/z cube — a sparse
        column-wise max, no disk re-read. Returns ``None`` if no cube has been built
        (call :meth:`build_mz_cube` first, i.e. "Build fast ion cache").

        Result is at the cube's bin resolution, so it's the *instant* fast-mode backdrop
        for the spectrum view — not the exact full-resolution skyline used for analysis.
        """
        if self._cube is None:
            return None
        # On-disk store: stream the per-bin column-max in bin-chunks (identical result
        # to the in-RAM sparse column max below).
        if isinstance(self._cube, CubeStore):
            return self._cube.max_spectrum(mask)
        axis, cube, _ = self._cube
        axis = np.asarray(axis, dtype=float)
        if mask is not None:
            idx = np.flatnonzero(np.asarray(mask, dtype=bool))
            if idx.size == 0:
                return axis, np.zeros(cube.shape[1])
            cube = cube[idx]
        if cube.shape[0] == 0:
            return axis, np.zeros(cube.shape[1])
        mx = cube.max(axis=0)                    # sparse 1×nbins (intensities ≥ 0 → 0 is the floor)
        mx = mx.toarray().ravel() if hasattr(mx, "toarray") else np.asarray(mx).ravel()
        return axis, mx.astype(float)

    def max_spectrum(self, mask=None):
        """Skyline (maximum-projection) spectrum: the per-m/z maximum across pixels.
        Surfaces ions confined to a few pixels that the mean spectrum dilutes. A
        boolean ``mask`` restricts to a region; the whole-dataset result is cached.

        Reads stream through the shared thread pool (:meth:`_read_many`) so the per-pixel
        disk-read+decode overlaps across cores — the exact path is still O(pixels) but no
        longer single-threaded. Prefer :meth:`cube_max_spectrum` for an instant backdrop."""
        if mask is None and self._max is not None:
            return self._max
        shared = self.store.shared_axis()
        idxs = np.flatnonzero(np.asarray(mask, dtype=bool)) if mask is not None \
            else range(self.n_pixels)
        if shared is not None:
            axis = np.asarray(shared, dtype=np.float64)
            acc = np.zeros(len(axis))
            for _, _, inten in self._read_many(idxs):
                acc[: len(inten)] = np.maximum(acc[: len(inten)], inten)
        else:
            axis = self._log_axis(self._bin_ppm)
            edges = _bin_edges(axis)
            acc = np.zeros(len(axis))
            for _, mz, inten in self._read_many(idxs):
                bi = np.searchsorted(edges, mz, side="right") - 1
                ok = (bi >= 0) & (bi < len(axis))
                tmp = np.zeros(len(axis))
                np.maximum.at(tmp, bi[ok], inten[ok])
                acc = np.maximum(acc, tmp)
        result = (axis, acc)
        if mask is None:
            self._max = result
        return result

    # ----- normalization --------------------------------------------------- #
    def norm_factors(self, method: str = "none", scope=None) -> np.ndarray:
        """Per-pixel normalization divisors for the **whole slide** (one entry per pixel).

        ``scope`` is an optional array of pixel rows defining the population the mean scale is
        taken over — typically the on-tissue pixels. The returned factors still cover every
        pixel; only the constant they are divided by changes. A cross-sample analysis wants
        this, because the whole-slide mean TIC (the default) drifts with how much background
        each section carries, injecting a per-slide multiplicative bias into between-sample
        contrasts. Scoped results are not cached — they are cheap and caller-specific.
        """
        if method in (None, "none", ""):
            return np.ones(self.n_pixels)
        if scope is not None:
            self.prime()
            raw = np.asarray(self._pix[method], dtype=float)
            rows = np.asarray(scope, dtype=int)
            return _finalize_norm(raw, float(getattr(self, "tic_max_amp", 3.0)),
                                  scale_from=raw[rows] if rows.size else None)
        cached = self._norm_cache.get(method)
        if cached is not None:                # depends only on _pix (cleared with caches)
            return cached
        self.prime()
        out = _finalize_norm(self._pix[method], float(getattr(self, "tic_max_amp", 3.0)))
        self._norm_cache[method] = out
        return out

    # ----- pass 2: compact feature matrix (pixels x peaks) ----------------- #
    def build_features(self, peaks, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                       progress=None) -> np.ndarray:
        """Stream once to extract integrated intensity at each peak m/z for every
        pixel. Result is cached; ion images for these peaks are then instant."""
        peaks = np.asarray(peaks, dtype=float)
        n, p = self.n_pixels, len(peaks)
        mat = np.zeros((n, p), dtype=np.float32)
        win = peaks * tol_ppm / 1e6
        lo = peaks - win
        hi = peaks + win
        # When peaks densely cover the spectrum a prefix-sum is cheaper; otherwise
        # summing each small window directly wins (the usual case: few wide peaks on
        # a long profile spectrum). Pick per spectrum.
        dense = p >= 256
        # Continuous mode: every pixel shares one m/z axis, so the window bounds are the
        # same for all pixels — compute them once instead of n redundant searchsorts.
        shared = self.store.shared_axis()
        a0 = np.searchsorted(shared, lo, side="left") if shared is not None else None
        b0 = np.searchsorted(shared, hi, side="right") if shared is not None else None
        # In-RAM dense path: each peak window is one vectorized reduction down all pixels
        # (p reductions over n rows, vs the n×p per-pixel streaming loop).
        M = self._dense()
        if M is not None and a0 is not None:
            for j in range(p):
                a, b = int(a0[j]), int(b0[j])
                if b <= a:
                    continue
                seg = M[:, a:b]
                if reduce == "max":
                    mat[:, j] = seg.max(axis=1)
                else:
                    s = seg.sum(axis=1, dtype=np.float64)
                    mat[:, j] = s / (b - a) if reduce == "mean" else s
            if progress:
                progress(n, n)
            self._feat = _Features(peaks=peaks, matrix=mat, tol_ppm=tol_ppm, reduce=reduce)
            return mat
        done = 0
        for i, mz, inten in self._read_many(range(n)):
            if a0 is not None:
                a, b = a0, b0
            else:
                a = np.searchsorted(mz, lo, side="left")
                b = np.searchsorted(mz, hi, side="right")
            if reduce != "max" and dense:
                csum = np.concatenate(([0.0], np.cumsum(inten)))
                sums = csum[b] - csum[a]
                mat[i] = sums / np.maximum(b - a, 1) if reduce == "mean" else sums
            else:
                _reduce_windows(inten, a, b, reduce, mat[i])
            done += 1
            if progress and done % 256 == 0:
                progress(done, n)
        if progress:
            progress(n, n)
        self._feat = _Features(peaks=peaks, matrix=mat, tol_ppm=tol_ppm, reduce=reduce)
        return mat

    def ensure_features(self, peaks, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                        progress=None) -> np.ndarray:
        """Build the feature matrix for ``peaks`` only if not already cached with
        the same peaks/tolerance/reduce. Returns the raw (un-normalized) matrix."""
        peaks = np.asarray(peaks, dtype=float)
        f = self._feat
        if f is not None and f.tol_ppm == tol_ppm and f.reduce == reduce:
            if len(f.peaks) == len(peaks) and np.allclose(f.peaks, peaks):
                return f.matrix                          # exact cache hit
            # Removal / sub-selection: when every requested peak is already a column of
            # the cached matrix (same tol/reduce), its extracted intensities are identical
            # — slice those columns instead of streaming the whole slide again. Turns
            # dropping a feature from a list (a whole-slide O(n_pixels × n_peaks) re-extract
            # that froze the GUI thread) into an O(n_pixels × kept) column copy.
            cols = _subset_columns(f.peaks, peaks)
            if cols is not None:
                self._feat = _Features(peaks=f.peaks[cols], matrix=f.matrix[:, cols],
                                       tol_ppm=tol_ppm, reduce=reduce)
                if progress:
                    progress(self.n_pixels, self.n_pixels)
                return self._feat.matrix
        self.build_features(peaks, tol_ppm=tol_ppm, reduce=reduce, progress=progress)
        return self._feat.matrix

    @property
    def feature_peaks(self):
        return None if self._feat is None else self._feat.peaks

    # ----- learned ion embeddings (in-memory cache) ------------------------ #
    def get_ion_embedding(self, key):
        """Return a cached :class:`smile_msi.ionembed.IonEmbedding` for ``key`` (a hashable
        cache key, e.g. ``(peaks_hash, tol_ppm, norm, dim, random_state)``), or ``None``.

        A small in-memory store (like the feature-matrix cache) so re-running learned
        co-localization doesn't retrain; not part of the constructor or serialization."""
        return self._ion_embedding.get(key)

    def set_ion_embedding(self, key, emb) -> None:
        """Cache a trained :class:`smile_msi.ionembed.IonEmbedding` under ``key`` (see
        :meth:`get_ion_embedding`)."""
        self._ion_embedding[key] = emb

    def feature_matrix(self, norm: str = "none") -> np.ndarray:
        """Normalized copy of the cached feature matrix (pixels x peaks). TIC/RMS/median
        normalization for MSI: Deininger et al. (2011), doi:10.1007/s00216-011-4929-z."""
        if self._feat is None:
            raise RuntimeError("call build_features(peaks) first")
        return self._feat.matrix / self.norm_factors(norm)[:, None]

    def features_for_rows(self, peaks, rows, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                          norm: str = "none") -> np.ndarray:
        """The normalized ``(len(rows) × peaks)`` feature matrix for **only** the given pixel
        rows, extracted *without* building or caching the whole-slide matrix.

        For a cohort/embedding caller that keeps a capped random subsample (or one region) of a
        million-pixel slide, this avoids the ``O(n_pixels × peaks)`` extraction and the float64
        copy spike of :meth:`feature_matrix`. With a dense in-RAM cube it reduces only the
        selected rows (bounding the transient to ``len(rows) × bins``) and is **numerically
        identical** to ``feature_matrix(norm)[rows]``. Without a dense cube it **streams only
        those pixels** (no whole-slide ``ensure_features``/``prime`` pass) and computes the
        per-pixel normalization scale over the requested subset — a global constant that the
        downstream per-feature z-scoring removes. The whole-slide ``_feat`` cache is untouched."""
        rows = np.asarray(rows, dtype=int)
        peaks = np.asarray(peaks, dtype=float)
        M = self._dense()
        shared = self.store.shared_axis() if M is not None else None
        if M is None or shared is None:
            # No dense cube: **stream only the requested pixels** and build their features
            # inline — so a cohort region-mean run touches a fraction of a large slide instead
            # of the whole-slide ensure_features + prime passes (the big win for a single big
            # slide). Per-pixel TIC/RMS/median normalization is computed from the same streamed
            # spectra (scale over this subset; a global constant z-scoring later removes anyway).
            order = {int(r): k for k, r in enumerate(rows)}
            out = np.zeros((rows.size, peaks.size), dtype=np.float32)
            lo = peaks - peaks * tol_ppm / 1e6
            hi = peaks + peaks * tol_ppm / 1e6
            facs = np.ones(rows.size)
            want_norm = norm not in (None, "none", "")
            for i, mz, inten in self._read_many(rows):
                k = order.get(int(i))
                if k is None:
                    continue
                a = np.searchsorted(mz, lo, side="left")
                b = np.searchsorted(mz, hi, side="right")
                _reduce_windows(inten, a, b, reduce, out[k])
                if want_norm:
                    facs[k] = (float(np.sqrt(np.mean(inten * inten))) if norm == "rms"
                               else _positive_median(inten) if norm == "median"
                               else float(inten.sum()))            # tic / sum
            if want_norm:
                g = _finalize_norm(facs, float(getattr(self, "tic_max_amp", 3.0)))
                out = out / g[:, None]
            return out
        win = peaks * tol_ppm / 1e6                       # norm_factors() primes lazily if norm!=none
        a0 = np.searchsorted(shared, peaks - win, side="left")
        b0 = np.searchsorted(shared, peaks + win, side="right")
        sub = M[rows]                                    # (len(rows) × bins) — bounded by the cap
        out = np.zeros((rows.size, peaks.size), dtype=np.float32)
        for j in range(peaks.size):
            a, b = int(a0[j]), int(b0[j])
            if b <= a:
                continue
            seg = sub[:, a:b]
            if reduce == "max":
                out[:, j] = seg.max(axis=1)
            else:
                s = seg.sum(axis=1, dtype=np.float64)
                out[:, j] = s / (b - a) if reduce == "mean" else s
        if norm not in (None, "none", ""):
            out = out / self.norm_factors(norm)[rows][:, None]
        return out

    def _feature_index(self, mz: float):
        if self._feat is None:
            return None
        d = np.abs(self._feat.peaks - mz)
        j = int(np.argmin(d))
        tol = mz * self._feat.tol_ppm / 1e6
        return j if d[j] <= tol else None

    # ----- fast ion-image cube (sparse pixels × m/z bins) ------------------ #
    def estimate_cube_bytes(self, bin_ppm: float = 15.0, max_bins: int = 60000,
                            min_intensity: float = 0.0, sample: int = 64) -> int:
        """Estimate the in-RAM cube size in bytes by sampling ``sample`` spectra and scaling
        the mean distinct-bins-per-pixel by the pixel count. Cheap (a handful of reads) — used
        to choose between the fast in-RAM build and the bounded-memory streaming build."""
        n = self.n_pixels
        if n == 0:
            return 0
        lo, hi = self.mz_range
        lo = max(lo, 1e-6)
        need_ppm = ((hi / lo) ** (1.0 / max_bins) - 1.0) * 1e6 if hi > lo else bin_ppm
        axis = self._log_axis(max(bin_ppm, need_ppm), max_bins)
        edges = _bin_edges(axis)
        nbins = len(axis)
        shared = self.store.shared_axis()
        bi0 = valid0 = None
        if shared is not None:
            bi0 = np.searchsorted(edges, shared, side="right") - 1
            valid0 = (bi0 >= 0) & (bi0 < nbins)
        idxs = np.unique(np.linspace(0, n - 1, min(sample, n)).astype(int))
        total = cnt = 0
        for i, mz, inten in self._read_many(idxs):
            cols, _ = self._bin_pixel(mz, inten, edges, nbins, min_intensity, bi0, valid0)
            total += int(np.unique(cols).size) if cols.size else 0
            cnt += 1
        if cnt == 0:
            return 0
        nnz_est = total / cnt * n
        return int(nnz_est * 8)              # data(4) + indices(4) per entry; indptr negligible

    def _bin_pixel(self, mz, inten, edges, nbins, min_intensity, bi0=None, valid0=None):
        """Bin one spectrum to ``(cols int32, vals float64)`` — the per-pixel work shared by
        the in-RAM and streaming cube builds, so both emit byte-identical entries. ``bi0``/
        ``valid0`` are the precomputed shared-axis bin indices (continuous mode); when absent
        the bins are searched per spectrum."""
        if bi0 is not None:
            mask = valid0 & (inten > min_intensity)
            cols = bi0[mask]
        else:
            bi = np.searchsorted(edges, mz, side="right") - 1
            mask = (bi >= 0) & (bi < nbins) & (inten > min_intensity)
            cols = bi[mask]
        return cols.astype(np.int32), np.asarray(inten[mask], dtype=np.float64)

    def build_mz_cube(self, bin_ppm: float = 15.0, max_bins: int = 60000,
                      min_intensity: float = 0.0, progress=None,
                      out_path=None, fingerprint: str = "", max_ram_bytes=None):
        """Stream once to build a sparse ``pixels × m/z-bin`` matrix. Afterwards an
        ion image at ANY m/z is a fast column-range slice (no file re-read) — the
        load-bearing path for browsing large files. Bins are ``bin_ppm`` ppm wide,
        so the result is a fast *approximation* at that resolution; picked-peak
        images still come from the exact feature matrix.

        With ``out_path`` set, the cube is assembled **out-of-core** straight into a
        :class:`~smile_msi.cubestore.CubeStore` at that path (Phase-2 streaming build):
        the whole cube never sits in RAM, so peak build memory is bounded by
        ``max_ram_bytes`` (default 256 MB) instead of ~4× the cube. ``_cube`` then becomes
        the lazily-read store. Without ``out_path`` the legacy in-RAM build runs unchanged."""
        from scipy import sparse

        # widen bins if needed so the cube spans the full m/z range within max_bins
        lo, hi = self.mz_range
        lo = max(lo, 1e-6)
        need_ppm = ((hi / lo) ** (1.0 / max_bins) - 1.0) * 1e6 if hi > lo else bin_ppm
        axis = self._log_axis(max(bin_ppm, need_ppm), max_bins)
        edges = _bin_edges(axis)
        n = self.n_pixels
        nbins = len(axis)
        # Continuous mode: bin index is identical for every pixel, so compute it (and the
        # in-range validity) once instead of an n× searchsorted in the per-pixel loop.
        shared = self.store.shared_axis()
        bi0 = None
        valid0 = None
        if shared is not None:
            bi0 = np.searchsorted(edges, shared, side="right") - 1
            valid0 = (bi0 >= 0) & (bi0 < nbins)

        if out_path is not None:
            # The streaming build writes straight into a Zarr CubeStore. If zarr/numcodecs
            # aren't importable (a source install predating the cube store, or a frozen build
            # missing the hidden import) don't crash the whole cache build — drop to the in-RAM
            # path below, which needs neither. The on-disk sidecar just isn't written; the cube
            # cache is a best-effort optimisation (see cubestore.py and session.save_cube).
            try:
                import numcodecs  # noqa: F401 — preflight: streaming build needs the store
                import zarr  # noqa: F401
            except ImportError:
                out_path = None
            else:
                return self._build_mz_cube_streaming(
                    out_path, axis=axis, edges=edges, nbins=nbins, bi0=bi0, valid0=valid0,
                    min_intensity=min_intensity, bin_ppm=bin_ppm, fingerprint=fingerprint,
                    max_ram_bytes=max_ram_bytes, progress=progress)
        # Assemble the cube in pixel CHUNKS: each chunk becomes a compact float32 CSR block
        # and the blocks are vstacked at the end. This bounds the transient COO memory to
        # one chunk (int32 indices) instead of holding a list of arrays for EVERY pixel and
        # then concatenating them — the int64 list-of-arrays + concatenate + per-pixel
        # nbins-wide bincount were the ~3 MB/pixel blow-up that drove large profile slides
        # into swap (29 GB at 200×200). Duplicate (row, col) entries within a pixel are
        # summed by the CSR constructor (in float64, before the float32 cast), so the result
        # is numerically identical to the old per-point bincount — no per-pixel bincount and
        # no nbins-wide temporary needed.
        CHUNK = 8192
        parts: list = []
        buf_r: list = []
        buf_c: list = []
        buf_v: list = []
        base = 0
        in_chunk = 0
        done = 0

        def flush(end):
            nonlocal buf_r, buf_c, buf_v
            nrows = end - base
            if buf_r:
                r = (np.concatenate(buf_r) - base).astype(np.int32)
                c = np.concatenate(buf_c)
                v = np.concatenate(buf_v)                     # float64 → sums collisions exactly
            else:
                r = np.empty(0, np.int32)
                c = np.empty(0, np.int32)
                v = np.empty(0, np.float64)
            part = sparse.csr_matrix((v, (r, c)), shape=(nrows, nbins))
            parts.append(part.astype(np.float32))             # store compact; sums already done
            buf_r, buf_c, buf_v = [], [], []

        for i, mz, inten in self._read_many(range(n)):
            cols, vals = self._bin_pixel(mz, inten, edges, nbins, min_intensity, bi0, valid0)
            if cols.size:
                buf_r.append(np.full(cols.size, i, dtype=np.int64))
                buf_c.append(cols)
                buf_v.append(vals)
            in_chunk += 1
            done += 1
            if progress and done % 256 == 0:
                progress(done, n)
            if in_chunk >= CHUNK:
                flush(base + in_chunk)
                base += in_chunk
                in_chunk = 0
        if in_chunk:
            flush(base + in_chunk)
        if progress:
            progress(n, n)
        cube = (sparse.vstack(parts, format="csc") if parts
                else sparse.csc_matrix((n, nbins), dtype=np.float32))
        self._close_cube()          # a rebuild over a reopened on-disk cube must free its handle
        self._cube = (axis, cube, edges)
        # Building (or rebuilding at a new bin_ppm/min_intensity) the cube changes which
        # ion-extraction path ion_vector takes for non-picked m/z — _cube_ion's binned
        # approximation vs the exact _cached_stream_ion — so isotope/ROI/FDR numbers in a
        # feature-list bundle differ before vs after the cube exists. Bump the generation so
        # cache_token() changes and the feature-list cache can't serve a pre-cube bundle.
        self._cache_gen += 1
        return self._cube

    def _build_mz_cube_streaming(self, out_path, *, axis, edges, nbins, bi0, valid0,
                                 min_intensity, bin_ppm, fingerprint, max_ram_bytes, progress):
        """Out-of-core cube build (see :meth:`build_mz_cube`): the cube is written straight to
        a :class:`CubeStore` at ``out_path`` so it never sits in RAM whole.

        Two passes, both memory-bounded:

        * **Pass 1** streams every spectrum, bins it (identically to the in-RAM build), and
          spills the per-pixel ``(row, col, value)`` entries to a temp file in *pixel order*,
          accumulating a per-column count. Only one spectrum is resident.
        * **Pass 2** turns that row-ordered spill into canonical CSC one **column band** at a
          time: select the band's entries, stable-sort by column (rows stay ascending, since
          they were spilled in pixel order), and stream the sorted ``data``/``indices`` block
          to the store. Per-column sums (the mean-spectrum numerator) are accumulated in
          float64 exactly as ``cube.T @ ones`` would. The band count is chosen so each band's
          transient stays under ``max_ram_bytes``.

        The resulting store is byte-identical to building the CSC in RAM and calling
        :meth:`CubeStore.create`, so ion images / mean & max spectra are unchanged."""
        import math
        import os
        import tempfile

        from scipy import sparse

        from .cubestore import CubeStore

        n = self.n_pixels
        dest_dir = os.path.dirname(str(out_path)) or "."
        spill_dtypes = (("r", np.int32), ("c", np.int32), ("v", np.float32))
        paths, handles = {}, {}
        for key, _dt in spill_dtypes:
            fd, p = tempfile.mkstemp(prefix=f".cube_spill_{key}_", suffix=".bin", dir=dest_dir)
            os.close(fd)
            paths[key] = p
            handles[key] = open(p, "wb")
        writer = None
        try:
            # --- pass 1: stream, bin, spill (row, col, val) in pixel order; count columns ---
            col_counts = np.zeros(nbins, dtype=np.int64)
            nnz = 0
            for i, mz, inten in self._read_many(range(n)):
                cols, vals = self._bin_pixel(mz, inten, edges, nbins, min_intensity, bi0, valid0)
                if cols.size:
                    # 1-row CSR sums within-pixel bin collisions in float64 then casts to
                    # float32 — the exact same value the in-RAM block CSR produces for this row.
                    row = sparse.csr_matrix(
                        (vals, (np.zeros(cols.size, dtype=np.int32), cols)),
                        shape=(1, nbins)).astype(np.float32)
                    c = row.indices.astype(np.int32)
                    handles["r"].write(np.full(c.size, i, dtype=np.int32).tobytes())
                    handles["c"].write(c.tobytes())
                    handles["v"].write(row.data.tobytes())
                    col_counts[c] += 1
                    nnz += int(c.size)
                if progress and (i & 255) == 0:
                    progress(i, 2 * n)                 # pass 1 is the first half of the bar
            for h in handles.values():
                h.flush()
                h.close()

            indptr = np.empty(nbins + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(col_counts, out=indptr[1:])

            # --- pass 2: column-band counting-sort -> CSC, streamed to the store ---
            # Read the spill with explicit chunked file reads (NOT mmap): mmap pages would
            # count against process RSS once touched and defeat the memory bound, whereas
            # np.fromfile leaves the spill in the OS page cache and keeps only one scan chunk
            # resident. Each band holds ~budget/_PER_ENTRY entries at once for the stable sort.
            budget = int(max_ram_bytes) if max_ram_bytes else (128 << 20)
            _PER_ENTRY = 44                           # r+c+v spill + selected + argsort + sorted out
            nbands = max(1, int(math.ceil(nnz * _PER_ENTRY / max(budget, 1)))) if nnz else 1
            band_edges = np.linspace(0, nbins, nbands + 1).astype(np.int64)
            colsum = np.zeros(nbins, dtype=np.float64)
            writer = CubeStore.create_streaming(
                out_path, axis=axis, edges=edges, indptr=indptr, nnz=nnz,
                fingerprint=fingerprint, n_pixels=n, bin_ppm=bin_ppm,
                min_intensity=min_intensity)
            scan = 1 << 22                            # entries per spill-scan chunk (bounded RAM)
            readers = {k: open(paths[k], "rb") for k, _dt in spill_dtypes}
            try:
                for bi in range(nbands):
                    c0, c1 = int(band_edges[bi]), int(band_edges[bi + 1])
                    if c1 <= c0:
                        continue
                    sel_r, sel_c, sel_v = [], [], []
                    for rd in readers.values():
                        rd.seek(0)
                    remaining = nnz
                    while remaining > 0:
                        k = min(scan, remaining)
                        remaining -= k
                        cc = np.fromfile(readers["c"], dtype=np.int32, count=k)
                        rr = np.fromfile(readers["r"], dtype=np.int32, count=k)
                        vv = np.fromfile(readers["v"], dtype=np.float32, count=k)
                        m = (cc >= c0) & (cc < c1)
                        if m.any():
                            sel_r.append(rr[m])
                            sel_c.append(cc[m])
                            sel_v.append(vv[m])
                    if not sel_c:
                        continue
                    C = np.concatenate(sel_c)
                    order = np.argsort(C, kind="stable")  # rows already ascending → canonical CSC
                    data_block = np.concatenate(sel_v)[order]
                    idx_block = np.concatenate(sel_r)[order]
                    sel_r = sel_c = sel_v = C = order = None
                    writer.write_block(data_block, idx_block)
                    # per-column sums for this band, float64-accumulated == (cube.T @ ones)[c0:c1]
                    local = indptr[c0:c1 + 1] - indptr[c0]
                    nzc = np.where(np.diff(local) > 0)[0]
                    if nzc.size:
                        colsum[c0 + nzc] = np.add.reduceat(data_block.astype(np.float64), local[nzc])
                    data_block = idx_block = None
                    if progress:
                        progress(n + int((bi + 1) / nbands * n), 2 * n)
            finally:
                for rd in readers.values():
                    try:
                        rd.close()
                    except Exception:  # noqa: BLE001
                        pass
            mean = self._mean if getattr(self, "_mean", None) is not None else None
            pix = self._pix if getattr(self, "_pix", None) is not None else None
            store = writer.finalize(colsum, mean=mean, pix=pix)
            writer = None
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        finally:
            for h in handles.values():
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
            for p in paths.values():
                try:
                    os.remove(p)
                except OSError:
                    pass
        if progress:
            progress(2 * n, 2 * n)
        self._close_cube()
        self._cube = store
        self._cache_gen += 1
        return self._cube

    def _close_cube(self):
        """Release an on-disk :class:`CubeStore`'s file handle before ``_cube`` is dropped
        or replaced (rebuild / preprocessing change / release). A no-op for the in-RAM
        tuple. On Windows a lingering open handle would block the atomic replace of a
        rebuilt cube, so close deterministically rather than waiting on GC."""
        cube = getattr(self, "_cube", None)
        if isinstance(cube, CubeStore):
            cube.close()

    def _cube_ion(self, mz, tol_ppm, reduce):
        # A lazily-read on-disk store (reopened slide): stream only the chunks the ion
        # window touches. Identical math to the in-RAM branch below.
        if isinstance(self._cube, CubeStore):
            return self._cube.ion(mz, tol_ppm, reduce, self.n_pixels)
        axis, cube, _ = self._cube
        win = mz * tol_ppm / 1e6
        a = np.searchsorted(axis, mz - win, side="left")
        b = np.searchsorted(axis, mz + win, side="right")
        if b <= a:
            return np.zeros(self.n_pixels)
        sub = cube[:, a:b]
        if reduce == "max":
            return np.asarray(sub.max(axis=1).todense()).ravel()
        s = np.asarray(sub.sum(axis=1)).ravel()
        return s / (b - a) if reduce == "mean" else s

    # ----- ion images ------------------------------------------------------ #
    def ion_vector(self, mz: float, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                   norm: str = "none") -> np.ndarray:
        """Per-pixel intensity at ``mz`` (+/- tol). Served from the cached feature
        matrix when the m/z is a picked peak; else the sparse cube (if built); else
        one streaming pass (LRU-cached)."""
        j = self._feature_index(mz)
        if j is not None and self._feat.reduce == reduce:
            vals = self._feat.matrix[:, j].astype(float)
        elif self._dense() is not None and self.store.shared_axis() is not None:
            # Resident in RAM (to_ram / demo / imaging-table) with no active transforms:
            # slice the arbitrary-m/z window straight out of the dense matrix in one
            # vectorized reduction. This is exact (full-resolution, not the binned cube)
            # and — crucially — replaces the per-pixel _stream_ion Python loop that would
            # otherwise run on the GUI thread and freeze the app ("not responding") on a
            # large slide. No cube is needed (and none is built for in-RAM stores).
            vals = self._dense_ion(mz, tol_ppm, reduce)
        elif self._cube is not None:
            vals = self._cube_ion(mz, tol_ppm, reduce)
        else:
            vals = self._cached_stream_ion(mz, tol_ppm, reduce)
        return vals / self.norm_factors(norm)

    def _dense_ion(self, mz, tol_ppm, reduce):
        """Arbitrary-m/z ion vector as a vectorized column-range reduction of the in-RAM
        dense matrix — no per-pixel loop, no cube, no disk. Mathematically identical to
        :meth:`_stream_ion` for the same window/reduce, just vectorized."""
        M = self._dense()
        axis = self.store.shared_axis()
        win = mz * tol_ppm / 1e6
        a = int(np.searchsorted(axis, mz - win, side="left"))
        b = int(np.searchsorted(axis, mz + win, side="right"))
        if b <= a:
            return np.zeros(self.n_pixels)
        sub = M[:, a:b]
        if reduce == "max":
            return np.asarray(sub.max(axis=1), dtype=float)
        s = np.asarray(sub.sum(axis=1, dtype=np.float64))
        return s / (b - a) if reduce == "mean" else s

    def _cached_stream_ion(self, mz, tol_ppm, reduce):
        """LRU cache (cap 256) for streamed arbitrary-m/z ion vectors, so revisiting
        an m/z is instant even when it's not a picked-peak feature."""
        key = (round(float(mz), 5), float(tol_ppm), reduce)
        cached = self._ion_cache.get(key)
        if cached is not None:
            return cached
        vals = self._stream_ion(mz, tol_ppm, reduce)
        if len(self._ion_cache) >= 256:
            self._ion_cache.pop(next(iter(self._ion_cache)))
        self._ion_cache[key] = vals
        return vals

    def _stream_ion(self, mz, tol_ppm, reduce):
        win = mz * tol_ppm / 1e6
        lo, hi = mz - win, mz + win
        vals = np.zeros(self.n_pixels)
        # Read through the parallel _read_many pool (disk-read+decode overlap across cores)
        # instead of a serial per-pixel _read — this is the loop that froze the GUI on a
        # no-cube slide and that estimate_fdr's off-peak decoys hammer. Numerically identical
        # (same per-pixel window reduction, results yielded in pixel order).
        for i, m, inten in self._read_many(range(self.n_pixels)):
            a = np.searchsorted(m, lo, side="left")
            b = np.searchsorted(m, hi, side="right")
            if b > a:
                seg = inten[a:b]
                vals[i] = seg.sum() if reduce == "sum" else (
                    seg.max() if reduce == "max" else seg.mean())
        return vals

    def ion_image(self, mz: float, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                  norm: str = "none") -> np.ndarray:
        # Non-acquired cells are NaN (no data), not 0 — otherwise the empty area of a
        # sparse acquisition (tissue a small fraction of the grid extent) renders as the
        # colormap floor, painting a solid colour 'box' behind the data instead of being
        # transparent. Acquired pixels that are genuinely 0 still read as 0.
        return self.to_image(self.ion_vector(mz, tol_ppm, reduce, norm), fill=np.nan)

    def composite_vector(self, mz_list, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                         norm: str = "none", weight: str = "raw") -> np.ndarray:
        """Sum the per-pixel ion vectors of several m/z (e.g. all ions of a lipid
        class) into one distribution.

        ``weight`` controls how member ions combine:

        * ``"raw"`` — sum raw intensities (a true total-abundance map). The most
          abundant member dominates, so faint members can be invisible.
        * ``"balanced"`` — scale each member to its own 99th-percentile before
          summing, so every ion contributes comparably. This maps *where the
          class is present* rather than its total abundance."""
        acc = np.zeros(self.n_pixels)
        for mz in mz_list:
            v = self.ion_vector(mz, tol_ppm=tol_ppm, reduce=reduce, norm=norm)
            if weight == "balanced":
                pos = v[v > 0]
                scale = float(np.percentile(pos, 99)) if pos.size else 0.0
                if scale > 0:
                    v = v / scale
            acc += v
        return acc

    def composite_image(self, mz_list, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                        norm: str = "none", weight: str = "raw") -> np.ndarray:
        """Composite ion image — e.g. a 'total sulfatide' or 'total ganglioside' map."""
        return self.to_image(self.composite_vector(mz_list, tol_ppm, reduce, norm, weight),
                             fill=np.nan)

    def ratio_image(self, num_mz: float, den_mz: float, tol_ppm: float = DEFAULT_TOL_PPM,
                    norm: str = "none", eps: float = 1.0) -> np.ndarray:
        """Ratio of two ion images, ``num/den`` (e.g. sulfatide/PC for myelin contrast).
        ``eps`` stabilizes pixels where the denominator is ~0."""
        num = self.ion_vector(float(num_mz), tol_ppm=tol_ppm, norm=norm)
        den = self.ion_vector(float(den_mz), tol_ppm=tol_ppm, norm=norm)
        return self.to_image(num / (den + eps), fill=np.nan)

    def tic(self) -> np.ndarray:
        """Per-pixel total ion current (computed during :meth:`prime`)."""
        self.prime()
        return self._pix["tic"]

    def tic_image(self) -> np.ndarray:
        self.prime()
        return self.to_image(self._pix["tic"])

    # ----- peak picking ---------------------------------------------------- #
    def is_centroided(self) -> bool:
        """Whether the data is centroided (vs profile). Delegates to the intake inspector
        (:func:`smile_msi.intake.detect_representation`), which reconciles the imzML
        spec_mode flag with a data heuristic and lets the data win on disagreement.
        Cached; defaults to profile (the safer general picker) when undetermined."""
        cached = getattr(self, "_is_centroided", None)
        if cached is not None:
            return cached
        try:
            from . import intake
            rep, _, _ = intake.detect_representation(self)
            val = (rep == "centroid")
        except Exception:  # noqa: BLE001 - never let detection break peak picking
            val = (self.spec_mode or "").strip().lower() == "centroid"
        self._is_centroided = val
        return val

    def pick_peaks(self, snr: float = 3.0, min_rel_intensity: float = 0.0,
                   max_peaks: int = 500, mask=None, projection: str = "mean",
                   prominence: float = 1.0, local_noise: bool = True,
                   centroided: bool | None = None):
        """Detect peaks in the mean (``projection='mean'``) or skyline/maximum
        (``projection='max'``) spectrum -> list of ``{mz, intensity, rel_intensity,
        snr}`` sorted by descending intensity.

        Two pipelines, chosen by data type (``centroided`` overrides auto-detection):

        * **Profile** (continuous trace) — local-maxima + signal-to-noise, the
          MALDIquant / Cardinal ``peakPick`` standard, reimplemented (not copied) from
          the papers on top of SciPy's ``find_peaks``. Noise (hence the S/N threshold) is
          estimated **per m/z window** by default (``local_noise=True``).

          Note this path applies **no baseline subtraction**, so the height test and the
          reported ``snr`` compare the **absolute** intensity to the noise *scale*
          (``intensity / noise``), not the baseline-relative ``(intensity − baseline) /
          noise`` of MALDIquant's / Cardinal's ``mad`` convention. On data with a strong
          continuum baseline the reported ``snr`` is therefore inflated by that baseline;
          ``prominence`` (in S/N units, requiring a peak to rise above its surrounding
          saddle; set ``0`` to disable) is the only genuinely baseline-relative gate. To
          match the cited S/N exactly, run a baseline step first (``preprocess`` baseline
          operators) so the trace is already baseline-corrected.
        * **Centroided** (already peak-picked) — there is no continuous baseline to run a
          profile finder against, so we **skip peak detection** and threshold the recorded
          centroids directly (Cardinal errors if you run ``peakPick`` on centroided input;
          the standard path goes straight to thresholding → alignment/binning). See
          :meth:`_pick_centroids`.

        Refs: peak finder — Virtanen et al. (2020, SciPy, BSD-3), doi:10.1038/s41592-019-0686-2;
        MAD × 1.4826 normal-consistency constant — Hampel (1974), doi:10.1080/01621459.1974.10482962;
        local-maxima+SNR convention — MALDIquant (Gibb & Strimmer 2012) / Cardinal
        (Bemis et al. 2015), reimplemented from publications, no GPL/Artistic source used.
        """
        from scipy.signal import find_peaks

        axis, spec = (self.max_spectrum(mask=mask) if projection == "max"
                      else self.mean_spectrum(mask=mask))
        if not len(spec) or spec.max() <= 0:
            return []
        if centroided is None:
            centroided = self.is_centroided()
        if centroided:
            return self._pick_centroids(axis, spec, snr, min_rel_intensity, max_peaks)
        noise = _local_noise(spec) if local_noise else np.full(len(spec), _mad(spec))
        height = np.maximum(snr * noise, min_rel_intensity * spec.max())
        # distance=3: a shallow one-bin dip at a strong peak's apex — detector
        # quantization/saturation ripple — otherwise reads as two adjacent local maxima
        # that both clear the height/prominence gate (the gate is a per-window noise
        # estimate, not a saddle-depth test), producing twin near-identical peaks that
        # then annotate to the same lipid and show up as duplicate feature-list rows.
        # Genuinely distinct ions are never this close on a profile-sampled axis.
        kw = {"height": np.where(height > 0, height, 1e-12), "distance": 3}
        if prominence and prominence > 0:
            prom = prominence * noise
            kw["prominence"] = np.where(prom > 0, prom, 1e-12)
        idx, _ = find_peaks(spec, **kw)
        base = spec.max()
        peaks = [{
            "mz": float(_centroid(axis, spec, i)),
            "intensity": float(spec[i]),
            "rel_intensity": float(spec[i] / base),
            "snr": float(spec[i] / noise[i]) if noise[i] > 0 else float("inf"),
        } for i in idx]
        peaks.sort(key=lambda d: d["intensity"], reverse=True)
        return peaks[:max_peaks]

    def _pick_centroids(self, axis, spec, snr, min_rel_intensity, max_peaks):
        """Peak 'picking' for already-centroided data: the recorded points **are** the
        peaks, so we don't run a profile peak-finder — we keep each centroid above an
        intensity / S-N floor. This is the centroided branch of the standard pipeline
        (Cardinal: ``peakPick`` is profile→centroid and refuses centroided input; you go
        straight to thresholding then ``peakAlign``/``peakFilter``, which here is the
        downstream tolerance-binning + frequency filter in
        :func:`spatial.find_spatial_features`).

        Unlike the profile path this keeps the **exact recorded m/z** of each centroid (no
        weighted ``_centroid`` re-estimate, which assumes a multi-sample hump). The noise
        floor is a single robust MAD over the centroid heights (MALDIquant-style, from
        Gibb & Strimmer 2012; reimplemented, no source copied)."""
        nz = np.flatnonzero(spec > 0)
        if nz.size == 0:
            return []
        base = float(spec[nz].max())
        noise = _mad(spec)                       # robust height floor over the centroids
        thresh = max(snr * noise, min_rel_intensity * base)
        keep = nz[spec[nz] >= thresh] if thresh > 0 else nz
        peaks = [{
            "mz": float(axis[i]),
            "intensity": float(spec[i]),
            "rel_intensity": float(spec[i] / base),
            "snr": float(spec[i] / noise) if noise > 0 else float("inf"),
        } for i in keep]
        peaks.sort(key=lambda d: d["intensity"], reverse=True)
        return peaks[:max_peaks]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _safe_meta(parser, kind: str) -> str:
    try:
        if kind == "polarity":
            val = getattr(parser, "polarity", "") or ""
        else:
            val = getattr(parser, "spectrum_mode", "") or getattr(parser, "spec_type", "") or ""
        return str(val).lower()
    except Exception:  # noqa: BLE001
        return ""


def _positive_median(inten) -> float:
    """Median over the *positive* intensities (the canonical MSI median-norm factor); 0.0 if
    there are none. Shared by prime() and features_for_rows so the streamed-subset path and
    the whole-slide path compute the same median factor."""
    pos = inten[inten > 0]
    return float(np.median(pos)) if len(pos) else 0.0


def _pixel_stats(inten):
    """Per-pixel (TIC, RMS, positive-median) for one spectrum's intensities — the normalization
    stats prime() accumulates, factored out of its two streaming branches."""
    tic = float(inten.sum())
    rms = float(np.sqrt(np.mean(inten ** 2))) if len(inten) else 0.0
    return tic, rms, _positive_median(inten)


def _reduce_windows(inten, a, b, reduce, out_row) -> None:
    """Reduce ``inten`` over each ``[a[j], b[j])`` peak window into ``out_row`` (sum/max/mean);
    empty windows are left untouched (0). The shared per-pixel inner loop of build_features
    and features_for_rows."""
    for j in range(len(a)):
        if b[j] > a[j]:
            seg = inten[a[j]:b[j]]
            out_row[j] = (seg.sum() if reduce == "sum"
                          else seg.max() if reduce == "max" else seg.mean())


def _finalize_norm(raw, max_amp, scale_from=None):
    """Turn raw per-pixel normalization factors (TIC/RMS/median) into per-pixel divisors:
    scale by the mean of the *positive* factors, floor non-positive pixels at that scale so
    they can't blow up, then cap the low-signal amplification at ``max_amp`` (<=0 disables).

    The cap matters because dividing by a near-zero per-pixel factor turns a faint
    background/edge pixel into a fake hotspot that hijacks the colour scale (the artifact
    SCiLS sidesteps by excluding such pixels). Shared by :meth:`MSIDataset.norm_factors`
    (whole-slide, cached ``_pix``) and the streamed-subset path of
    :meth:`MSIDataset.features_for_rows`.

    ``scale_from`` chooses the population the mean scale is computed over, independent of the
    pixels being normalized. The divisor is ``TIC_pixel / mean(TIC over scale_from)``, so that
    mean enters every normalized intensity as one multiplicative constant per slide. Left as
    the whole slide (the default) it depends on how much empty matrix the section happens to
    sit in; pointed at the on-tissue pixels instead, it doesn't. Cross-sample comparisons
    care, because that constant differs between slides.
    """
    raw = np.asarray(raw, dtype=float)
    src = raw if scale_from is None else np.asarray(scale_from, dtype=float)
    pos = src[src > 0]
    scale = float(pos.mean()) if pos.size else 1.0
    out = np.where(raw > 0, raw, scale) / scale
    if max_amp > 0:
        np.maximum(out, 1.0 / max_amp, out=out)
    return out


def _mask_key(mask: np.ndarray) -> str:
    """Stable, compact cache key for a boolean mask: a blake2b digest over the
    bit-packed mask plus a shape/popcount guard. ~16 bytes regardless of slide
    size, versus the previous ``mask.tobytes()`` key that allocated ~1 byte/pixel
    on every call (including cache hits) and retained ~1 MB per LRU entry. The
    digest is over the *full* packed mask (not a truncated Python ``hash``), so
    collisions are cryptographically negligible for the bounded 8-entry cache."""
    packed = np.packbits(mask.ravel())
    h = hashlib.blake2b(packed.tobytes(), digest_size=16)
    h.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    h.update(np.int64(int(mask.sum())).tobytes())
    return h.hexdigest()


def _bin_edges(axis: np.ndarray) -> np.ndarray:
    mids = (axis[:-1] + axis[1:]) / 2
    first = axis[0] - (mids[0] - axis[0]) if len(mids) else axis[0]
    last = axis[-1] + (axis[-1] - mids[-1]) if len(mids) else axis[-1]
    return np.concatenate(([first], mids, [last]))


def _mad(x: np.ndarray) -> float:
    """Robust scale estimate: MAD × 1.4826 (= 1/Φ⁻¹(3/4), the normal-consistency
    constant), over positive values. The 1.4826 standardization of the MAD as a
    consistent σ estimator is Hampel (1974), J. Am. Stat. Assoc. 69(346):383–393,
    doi:10.1080/01621459.1974.10482962 (cf. Rousseeuw & Croux 1993 for Sn/Qn
    alternatives)."""
    pos = x[x > 0]
    if not len(pos):
        return 0.0
    med = np.median(pos)
    return float(np.median(np.abs(pos - med)) * 1.4826)


def _local_noise(spec: np.ndarray, n_windows: int = 32) -> np.ndarray:
    """Per-sample noise estimate: robust MAD computed in ``n_windows`` m/z bins and
    interpolated back to full length, so the S/N threshold tracks the local baseline
    (empty windows inherit the median window noise)."""
    n = len(spec)
    if n == 0:
        return np.zeros(0)
    edges = np.linspace(0, n, min(n_windows, n) + 1).astype(int)
    centers, vals = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            centers.append((a + b) / 2.0)
            vals.append(_mad(spec[a:b]))
    vals = np.asarray(vals, float)
    pos = vals[vals > 0]
    fill = float(np.median(pos)) if len(pos) else float(_mad(spec))
    vals = np.where(vals > 0, vals, fill)
    if len(centers) < 2:
        return np.full(n, vals[0] if len(vals) else fill)
    return np.interp(np.arange(n), np.asarray(centers, float), vals)


def _centroid(axis: np.ndarray, spec: np.ndarray, i: int) -> float:
    lo, hi = max(0, i - 1), min(len(axis), i + 2)
    w = spec[lo:hi]
    if w.sum() <= 0:
        return float(axis[i])
    return float(np.average(axis[lo:hi], weights=w))
