"""Profile warm-reopen peak RSS + ion latency: lazy Zarr CubeStore vs the legacy npz
(full-CSC-in-RAM) cube cache. Run with the project venv:

    .venv/bin/python scripts/profile_cubestore.py [n_pixels] [nbins]

The parent builds one synthetic sparse cube, persists it both ways (npz + .cube.zarr),
then spawns two *isolated* child processes — one loading the npz (materialises the whole
CSC), one loading the Zarr store (keeps only axis/indptr in RAM, reads chunks on demand).
Each child reports its own peak RSS (ru_maxrss) and the wall-time of 20 ion-image reads,
so the RSS delta isolates the cube's resident memory.
"""
import os
import sys
import time
import resource
import tempfile
import subprocess

import numpy as np
from scipy import sparse


def _peak_rss_mb():
    # ru_maxrss is KiB on Linux, bytes on macOS.
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0 if sys.platform != "darwin" else kb / (1024.0 * 1024.0)


def _cur_rss_mb():
    """Current resident set (VmRSS), MB — Linux. Lets us measure the *growth* from loading
    the cube, isolating it from the shared interpreter/numpy/scipy/zarr baseline."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return _peak_rss_mb()


def _make_cube(n_pixels, nbins, per_px, seed=0):
    rng = np.random.RandomState(seed)
    rows = np.repeat(np.arange(n_pixels), per_px)
    cols = rng.randint(0, nbins, size=n_pixels * per_px)
    vals = (rng.rand(n_pixels * per_px) * 1000.0).astype(np.float32)
    csc = sparse.csr_matrix((vals, (rows, cols)), shape=(n_pixels, nbins)).tocsc()
    axis = np.sort(rng.rand(nbins)).cumsum().astype(np.float64) + 100.0
    edges = np.concatenate([[axis[0] - 1.0], (axis[:-1] + axis[1:]) / 2.0, [axis[-1] + 1.0]])
    return axis, csc, edges


def _child(mode, path, n_pixels, nbins, axis_npy):
    """Run in a fresh process: load the cube one way, do 20 ion reads, print RSS + timing."""
    axis = np.load(axis_npy)
    rng = np.random.RandomState(123)
    mzs = axis[rng.randint(nbins // 20, nbins - 2, size=20)]
    tol = 60.0
    # import the read deps first so the baseline reflects a ready-to-read process, then
    # measure the resident growth that *loading the cube* adds on top of that baseline.
    from smile_msi.cubestore import CubeStore  # noqa: F401 (warms zarr/numcodecs for both)
    base = _cur_rss_mb()

    if mode == "npz":
        with np.load(path) as z:
            csc = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]),
                                    shape=tuple(int(x) for x in z["shape"]))
        csc_axis = axis

        def ion(mz):
            win = mz * tol / 1e6
            a = np.searchsorted(csc_axis, mz - win, "left")
            b = np.searchsorted(csc_axis, mz + win, "right")
            if b <= a:
                return np.zeros(n_pixels)
            sub = csc[:, a:b]
            return np.asarray(sub.sum(axis=1)).ravel()
    else:
        store = CubeStore.open(path, fingerprint="prof", n_pixels=n_pixels)

        def ion(mz):
            return store.ion(mz, tol, "sum", n_pixels)

    loaded = _cur_rss_mb()
    # warm one read (touch chunks / page in), then time 20
    ion(mzs[0])
    t0 = time.perf_counter()
    for mz in mzs:
        ion(float(mz))
    dt = (time.perf_counter() - t0) / len(mzs) * 1e3   # ms per ion read
    after = _cur_rss_mb()

    # NB: resource.getrusage(ru_maxrss) is unreliable here — in a container it can report a
    # cgroup-wide peak shared across processes — so report current VmRSS deltas instead.
    print(f"{mode}\t{base:.1f}\t{loaded:.1f}\t{after:.1f}\t{loaded - base:+.1f}\t{dt:.2f}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        _, _, mode, path, npx, nb, axis_npy = sys.argv
        _child(mode, path, int(npx), int(nb), axis_npy)
        return

    n_pixels = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    nbins = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    per_px = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    from smile_msi import session
    from smile_msi.cubestore import CubeStore

    d = tempfile.mkdtemp()
    axis, csc, edges = _make_cube(n_pixels, nbins, per_px)
    nnz = csc.nnz
    dense_gb = n_pixels * nbins * 4 / 1e9
    csc_mb = (csc.data.nbytes + csc.indices.nbytes + csc.indptr.nbytes) / 1e6
    print(f"cube: {n_pixels} px x {nbins} bins, nnz={nnz:,} "
          f"({nnz / (n_pixels * nbins) * 100:.2f}% dense)")
    print(f"  dense float32 would be {dense_gb:.1f} GB; CSC arrays = {csc_mb:.0f} MB resident\n")

    npz_path = os.path.join(d, "c.cache.npz")
    np.savez(npz_path, data=csc.data, indices=csc.indices, indptr=csc.indptr,
             shape=np.asarray(csc.shape, dtype=np.int64), axis=axis, edges=edges,
             fingerprint=np.array("prof"))
    zarr_path = os.path.join(d, "c.cube.zarr")
    CubeStore.create(zarr_path, axis, csc, edges, fingerprint="prof",
                     n_pixels=n_pixels).close()
    print(f"npz size  = {os.path.getsize(npz_path) / 1e6:.0f} MB")
    print(f"zarr size = {os.path.getsize(zarr_path) / 1e6:.0f} MB (Blosc/zstd compressed)\n")

    axis_npy = os.path.join(d, "axis.npy")
    np.save(axis_npy, axis)

    print("mode\tbaseRSS\tloadedRSS\tafterRSS\tΔload\tion(ms/read)")
    for mode, path in (("npz", npz_path), ("zarr", zarr_path)):
        subprocess.run([sys.executable, __file__, "--child", mode, path,
                        str(n_pixels), str(nbins), axis_npy], check=True)
    print("\nΔload = resident growth from loading the cube (the headline: npz materialises "
          "the whole CSC; zarr keeps only axis/indptr + reads chunks on demand). RSS in MB.")


if __name__ == "__main__":
    main()
