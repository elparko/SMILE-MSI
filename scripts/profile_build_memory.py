"""Profile build_mz_cube's *peak transient* RAM vs the final cube size — the Phase-2
question: does assembling the cube hold a multiple of it in RAM at once?

Uses an on-the-fly lazy store (spectra generated per read, never all resident) so the
measured growth is the build's transient, not the input. A watcher thread samples VmRSS
so we capture the peak during the parts-list accumulation + vstack, not just the endpoints.

    .venv/bin/python scripts/profile_build_memory.py [n_pixels] [n_channels] [per_px]
"""
import sys
import time
import threading

import numpy as np

sys.path.insert(0, ".")
from smile_msi.msi import MSIDataset, SpectrumStore  # noqa: E402


class LazySynthStore(SpectrumStore):
    """Continuous-mode (shared-axis) store that synthesises each spectrum on read, so the
    whole dataset is never resident — mimicking a lazy imzML well enough for memory profiling."""
    supports_parallel = False
    in_memory = False

    def __init__(self, n_pixels, n_channels, per_px, seed=0):
        side = int(np.ceil(np.sqrt(n_pixels)))
        coords = np.array([(x, y) for y in range(1, side + 1)
                           for x in range(1, side + 1)])[:n_pixels]
        self.coordinates = np.asarray(coords, dtype=int)
        self.axis = np.linspace(150.0, 1100.0, n_channels)
        self.n = n_pixels
        self.per_px = per_px
        self.seed = seed

    def __len__(self):
        return self.n

    def get(self, i):
        rng = np.random.RandomState(self.seed + int(i))
        v = np.zeros(self.axis.shape[0], dtype=np.float64)
        idx = rng.randint(0, self.axis.shape[0], self.per_px)
        v[idx] = rng.rand(self.per_px) * 1000.0
        return self.axis, v

    def shared_axis(self):
        return self.axis


def _vmrss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


class Sampler(threading.Thread):
    def __init__(self, dt=0.002):
        super().__init__(daemon=True)
        self.dt = dt
        self.peak = 0.0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            self.peak = max(self.peak, _vmrss_mb())
            time.sleep(self.dt)

    def stop(self):
        self._stop_evt.set()
        self.join()


def _run_child(mode, n_pixels, n_channels, per_px):
    import os
    import tempfile

    ds = MSIDataset(LazySynthStore(n_pixels, n_channels, per_px))
    _ = ds.mean_spectrum()                       # prime the shared axis path like the GUI does
    base = _vmrss_mb()

    s = Sampler()
    s.start()
    t0 = time.perf_counter()
    if mode == "stream":
        zp = os.path.join(tempfile.mkdtemp(), "b.cube.zarr")
        cube = ds.build_mz_cube(min_intensity=0.0, out_path=zp, fingerprint="prof")
        nnz = cube.nnz
        csc_mb = nnz * 8 / 1e6                    # data(4) + indices(4) per entry
    else:
        _, cube, _ = ds.build_mz_cube(min_intensity=0.0)
        nnz = cube.nnz
        csc_mb = (cube.data.nbytes + cube.indices.nbytes + cube.indptr.nbytes) / 1e6
    dt = time.perf_counter() - t0
    s.stop()
    after = _vmrss_mb()
    print(f"{mode}\t{nnz}\t{csc_mb:.0f}\t{base:.0f}\t{after:.0f}\t{s.peak:.0f}\t{dt:.1f}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        _run_child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
        return

    import subprocess

    n_pixels = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    n_channels = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    per_px = int(sys.argv[3]) if len(sys.argv) > 3 else 600

    print("mode\tnnz\tcubeMB\tbaseRSS\tafterRSS\tPEAK\tsecs")
    rows = {}
    for mode in ("inram", "stream"):
        out = subprocess.run([sys.executable, __file__, "--child", mode,
                              str(n_pixels), str(n_channels), str(per_px)],
                             check=True, capture_output=True, text=True).stdout.strip()
        print(out)
        rows[mode] = out.split("\t")
    cube_mb = float(rows["inram"][2])
    pk_in = float(rows["inram"][5]) - float(rows["inram"][3])     # peak above baseline
    pk_st = float(rows["stream"][5]) - float(rows["stream"][3])
    print(f"\ncube ~{cube_mb:.0f} MB | build PEAK above baseline:  "
          f"in-RAM {pk_in:.0f} MB ({pk_in/max(cube_mb,1):.1f}x)  ->  "
          f"stream {pk_st:.0f} MB ({pk_st/max(cube_mb,1):.2f}x)   "
          f"= {pk_in/max(pk_st,1):.1f}x less")


if __name__ == "__main__":
    main()
