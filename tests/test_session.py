"""Tests for session.py store helpers (cache inventory, deletion) and the home redirect."""
import json
import os

import numpy as np

from smile_msi import library, session


def _write_session(d, stem, source, fp="n100:w10:h10:mz100.00-900.00:abcdef012345",
                   n_pixels=100, regions=0):
    data = {"source": source, "dataset_fingerprint": fp, "n_pixels": n_pixels,
            "named_regions": [{"name": f"r{i}"} for i in range(regions)]}
    with open(os.path.join(d, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_cache_inventory_classifies_store_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    d = session.sessions_dir()
    src = tmp_path / "slideA.imzML"; src.write_bytes(b"x")
    # live slide: session + cube + runs dir
    _write_session(d, "slideA__11111111", str(src), regions=2)
    (tmp_path / "sessions" / "slideA__11111111.cube.zarr").write_bytes(b"z" * 10)
    os.makedirs(tmp_path / "sessions" / "slideA__11111111.runs")
    # older-fingerprint duplicate of the same slide (fewer regions), with its own cube
    _write_session(d, "slideA__22222222", str(src), fp="n100:w10:h10:mz100.00-950.00:abcdef012345",
                   regions=0)
    (tmp_path / "sessions" / "slideA__22222222.cube.zarr").write_bytes(b"z" * 20)
    # orphaned cube, legacy npz beside a zarr, interrupted-build leftovers
    (tmp_path / "sessions" / "gone__33333333.cube.zarr").write_bytes(b"z" * 30)
    (tmp_path / "sessions" / "slideA__11111111.cache.npz").write_bytes(b"n" * 5)
    (tmp_path / "sessions" / ".cube_spill_03_abc.bin").write_bytes(b"s" * 7)
    (tmp_path / "sessions" / "slideA__11111111.cube.zarr.tmp").write_bytes(b"t")
    # a session whose source is gone: kept, reported
    _write_session(d, "moved__44444444", str(tmp_path / "nowhere.imzML"),
                   fp="n50:w5:h10:mz100.00-900.00:0123456789ab", n_pixels=50)
    # unreadable JSON
    (tmp_path / "sessions" / "broken__55555555.json").write_text("{not json")

    rows = {r["name"]: r for r in session.cache_inventory()}
    assert rows["slideA__11111111.json"]["status"] == "in use" and not rows["slideA__11111111.json"]["removable"]
    assert rows["slideA__11111111.cube.zarr"]["status"] == "in use"
    assert rows["slideA__11111111.runs"]["kind"] == "runs" and rows["slideA__11111111.runs"]["status"] == "in use"
    assert rows["slideA__22222222.json"]["removable"] and "duplicate" in rows["slideA__22222222.json"]["status"]
    assert rows["slideA__22222222.cube.zarr"]["removable"]
    assert rows["gone__33333333.cube.zarr"]["status"].startswith("orphaned")
    assert rows["slideA__11111111.cache.npz"]["status"].startswith("superseded")
    assert rows[".cube_spill_03_abc.bin"]["kind"] == "temporary" and rows[".cube_spill_03_abc.bin"]["removable"]
    assert rows["slideA__11111111.cube.zarr.tmp"]["kind"] == "temporary"
    assert rows["moved__44444444.json"]["status"].startswith("source missing")
    assert not rows["moved__44444444.json"]["removable"]
    assert rows["broken__55555555.json"]["status"] == "unreadable" and rows["broken__55555555.json"]["removable"]
    assert rows["gone__33333333.cube.zarr"]["size"] == 30

    removable = [r["path"] for r in rows.values() if r["removable"]]
    deleted, failed = session.delete_cache_paths(removable)
    assert not failed and set(deleted) == set(removable)
    left = {r["name"] for r in session.cache_inventory()}
    assert left == {"slideA__11111111.json", "slideA__11111111.cube.zarr", "slideA__11111111.runs",
                    "moved__44444444.json"}


def test_home_dir_redirect_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SMILE_MSI_HOME", raising=False)
    monkeypatch.setattr(library, "default_home_dir", lambda: str(tmp_path / "default"))
    assert library.home_dir() == str(tmp_path / "default")
    assert library.home_redirect() == ""
    library.set_home_redirect(str(tmp_path / "bigdisk"))
    assert library.home_redirect() == str(tmp_path / "bigdisk")
    assert library.home_dir() == str(tmp_path / "bigdisk") and os.path.isdir(tmp_path / "bigdisk")
    assert session.sessions_dir().startswith(str(tmp_path / "bigdisk"))
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "env"))
    assert library.home_dir() == str(tmp_path / "env")          # env var still wins
    monkeypatch.delenv("SMILE_MSI_HOME")
    library.set_home_redirect(None)
    assert library.home_redirect() == "" and library.home_dir() == str(tmp_path / "default")


def test_parse_cache_amend_keeps_arrays(tmp_path):
    imz = tmp_path / "t.imzML"; ibd = tmp_path / "t.ibd"
    imz.write_bytes(b"<x/>"); ibd.write_bytes(b"\x00" * 32)
    session.save_parse_cache(str(imz), mz_offsets=[16], mz_lengths=[1], int_offsets=[24],
                             int_lengths=[1], mz_precision="d", int_precision="f",
                             coordinates=[(1, 1)])
    session.amend_parse_cache(str(imz), mz_bounds=np.array([1.0, 2.0]))
    got = session.load_parse_cache(str(imz))
    assert got["coordinates"].tolist() == [[1, 1]] and got["mz_bounds"].tolist() == [1.0, 2.0]


def test_build_session_carries_preprocessing_and_calibration(tmp_path):
    cfg = {"recalibrate": {"refs": [885.5499], "tol_ppm": 40.0}}
    cal = {"source": "lock-mass", "refs": [885.5499], "verified": True, "median_after": 0.2}
    data = session.build_session(source="/x/s.imzML", settings={}, peaks=[],
                                 preprocessing=cfg, calibration=cal)
    assert data["preprocessing"] == cfg and data["calibration"] == cal
    p = str(tmp_path / "s.json")
    session.save_session(p, data)
    back = session.load_session(p)
    assert back["preprocessing"] == cfg and back["calibration"]["verified"] is True
    empty = session.build_session(source="/x/s.imzML", settings={}, peaks=[])
    assert empty["preprocessing"] is None and empty["calibration"] is None
