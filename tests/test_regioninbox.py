import os

import numpy as np

from smile_msi import regioninbox


def _session(tmp_path):
    p = tmp_path / "slide__abc.json"
    p.write_text("{}")
    return str(p)


def test_posted_regions_come_back_once_with_their_masks(tmp_path):
    sp = _session(tmp_path)
    mask = np.zeros(50, bool)
    mask[[3, 7, 40]] = True
    regioninbox.post_regions(sp, [{"name": "core", "mask": mask, "group": "Group A",
                                   "parent": "Nerve 1"}], n_pixels=50)
    regions, problems = regioninbox.take_regions(sp, 50)
    assert problems == []
    assert [r["name"] for r in regions] == ["core"]
    assert regions[0]["group"] == "Group A"
    assert regions[0]["parent"] == "Nerve 1"
    assert np.array_equal(regions[0]["mask"], mask)
    assert regioninbox.take_regions(sp, 50) == ([], [])


def test_a_file_for_another_pixel_count_is_set_aside_not_applied(tmp_path):
    sp = _session(tmp_path)
    path = regioninbox.post_regions(sp, [{"name": "x", "mask": np.ones(10, bool)}], n_pixels=10)
    regions, problems = regioninbox.take_regions(sp, 12)
    assert regions == [] and len(problems) == 1
    assert not os.path.exists(path) and os.path.exists(path[:-5] + ".rejected")
    assert regioninbox.pending(sp) == []


def test_a_half_written_file_is_not_picked_up(tmp_path):
    sp = _session(tmp_path)
    os.makedirs(regioninbox.inbox_dir(sp))
    (tmp_path / "slide__abc.inbox" / "regions-x.tmp").write_text("{")
    assert regioninbox.pending(sp) == []


def test_the_inbox_sits_beside_the_session_and_leaves_it_alone(tmp_path):
    sp = _session(tmp_path)
    regioninbox.post_regions(sp, [{"name": "x", "mask": np.ones(4, bool)}], n_pixels=4)
    assert regioninbox.inbox_dir(sp) == str(tmp_path / "slide__abc.inbox")
    assert (tmp_path / "slide__abc.json").read_text() == "{}"
