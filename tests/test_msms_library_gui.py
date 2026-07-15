"""GUI smoke test for the MS/MS library-match handler (plan 04, headless)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from smile_msi.gui import features as feat_mod  # noqa: E402
from smile_msi.gui.features import FeaturesTabMixin  # noqa: E402

_PEAKS = "184.0733 100\n86.0964 40\n104.1070 25\n"


def _write(p, name, prec):
    p.write_text(f"Name: {name}\nPrecursorMZ: {prec}\nNum Peaks: 3\n{_PEAKS}")
    return str(p)


class _FakeBar:
    def __init__(self):
        self.msg = ""

    def showMessage(self, m):
        self.msg = m


class _FakeTab:
    def __init__(self, df):
        self.feat_df = df
        self._bar = _FakeBar()
        self.rendered = False

    def statusBar(self):
        return self._bar

    def _render_feature_table(self):
        self.rendered = True


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    return tmp_path


def test_do_msms_library_writes_top_hit(home, tmp_path, monkeypatch):
    acq = _write(tmp_path / "acq.msp", "unknown", "760.585")
    lib = _write(tmp_path / "lib.msp", "PC 34:1 [M+H]+", "760.585")
    paths = iter([(acq, ""), (lib, "")])
    monkeypatch.setattr(feat_mod.filedialogs, "get_open_file_name",
                        lambda *a, **k: next(paths))

    tab = _FakeTab(pd.DataFrame({"mz": [760.585], "class": ["PC"]}))
    FeaturesTabMixin.do_msms_library(tab)

    assert tab.rendered
    assert "msms_lib" in tab.feat_df.columns
    assert "PC 34:1" in tab.feat_df["msms_lib"].iloc[0]   # matched the library entry
    assert "MS/MS library" in tab._bar.msg


def test_do_msms_library_no_feat_df():
    tab = _FakeTab(None)
    FeaturesTabMixin.do_msms_library(tab)                 # graceful: prompts to build first
    assert "Build the feature list" in tab._bar.msg
