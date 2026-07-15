"""Tests for scripts/lmsd_to_csv.py — the LMSD .sdf → import-ready CSV converter.

Pure/text parsing + the element/mass/category filters, plus a round-trip through
``lipiddb.load_external_db`` so the emitted CSV really imports.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lmsd_to_csv.py"


@pytest.fixture(scope="module")
def lmsd():
    spec = importlib.util.spec_from_file_location("lmsd_to_csv", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A tiny SDF with the field shapes LMSD uses: a molblock header (ignored) then `> <FIELD>`
# value pairs, records delimited by $$$$. Includes a fluorine lipid (off-element) and a
# low-mass sterol so the filters have something to drop.
SDF = """
  Marvin  01010000002D

  0  0  0  0  0  0            999 V2000
M  END
> <LM_ID>
LMGP01010001

> <COMMON_NAME>
PC(16:0/18:1)

> <ABBREVIATION>
PC 34:1

> <FORMULA>
C42H82NO8P

> <CATEGORY>
Glycerophospholipids

> <MAIN_CLASS>
Glycerophosphocholines

$$$$

  Marvin  01010000002D

  0  0  0  0  0  0            999 V2000
M  END
> <LM_ID>
LMST01010001

> <COMMON_NAME>
Cholesterol

> <ABBREVIATION>
Cholesterol

> <FORMULA>
C27H46O

> <CATEGORY>
Sterol Lipids

> <MAIN_CLASS>
Sterols

$$$$

  Marvin  01010000002D

  0  0  0  0  0  0            999 V2000
M  END
> <ABBREVIATION>
FluoroStd 18:1

> <FORMULA>
C20H30F2O2

> <CATEGORY>
Fatty Acyls

$$$$
"""


def test_parse_sdf_extracts_data_fields(lmsd):
    recs = lmsd.parse_sdf(SDF)
    assert len(recs) == 3
    assert recs[0]["ABBREVIATION"] == "PC 34:1"
    assert recs[0]["FORMULA"] == "C42H82NO8P"
    assert recs[0]["MAIN_CLASS"] == "Glycerophosphocholines"
    assert recs[1]["COMMON_NAME"] == "Cholesterol"
    # the molblock (M END etc.) is not captured as a field
    assert "M  END" not in recs[0]


def test_to_rows_drops_off_element_by_default(lmsd):
    rows, counts = lmsd.to_rows(lmsd.parse_sdf(SDF))
    names = [r["ABBREVIATION"] for r in rows]
    assert "PC 34:1" in names and "Cholesterol" in names
    assert "FluoroStd 18:1" not in names          # F outside H,C,N,O,P,S,Na,K,Cl → dropped
    assert counts["off_element"] == 1 and counts["kept"] == 2


def test_to_rows_category_and_mass_filters(lmsd):
    recs = lmsd.parse_sdf(SDF)
    only_sterol, _ = lmsd.to_rows(recs, categories=["sterol"])
    assert [r["ABBREVIATION"] for r in only_sterol] == ["Cholesterol"]
    # PC 34:1 ~759 Da, cholesterol ~386 Da → a 700 Da floor keeps only the PC
    heavy, _ = lmsd.to_rows(recs, min_mass=700.0)
    assert [r["ABBREVIATION"] for r in heavy] == ["PC 34:1"]


def test_keep_all_elements_retains_off_element(lmsd):
    rows, _ = lmsd.to_rows(lmsd.parse_sdf(SDF), keep_all_elements=True)
    assert "FluoroStd 18:1" in [r["ABBREVIATION"] for r in rows]


def test_cli_roundtrips_into_load_external_db(lmsd, tmp_path):
    """End-to-end: SDF file → CLI → CSV → the app's importer parses it and annotates."""
    from smile_msi.lipiddb import load_external_db
    from smile_msi.masses import ion_mz
    from smile_msi.match import Annotator

    sdf = tmp_path / "lmsd.sdf"
    sdf.write_text(SDF, encoding="utf-8")
    out = tmp_path / "lmsd.csv"
    rc = lmsd.main([str(sdf), "-o", str(out)])
    assert rc == 0 and out.exists()

    db = load_external_db(str(out))               # the exact call the GUI importer makes
    names = {l.name for l in db}
    assert {"PC 34:1", "Cholesterol"} <= names and "FluoroStd 18:1" not in names
    ann = Annotator("negative", 10.0, db=db)
    pc = next(l for l in db if l.name == "PC 34:1")
    hit = ann.annotate_mz(ion_mz(pc.neutral_mass, "[M-H]-"))
    assert hit and hit[0].lipid.name == "PC 34:1"
