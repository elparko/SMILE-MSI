# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the SMILE MSI desktop app.

One spec, every OS — but PyInstaller does NOT cross-compile: run it on the OS you want a
bundle for (GitHub Actions builds the Windows + macOS bundles for us). Produces a
one-folder bundle in ``dist/SMILE MSI/`` (fast start, easy to zip and ship), plus a
``SMILE MSI.app`` on macOS.

    pip install -e ".[gui,build,embed,shap]"  # embed = real UMAP; shap = SHAP biomarkers frozen in
    pyinstaller --noconfirm smile_msi.spec
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# These packages import submodules dynamically (by string), so PyInstaller's static
# analysis misses them — pull them in wholesale.
hiddenimports = (
    collect_submodules("sklearn")
    + collect_submodules("scipy")
    + collect_submodules("pyqtgraph")
    + collect_submodules("qtawesome")          # icon library (loads fonts by string)
    # statsmodels resolves model/formula machinery dynamically, so the static analysis misses
    # most of it; patsy comes along as its formula backend.
    + collect_submodules("statsmodels")
    + collect_submodules("patsy")
    + ["pyimzml", "pyimzml.ImzMLParser", "pyimzml.ImzMLWriter", "PIL", "qtpy"]
)
binaries = []

# _ctypes needs libffi at runtime. On some Windows Pythons (Microsoft Store builds, and
# occasionally brand-new point releases) PyInstaller fails to bundle libffi-*.dll, so the
# frozen app dies with "ImportError: DLL load failed while importing _ctypes: The specified
# module could not be found" the instant anything imports ctypes — matplotlib's runtime hook
# does, at launch. Bundle it explicitly from the interpreter's DLLs dir (a no-op duplicate on
# a python.org build where PyInstaller already grabbed it; the fix for the ones where it didn't).
if sys.platform == "win32":
    import glob
    for _base in {sys.base_prefix, sys.prefix}:
        for _ffi in glob.glob(str(Path(_base) / "DLLs" / "libffi-*.dll")):
            binaries.append((_ffi, "."))

# Non-code data the app reads at runtime: the package's bundled logo (loaded relative to
# the package, so collect_data_files preserves the path), pyqtgraph's own assets, and
# qtawesome's icon fonts + charmap JSONs (without these the frozen app silently degrades
# to the unicode-glyph fallback instead of the vector icons).
datas = (collect_data_files("smile_msi") + collect_data_files("pyqtgraph")
         + collect_data_files("qtawesome"))
for doc in ("USER_GUIDE.md", "README.md"):           # opened from Help → these live at repo root
    if Path(doc).exists():
        datas.append((doc, "."))

# License texts we are OBLIGED to ship with a distributed binary: LGPL-3.0 for Qt/PySide6,
# CC BY 4.0 (+ attribution) for the Font Awesome icons, plus our own LICENSE/NOTICE and the
# third-party summary. See THIRD_PARTY_LICENSES.md. Placed at the bundle root and in
# licenses/ next to the executable so the obligations are actually met, not just claimed.
for _lic in ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md"):
    if Path(_lic).exists():
        datas.append((_lic, "."))
if Path("licenses").is_dir():
    datas.append(("licenses", "licenses"))

# Zarr (v2) cube store (cubestore.py): zarr imports its codec/storage submodules by string
# and numcodecs ships compiled Blosc/zstd codecs — AND cubestore.py imports both LAZILY (inside
# function bodies), so PyInstaller's static graph never sees them. Without this the frozen app
# raises "No module named zarr" the first time it builds/streams a cube. These are core deps
# (not optional), so no try/except: a build env missing them should fail loudly.
for _pkg in ("zarr", "numcodecs"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# Optional UMAP stack (the [embed] extra: umap-learn → numba → llvmlite → pynndescent).
# These do heavy dynamic importing and ship compiled shared libs (llvmlite especially), so
# collect_all grabs their modules + data + binaries. Skipped cleanly when the extra isn't
# installed — the app then falls back to t-SNE.
for _pkg in ("umap", "numba", "llvmlite", "pynndescent"):
    try:
        _d, _b, _h = collect_all(_pkg)
    except Exception:  # noqa: BLE001 — extra not present in this build
        continue
    datas += _d
    binaries += _b
    hiddenimports += _h

# SHAP biomarkers (the [shap] extra): shap imports numba-compiled kernels and many submodules
# dynamically, so the static graph misses them. Freeze it in so the packaged app's "SHAP
# biomarkers" analysis works offline — explain.py deliberately REFUSES to pip-install shap in
# a frozen build (a frozen exe can't self-install), so it must be bundled here or the feature
# is dead in the shipped app.
try:
    _d, _b, _h = collect_all("shap")
    datas += _d
    binaries += _b
    hiddenimports += _h
except Exception:  # noqa: BLE001 — [shap] extra not present in this build
    pass

# Trim weight: never used, and each drags in large or platform-specific deps.
excludes = ["PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "notebook",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"]

ICON = None
if sys.platform == "win32" and Path("scripts/SMILE MSI.ico").exists():
    ICON = "scripts/SMILE MSI.ico"
elif sys.platform == "darwin" and Path("scripts/SMILE MSI.icns").exists():
    ICON = "scripts/SMILE MSI.icns"

a = Analysis(
    ["scripts/launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SMILE MSI",
    console=False,             # windowed GUI app — no console window on Windows
    disable_windowed_traceback=False,
    icon=ICON,
)
coll = COLLECT(exe, a.binaries, a.datas, name="SMILE MSI")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="SMILE MSI.app",
        icon=ICON,
        bundle_identifier="org.smilemsi.app",
        info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "11.0"},
    )
