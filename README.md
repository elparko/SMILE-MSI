# SMILE MSI

*Spatial Mass Imaging of Lipid Environments — desktop MSI analysis. (Python package: `smile_msi`.)*

[![CI](https://github.com/elparko/SMILE-MSI/actions/workflows/ci.yml/badge.svg)](https://github.com/elparko/SMILE-MSI/actions/workflows/ci.yml)

**Open-source MALDI-MSI analysis — a desktop imaging workspace.** Load a raw
imaging dataset (imzML/ibd), explore it spatially, segment it, find the ions that
discriminate regions, and identify every feature as a lipid — all locally, with no
database file and no network.

It's a **desktop app** for interactive work, backed by a **Python library** you can script.

📖 **New here? See the [User Guide](USER_GUIDE.md)** (also in-app under **Help → Quick start**).

---

## What it does

**Imaging & spectra**
- Loads **imzML/ibd** (the open MSI standard) with a *lazy, out-of-core* backend — large
  `.ibd` files are read on demand, not pulled into RAM.
- **Ion images** of any *m/z* (± ppm) with TIC / RMS / median normalization, hotspot
  (quantile) contrast, and colormaps.
- **Mean spectrum**, **single-pixel** spectra, and **region** spectra with overlay comparison.
- **RGB overlay** — assign three ions to red/green/blue channels.

**Preprocessing**
- Baseline correction (SNIP), Savitzky-Golay / Gaussian smoothing, m/z recalibration to
  lock masses, vector / reference normalization — composable, axis-preserving.

**Find structure**
- **Peak picking** on the mean spectrum.
- **Spatial feature finding** — rank/keep ions by **Moran's I** spatial autocorrelation, so you
  focus on features that are actually spatially structured, not just intense.
- **Segmentation** (k-means; automatic *k* by silhouette; spatially-aware variant; the two combine).
- **Component analysis** — PCA & NMF component images, t-SNE / UMAP pixel embedding.
- **Co-localization** — rank ions by spatial similarity to a target; full co-localization matrix.

**Statistics**
- **Exact** rank-based ROC **AUC** + Mann–Whitney *p* + Benjamini-Hochberg **FDR** between two
  regions (drawn ROIs or segmentation clusters) — no Gaussian approximation; per-pixel values.
- **Discriminating features** per region (one-vs-rest) and **multi-group** (Kruskal–Wallis).

**Lipid identification**
- In-silico **m/z → lipid** annotation (negative/positive), sum-composition level.
- A **Feature List** with best ID, alternative candidates, adduct, ppm, **isotope-pattern**
  consistency, **adduct corroboration**, a folded **confidence** label, an **annotation-FDR**
  estimate (target-decoy), and per-lipid **research links** (PubMed / Europe PMC / Scholar).
- Formatted **Excel report** and CSV exports.

**Advanced**
- **Cross-sample cohorts** — pool many slides into a cohort, QC/diagnose acquisition **batch
  effects** (empirical-Bayes ComBat diagnostics), and run **nested / hierarchical statistics**
  (subject-aware mixed-effects, moderated-*t*) with consensus feature lists across sections.
- **Multi-omics co-mapping** — bring in **spatial transcriptomics** (10x Visium / AnnData `.h5ad`
  / Space Ranger) or another modality and correlate MSI ions against it in a shared coordinate space.
- **Image registration** — align an **optical / histology** image (or a second modality) onto the
  MSI grid by landmarks / affine transform.
- **Overlays & co-localization** — **RGB multi-ion overlays**, rank ions by spatial similarity to a
  target ion, and compute a full co-localization matrix.
- **MS/MS spectral matching** — corroborate identifications against a spectral library by entropy
  and (modified) cosine similarity.
- **On-tissue quantification (qMSI)** — absolute concentrations from calibration curves fitted on
  concentration-defined standard regions, with LOD/LOQ and out-of-range flagging.
- **Machine-learning biomarkers** — PLS-DA, nearest-shrunken-centroids, and **SHAP** feature
  importance, with cross-validated classification maps.
- **Optional** — learned **ion embeddings** (`[ion-embed]`), **single-cell** segmentation
  (`[cells]`), and **3-D** serial-section reconstruction (`[viz3d]`).


---

## Install

### Download the app (no Python needed)

Grab the latest bundle from the
[**Releases**](https://github.com/elparko/SMILE-MSI/releases/latest) page:

- **Windows** — if a `SMILE-MSI-windows.zip` is attached to the latest release, download it,
  unzip anywhere, open the `SMILE MSI` folder and run **`SMILE MSI.exe`**. (Unsigned app: if
  SmartScreen warns, *More info → Run anyway*.) If no Windows bundle is posted yet, install
  [from source](#from-source-developers) instead — it's a two-line install.
- **macOS** — download `SMILE-MSI-macos.zip`, unzip, move **`SMILE MSI.app`** to
  Applications. First launch: right-click → *Open* (unsigned).

**Staying up to date:** in the app, **Help → Check for updates…** tells you when a newer
release is out and opens the download page — replace the old folder and you're current.
Releases are built and published automatically by GitHub Actions on every version tag.

### From source (developers)

With [uv](https://docs.astral.sh/uv/) (recommended):
```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[gui]'        # core engine + desktop app
```
The base install (`uv pip install -e .`) gives the analysis engine as a Python API
(`import smile_msi` — see [Scripting](#scripting) / the `examples/` folder); the `[gui]`
extra adds the desktop app (PySide6 + pyqtgraph). `[embed]` adds UMAP. There is no separate
command-line analysis pipeline — the `smile-msi` command opens the desktop app (so it needs
the `[gui]` extra); headless use is via the Python API or the in-app Analysis Script console.

## Use it

### Desktop app
```bash
smile-msi              # launches the app  (or: smile-msi data.imzML to open a file)
smile-msi-gui          # same thing
```
Or **double-click** `run_app.command` (macOS) / `run_app.bat` (Windows) — first launch builds
the environment, then it opens instantly. Click **Load demo dataset** to try every panel with a
synthetic brain-like section (no data needed). For very large files, set **Quick-load every Nth
pixel** to preview fast, then reload at full resolution.

Workflow: *Load → Find peaks → (preprocess) → ion images / segmentation / components →
draw ROIs or pick clusters → ROI statistics → Feature list → export report + provenance.*

Power users (and AI assistants) can drive the engine directly: **Data → Analysis script…**
(⇧⌘J) opens a Python console with pre-bound functions (`find_peaks`, `segment`, `compare`,
`annotate`, …), runs against the loaded slide, and saves scripts as reusable workflows. Its
**AI guide** button (also [`SCRIPTING.md`](SCRIPTING.md)) emits a hand-to-AI-assistant reference so
an assistant can write a correct workflow for you. [`SCRIPTING.md`](SCRIPTING.md) is the full
scripting reference; it is generated from that in-app AI guide (`scripting.capabilities_doc()`).

### Build a standalone app
No-Python-required bundles (a `SMILE MSI.app` on macOS, a `SMILE MSI` folder on Windows):
```bash
pip install -e '.[gui,build]'
pyinstaller --noconfirm smile_msi.spec      # -> dist/SMILE MSI(.app)
```
On Windows you can instead double-click `scripts\build_windows.bat`. The **build-app**
GitHub Actions workflow builds the macOS + Windows bundles, smoke-tests each frozen bundle,
and **publishes them to a GitHub Release** whenever you push a version tag:
```bash
git tag v0.3.1 && git push --tags        # → Release with downloadable zips
```

### Python library
```python
from smile_msi.msi import MSIDataset
from smile_msi import spatial, multivariate, annotate

ds = MSIDataset.from_imzml("data.imzML")      # lazy
ds.prime()                                     # mean spectrum + norm stats (1 streaming pass)
peaks = [p["mz"] for p in ds.pick_peaks(snr=3)]

seg  = spatial.auto_segment(ds, peaks)                       # regions
disc = spatial.discriminating_features(ds, seg.labels, peaks)
feat = annotate.build_feature_list(ds, peaks, mode="negative")   # identified feature list
img  = ds.ion_image(peaks[0])                               # numpy (height, width)
```

## Scientific notes & caveats
- **Lipid IDs are sum-composition level** (e.g. `PE 38:4`) and do not resolve true isobars or
  sn-position. The annotation FDR is high precisely because the in-silico database enumerates
  every composition — **confirm key hits with MS/MS.**
- **ROI AUC is exact** (rank-based, from per-pixel values), unlike the peak-table tool which can
  only approximate AUC from a mean and SD.
- **Normalization** matters — use TIC/RMS/median to remove pixel-to-pixel total-signal artifacts.
- Coverage = glycerophospholipids, ether/plasmalogens, lyso forms, sphingolipids (incl.
  sulfatides), **gangliosides** (GM/GD/GT series), **sterols & cholesteryl esters**, and fatty
  acids. Small metabolites and oxylipins still return unidentified.

## Tests
```bash
pytest -q
```

## License
Apache-2.0 — see [LICENSE](LICENSE). Third-party dependency licenses are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (note: the optional desktop GUI links
PySide6 / Qt for Python under the LGPL-3.0).
