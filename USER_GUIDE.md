# SMILE MSI — User Guide

A practical walkthrough of the desktop app, from loading data to a publishable
result. No prior MSI-software experience assumed.

> **Try it with no data:** launch the app and click **Load demo dataset** — a
> synthetic brain-like section with white-matter, gray-matter, and lesion regions.
> Every panel below works on it immediately.

---

## 1. Install & launch

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[gui]'
smile_msi                 # launches the app  (or: smile_msi yourdata.imzML)
```
Or **double-click** `run_app.command` (macOS) / `run_app.bat` (Windows). First launch
builds the environment; after that it opens instantly. Everything runs locally — your
data never leaves the machine.

## 2. Load data (left "Dataset" panel)

| Source | How |
|---|---|
| **imzML + ibd** | *Open imzML…* (the `.ibd` must sit beside the `.imzML`). |
| **Imaging table** | *Open table (CSV/TSV)…* — *long* (`x,y,mz,intensity`) or *wide* (`x,y` then one column per m/z). |
| **Demo** | *Load demo dataset*. |
| **Vendor binary** (Bruker `.d`, Thermo/Waters `.raw`) | Convert to imzML first (vendor exporter, ProteoWizard `msconvert`, or imzMLConverter), then open the imzML. |

- **Quick-load every Nth pixel** subsamples a huge file for a fast preview; reload at 1 for full resolution.
- **Export to imzML…** converts any loaded dataset (e.g. a CSV) to the open imzML standard.
- The progress bar + **Cancel** apply to long operations (loading, peak picking).

## 3. Set global options (left "Options" panel)

- **Mode** — negative or positive (matches how the data was acquired; not inferred).
- **Extraction tolerance (ppm)** — window for ion-image extraction, the feature matrix, and spatial stats.
- **Identification tolerance (ppm)** — the (tighter) mass-accuracy window for lipid IDs, the
  feature list, and annotation FDR. Default **5 ppm** — correct for a well-calibrated dataset;
  see *Calibration & annotation settings* below before loosening it.
- **Normalization** — TIC / RMS / median (per-pixel), or none. Removes total-signal artifacts.
- **Reduce** — how multiple in-window peaks per pixel combine (sum / max / mean).
- **Colormap** and **Hotspot clip %** — display only; hotspot clip drops the top saturating pixels.
- **Preprocessing** — optional SNIP baseline + Savitzky-Golay smoothing + lock-mass recalibration;
  click **Apply**, then re-pick peaks.

### Calibration & annotation settings

Wrong lipid labels are usually a **calibration** problem, not a matching one. If your masses run
systematically off (e.g. a confident sulfatide reads as a formate adduct of an odd-chain ether
lipid), the true `[M−H]⁻` has been pushed outside the window and a contrived isobar wins.

1. **Measure the offset.** *Data → Calibration check…* compares the mean spectrum to known
   reference ions (oleic/DHA/nervonic acids, PI 38:4, brain PS, myelin sulfatides) and reports the
   median ppm offset and whether it's a constant shift (flat slope) or a mass-dependent stretch.
2. **Correct it.** If it's a constant offset, click **Apply lock-mass recalibration** — it fills
   the Preprocessing lock-mass field with the matched anchors, applies it, and re-verifies (target
   **< 2 ppm**). A mass-dependent stretch needs multi-point recalibration instead. **Re-pick peaks**
   after applying.
3. **Then tighten / trust the tolerance.** With the data recalibrated, the 5 ppm identification
   tolerance and the built-in chemistry filters (base `[M−H]⁻`/`[M+H]⁺` preferred over adducts;
   Cl/formate/acetate only for choline & ceramide classes; even-chain, ≤6-double-bond database)
   keep annotations honest. Order matters: **recalibrate → then tighten → then verify** — tightening
   first would drop your still-uncalibrated real peaks.

## 4. Find peaks

On the **Ion image** tab, set **Min S/N** and **Min relative intensity**, then **Find peaks**.
This builds the feature set everything else uses. Picked peaks appear in the table (with lipid
IDs) and as markers on the mean spectrum. Select a row — or drag the green line — to view that
ion's image.

## 5. The tabs

> Views are organized into **9 top-level tabs**: **Ion image · Segmentation · Compare regions ·
> Co-localization · Cohort · Explore · Advanced · More · Report**. Several views described below
> nest one click deeper — *Region comparison* and *Discriminating & ROI stats* under **Compare
> regions**; *Components* and *Feature space* under **Explore**; *Classify / Markers (SSC) /
> Per-ion (DGMM)* under **Advanced**; *Montage*, *Feature lists* and *Quality (QC)* under **More**.

- **Ion image** — the active m/z mapped across the tissue. Turn on **Draw ROI** and outline an area
  (rectangle, circle, polygon, or freehand brush) — or start the mask from a **Threshold (signal)**
  or an **Existing region** (see *Regions by construction* below) — refine it with the **Refine**
  pill (inner rim / outer collar / band of N px, shown in µm too, or *Everything else*), then
  **Save as region** or click **→ Feature list** right on the toolbar
  to turn that ROI into an annotated feature list in one step (it's saved as a region and its peaks are
  picked + identified; switch sets via *Features ▸ Feature set*). With the ROI tool off, click any
  pixel to overlay its spectrum; the ROI's mean spectrum overlays automatically. Manage which
  overlaid traces are shown via the spectrum toolbar's **Traces…** button, and build a total-class or
  A/B ratio map with the **Composite…** button on the ion-image toolbar. To show a co-registered
  optical/histology image behind the tissue, load and align it via *File → Image setup…*, then toggle
  it with *View → Show optical image*. **Save image…** writes a publication PNG (colorbar, optional
  scale bar from the imzML pixel size).
- **Feature list** — *Build feature list* identifies every peak: best lipid + alternatives, adduct,
  ppm, **confidence**, **isotopologue?** flag, **spatial (Moran's I)** structure, and an **annotation
  FDR** (computed on monoisotopic peaks). **Confirm with MS/MS…** loads an MGF of fragment spectra and
  adds a confirmed/unsupported verdict per feature. Single-click a row to see its ion image;
  double-click to open PubMed. **Export CSV** / **Export methods report** (provenance).
- **Montage** — a labeled grid (4–20) of annotated ion images, ranked by intensity or spatial
  structure (Moran's I); one-click PNG export.
- **Segmentation** — partition the tissue into regions. **Auto** picks the cluster count;
  **Spatially-aware** yields coherent regions. The cluster table lists each region's top lipids and
  fills the Region A/B selectors. The **Cluster tree (HCA)** beside the map shows the hierarchy the
  Detail slider cuts — drag its dashed line (or the slider) to set how many segments to keep; a big
  vertical gap in the tree is a natural place to cut. Branch colours match the map.
- **Components** — PCA / NMF component images, or a UMAP / t-SNE pixel embedding coloured by molecular
  similarity. The table shows each component's driving ions.
- **Colour overlay** — composite several ions in different colours. This is the **Color overlay**
  checkbox in the right-hand *Features* panel (tick it, then choose which features show and their
  colours) — not a separate tab.
- **Discriminating & ROI stats** — pick Region A/B (drawn ROIs or clusters), then *Run ROI statistics*
  (exact rank-based AUC + Mann-Whitney + FDR), *Discriminating features* (per region), or *Multi-group*
  (Kruskal-Wallis). Export CSV or a formatted Excel report (with a provenance sheet).
- **Discriminating & ROI stats** — also includes a **volcano plot** (effect size vs FDR); click a
  point to jump to that ion's image.
- **Region comparison** — two regions' mean spectra and top lipids **side by side** (optional overlay).
- **Co-localization** — rank all ions by spatial similarity to the active m/z, and **Detect modules**
  to cluster ions into co-localized groups (reordered correlation heatmap + module table).

### Regions by construction (threshold → collar → everything else)

Compartments that follow the tissue's own signal instead of a hand-drawn outline. On the
Draw-ROI bar the shape picker doubles as a **Source** list:

- **Threshold (signal)** — the mask is every pixel where a signal reaches a cut. Open
  **Threshold ▾** to pick the signal (the active feature, a lipid class composite, or *Sum of
  features…* such as the two sulfatide species) and the cut (a percentile of the signal pixels,
  default 60; tick *absolute intensity* to cut on raw intensity). *Fill holes* makes a fascicle
  interior solid; *drop islands* removes specks. The footprint paints live on the image with a
  pixel count.
- **Existing region** — the mask starts from a saved region, so a cluster-built or drawn
  endoneurium can grow a collar.
- **Refine** — *Filled* keeps the mask; *Inner rim* / *Outer collar* / *Band* keep a ring of N px
  around its boundary (the µm readout uses the slide's pixel size); *Everything else* inverts it
  to every other acquired pixel.
- **Save as region** — names it from the build (*ST 42:2;O3 ≥ p60*, *endo · outer collar 6 px*,
  *not endo*); a refined existing region nests under its source.
- **Compartments…** — the whole nerve recipe in one dialog: signal + cut → **endoneurium**,
  collar width → **perineurium**, everything else → **epineurium**, previewed as one overlay.
  *Create regions* adds the three (names editable) tagged as their own groups, so Multi-group
  features, Discriminating features and Region comparison list them straight away.

The same operations sit on the Regions panel: right-click a region ▸ **Derive** ▸ *Inner rim… /
Outer collar… / Band… / Everything else / Fill holes*. Every step is undoable (⌘Z) and recorded
in the provenance as a `region_derive` step. From a script (section 11) the same masks come from
`threshold_mask`, `ring`, `invert` and `composite`, and `add_region` pushes them back into the app.

## 6. Export & reproducibility

- **Copy a table (⌘C / Ctrl+C)** — click any table, tree or list in the app and press ⌘C: the
  **whole visible grid** lands on the clipboard as CSV, header row included, in the order and with
  the sort and filters you are looking at. Select two or more rows first and only those rows are
  copied. Right-click any table for the same **Copy table (CSV)** and **Copy selected rows**,
  plus **Copy for Excel (tab-separated)** — Excel pastes comma-separated text into a single
  column, tabs land it in real cells — and **Export table…** for a file.
- **File → Export… (⌘E)** — the **Export hub**: one dialog for every artifact, with format and
  design optionality.
  - *What to export:* the active **ion image**, a **colour overlay**, an **ion-image gallery**
    (one file per visible feature → a folder), a **spectrum figure**, **spectra data**, the
    **feature table**, **ROI statistics**, the **segmentation map**, or a **full data book (PDF)**.
  - *Ion images* are rendered **full-bleed** with a tasteful, semi-transparent "glass" card
    composited **into a corner of the image** (not the side margin): a mini intensity **spectrum**
    with the active m/z marked, a compact **colour scale**, the lipid ID, and a scale bar — so a
    single image is self-describing.
  - *Design options:* card theme (dark glass / light), colormap, corner placement, which overlay
    elements to show, scale-bar length, resolution (DPI), and figure width.
  - *Formats:* PNG · TIFF · JPEG · PDF · SVG (images) and CSV · TSV · XLSX · JSON · Markdown (tables/spectra).
- **Full data book (PDF)** — a single multi-page report: cover, dataset summary,
  **methods + provenance + references**, an ion-image gallery (each with its corner overlay),
  spectra, segmentation, and discriminating-feature statistics. Tick exactly the sections you want.
- **File → Save session… / Open session…** — persist and restore your work (dataset source,
  settings, picked peaks, segmentation, ROIs, active m/z).
- **Excel report** (ROI statistics) — formatted, with an Overview charts sheet, a Key sheet, and a
  **Provenance & methods** sheet (input checksum, every step + parameters, software versions, and an
  auto-drafted methods paragraph).
- **Export CSV (with analyses)…** (Feature list ▾ menu, or right-click the feature table) — **one
  CSV that carries every analysis you have run.** Your feature list is the rows; each completed
  analysis adds its per-ion results as extra columns, named after the analysis so nothing
  collides: `roi_comparison.AUC`, `roi_comparison.q_value`,
  `discriminating_features.endo.log2_fc`, `pca.component0.loading`, `colocalize.score`, … Tick
  the analyses you want — all of them by default, or none for a list carrying just its lipid
  annotation — and it lands ready to analyse in R, Excel or pandas with nothing left to join.
  - An analysis with one row per *region* (or component, or region-pair) becomes one column per
    region, so a feature stays exactly one row.
  - Run the same analysis twice and the columns say which is which —
    `roi_comparison_endo_vs_peri.AUC` next to `roi_comparison_peri_vs_epi.AUC`.
  - Analyses that aren't per-ion — lipid-class comparison, cross-validation, per-ion segmentation
    — are listed greyed with the reason, so an absent column block never looks like a failed run.
  - An ion an analysis never scored gets a blank cell; no feature is ever dropped.
  - m/z are matched within a tolerance (5 ppm by default), so an analysis you ran *before*
    recalibrating still lines up with the list.
- **Export methods report…** (Feature list ▾ menu) — `methods.md` + `provenance.json` for a manuscript supplement.
- Montage / Components views keep a direct **Export…** (PNG/TIFF/JPEG/SVG, rendered at 2× for crispness).

### Analysis profiles (standardized, versioned settings)

**Data → Analysis profiles…** lets you lock the processing settings that should stay constant
across samples and slides into one named, citable object — polarity & mass tolerances,
normalization & the TIC amplification cap, peak-picking thresholds, the pre-processing chain,
segmentation algorithm/metric/defaults, the spatial-finder gates, and the **random seed**.

- The **active profile pre-fills the defaults for every new sample**. It is *soft*: you can still
  change any value for one sample — but the change is recorded in that sample's session/provenance
  ("deviated from *Nerve-Lipid v2*: ppm 8 vs 5"), so nothing is silently different.
- **Named, versioned, shareable.** "Save as new version" keeps the old version on disk (a result
  that recorded *v2* can still be reproduced after the lab moves to *v3*). **Export…/Import…** share a
  profile as a `.json` file. The read-only **SMILE default** is the consensus baseline to start from.
- The active profile is remembered between sessions, and its name + version + content-hash are
  stamped into every saved session — so a result always traces back to the exact method that made it.
- Setting a **random seed** here makes clustering/segmentation and embeddings reproduce identically
  on the same machine.

Profiles live in `~/.smile-msi/profiles/` (see *Where presets live* below).

## 7. Scientific caveats

- Lipid IDs are **sum-composition level** (e.g. `PE 38:4`); they don't resolve true isobars or sn-position.
  The annotation FDR is high because the in-silico database enumerates every composition — **confirm key
  hits with MS/MS.**
- **ROI AUC is exact** here (rank-based, per-pixel), unlike mean/SD peak-table approximations.
- **Replication unit (pseudoreplication).** In Region comparison, the p/q values depend on what counts
  as one independent observation. Pixels in a region are spatially correlated — they are **not**
  independent replicates, so per-pixel p-values are wildly overconfident. Use the **Replicate unit**
  control (⚙ More): *Auto* runs an across-ROI test when both sides have ≥2 ticked regions (each ROI =
  one replicate), otherwise it gives a clearly-labelled **descriptive** read; *Pixel* forces the
  descriptive read. For inference *across subjects*, make each tissue section a sample in the **Cohort
  tab** (Group A vs B, sample-as-replicate). A single region per side cannot support a population claim —
  treat its AUC / signed log2 fold-change as descriptive effect sizes, not significance.
- Coverage = common membrane lipids; gangliosides, sterols, and small metabolites return unidentified.

## 8. Typical workflow

1. Load data → set mode/normalization.
2. Find peaks → (optional) preprocess and re-pick.
3. Segment (Auto + Spatially-aware) → review cluster signatures.
4. Choose two regions (clusters or drawn ROIs) → ROI statistics / Region comparison.
5. Build the Feature list → review IDs, confidence, spatial structure.
6. Export the Excel report + methods/provenance.

## 9. Working with multiple files (cohorts)

A **cohort** is the level *above* a sample's regions: where a region groups pixels within
one slide, a cohort groups whole slides across files. Use it to compare samples and run
multi-sample batch statistics — without holding every dataset in memory.

- **Samples** section (top of the right-hand panel, collapsible): the roster of files in
  the current cohort, bucketed by their **group** label.
  - *Open files…* opens one or more imzML files (the first becomes active; the rest are
    added to the cohort). *Add saved…* adds samples you've already worked on.
  - **Double-click** a sample to switch to it — the app auto-saves the sample you're
    leaving and reloads the picked one, so only one dataset is ever active.
  - **Right-click** → *Assign to group* (e.g. `control` vs `synkinetic`) or *Remove*.
  - The roster auto-saves as the "Workspace" cohort and is restored on the next launch.
- **Cohort** tab — cross-sample comparison. Pick the feature axis (a *Consensus* of peaks
  shared across samples, or the active feature set), choose **Group A vs B**, and *Run
  batch comparison*. You get a volcano + table (per-feature log2 fold-change, Mann-Whitney
  p, BH q with the sample as the replicate) and a sample × feature heatmap. Each sample's
  *saved peaks* are used, so this never reloads the raw cubes. Click a volcano point to
  view that ion on the active sample; *Export CSV…* saves the comparison.

## 10. Analyze gallery & run History

Every downstream analysis is reached from one place: the **Analyze** tab, a gallery of cards —
one per analysis (markers, per-ion segmentation, discriminating features, region A/B comparison,
co-localization, identify lipids, …), grouped by category and each with a short description. A
**This slide / Cohort** scope toggle filters the cards; a card whose prerequisites aren't met is
disabled and says why ("Needs: two regions").

- **Run one** — click a card to open its **Configure & Run** popup: pick the data scope (feature
  set, regions, groups) and edit the parameters, then **Run**. The result renders in the same
  popup, with **Apply to session** (commit peaks / regions / segmentation) and **Add to report**.
- **History** (*Analyses → History*) — every run you do is recorded automatically: analysis,
  dataset, status and a one-line summary. Select a row to **Open** it (re-shows the stored result
  exactly as computed, no recompute — a banner flags it if the data has since changed), **Re-run**
  it against the current data, **Duplicate** it into a fresh popup pre-filled with its settings,
  **Add to report**, or **Delete**. Run payloads live beside the session in a `<session>.runs/`
  folder; only a lightweight index is stored in the session itself.
- **Take them all with you** — because every run is recorded, *Feature list ▾ → Export CSV (with
  analyses)…* (§6) can fold each one's per-ion results into a single CSV alongside the lipid
  annotation. Run the analyses you want, then export once.

For a saveable, replayable *pipeline* across many slides, use the **Analysis Script** console
(§11) — it drives the same engine from Python and saves reusable workflow presets. *(The older
Flow designer was retired; any flows you had saved under `~/.smile-msi/flows/` convert to
scripting presets with `python scripts/migrate_flows_to_presets.py`.)*

### Where presets live (the `~/.smile-msi` folder)

Script-Console presets are plain JSON files under the app's home directory (`~/.smile-msi/`,
overridable with the `SMILE_MSI_HOME` env var). They hold *no data*, so they're portable: back
them up, version them, or copy them to another machine.

- **`workflows/`** — Script-Console presets. Top-level keys: `version`, `name`, `code`, `description`.

## 11. Analysis Script (drive the engine with Python)

For batch/replay across slides, or analyses a single popup can't express, the **Script Console**
(*Data → Analysis script…*, ⇧⌘J) lets you write a short Python script against the same engine and
run it on the loaded slide, then save it as a reusable workflow preset.

- **Write it** — bare-name functions are pre-bound: `find_peaks()`, `segment()`, `compare('Group A',
  'Group B')`, `colocalize()`, `annotate()`, `pca()`, … plus `ds` (the dataset) and `np`. Surface
  output with `log(...)`, `table(df, "title")`, `image(mz, "title")` and `record(name, value)`.
- **Run it** — ⌘/Ctrl+Return (or the **▶ Run** button). Tables and ion images appear as tabs; the
  **Log** tab shows prints, recorded values and any error (with your line numbers).
- **Apply to app** — push the script's features or segmentation back into the app's views, or
  add the regions it staged with `add_region(name, mask)` to the slide (masks built with
  `threshold_mask(composite([...]), 60)`, `ring(mask, width_px=6, mode="outer")`, `invert(mask)`
  — see the *Compartments by construction* example).
- **AI guide** — the **AI guide** button shows (and copies) a self-describing reference of every
  function plus this slide's context. Hand it to an AI assistant (e.g. a chat assistant) and ask it to write
  a workflow for you; paste the result back in. The same text lives in
  [`SCRIPTING.md`](SCRIPTING.md).
- **Save as workflow** — store a script under `~/.smile-msi/workflows/` (*Workflows → Save as…*) and
  reload it on any slide. **Examples ▾** drops in starter scripts.
