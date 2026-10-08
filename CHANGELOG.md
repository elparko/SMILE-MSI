# Changelog

All notable changes to SMILE MSI. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- **The same ion no longer changes intensity mid-session.** `run_analysis(...)` in scripts (and
  the MCP / chat tool of the same name) let a registry step extract at its own default window
  (50 ppm) instead of the session's tolerance, so after e.g. segmentation an ion's values were
  read from a wider window — a 0.81 → 1.39 jump for the same ion on the demo slide. It now uses
  the session's tolerance and normalization, as the named wrappers always did.

### Added
- **Plain-language analysis chat — `smile-msi chat`.** A browser chat (local server, standard
  library only) where you describe an analysis and a model runs it with the MCP server's tools:
  tool calls stream in as they run, figures appear inline (and the model sees them), and every
  message, tool call, argument, result, approval, dataset fingerprint and token count goes to an
  append-only session log, exportable as Markdown ending in a replayable list of tool calls.
  Bring your own model: Claude through the official SDK (`[agent]` extra) or any
  OpenAI-compatible local server (Ollama, LM Studio, llama.cpp, vLLM). The model can create,
  edit and delete its own versioned analysis tools — each such change waits for your approval
  and shows the code — and save analyses as replayable flows.
- **Analysis setup before the chat starts.** A setup card (then a sidebar) fixes the question,
  replication unit, q and AUC thresholds and the method settings from an Analysis Profile before
  any analysis runs. It is logged, applied to the slide, filled into tool calls, and any call that
  differs is flagged as a deviation; it can be saved as a profile the desktop app can load.
- **Chat hardening.** Adversarial testing (model-driven and code review) found ways to wedge a
  conversation, read files outside the data folder, and get misleading "ok" results; all fixed,
  with a regression test each (see the commit for the list).
- **Segments coloured by the tree.** Segmentations cut from the granularity tree are coloured
  with Tree Colors (Tennekes & de Jonge 2014) in OKLCh instead of a 12-colour cycling palette:
  sibling segments share a hue family, a segment keeps its hue family as Detail gets finer, and
  colours no longer repeat past 12 segments. The segment map, live Detail preview, dendrogram,
  figure export and joint segmentation all use them; manual splits get shades of the parent;
  the colours are saved with the session.
- **Copying a table copies the table.** Qt gives a grid no copy of its own, so ⌘C over a result
  table left the clipboard holding whatever was there before, and the one right-click *Copy* that
  existed sent the selected cells only — click a row, press copy, get one number. ⌘C / Ctrl+C now
  works over **any** table, tree or list in the app and copies the whole visible grid as **CSV
  with a header row**, in the order you are looking at (a sorted table copies sorted; filtered-out
  and hidden rows and columns stay out). Select two or more rows and only those are copied.
  Checkbox-only columns — the segmentation ✓, the standards *Use* tick — copy as `yes`/`no`
  instead of an empty column. Excel pastes comma-separated text into one column, so there is also
  a **Copy for Excel (tab-separated)** that lands in real cells. Every table gained a right-click
  **Copy table (CSV)** / **Copy selected rows** / **Copy for Excel**,
  including the ones that previously had no way out at all: the calibration check, the timeline,
  the report-book table preview, the cohort roster, the lipid tree, the segment table, the
  feature-lists manager, the shared-cluster table, the co-localization pairs, the cache list and
  the nested lipid-class test. The status bar says how many rows and columns went.
- **Calibrate each region separately.** Samples in different embedding media on one slide
  drift by different amounts, and a single slide-wide lock mass fits the average of those
  populations and leaves every one of them mis-assigned. Data ▸ Calibration check gained a
  **Measure over** picker: whole slide, one region, or every region separately. Measuring per
  region reports each region's offset and the **spread between them** — the number that says
  whether one slide-wide correction can work at all — and Apply gives each region its own
  lock mass, with its own tolerance window sized to its own drift. The correction stays
  axis-preserving, so all pixels keep one shared m/z axis and regions remain comparable bin
  for bin; pixels in no region fall back to the slide-wide reference m/z. The banner reports
  the **worst** region, never the best. Saved in the session and re-applied on reopen, with
  each region's mask fingerprint in the preprocessing hash so redrawing a region invalidates
  the fast cube instead of silently reusing the old pixel assignment. Scripting:
  `api.measure_calibration(region=…)` and `api.calibrate_regions()`
  (`preprocess.recalibrate_regions`, `intake.measure_calibration_offset(mask=…)`).

### Fixed
- **A narrow extraction window no longer misses the peak it is centred on.** On a profile
  axis sampled more coarsely than the tolerance window (the demo's 10 mDa grid against 10 ppm,
  5.6 mDa at m/z 282), a window between two samples extracted nothing, so FA 18:1's real M+1
  read as absent (M+1/M 0.0, isotope co-localisation 0.00). Every window on a shared profile
  axis now holds at least the nearest sample; a gap in the axis is never bridged. FA 18:1 now
  measures M+1/M 0.19 against a theoretical 0.20.
- **"High" confidence needs a passing isotope check.** Mass accuracy, Moran's I and sibling
  adducts could add up to High with no isotope envelope at all; such an ID is now Medium,
  and says why. An unidentified isotopologue names its parent ("M+1 isotopologue of
  890.6388"), and scripted/assistant `annotate()` fills the intensity and S/N columns from the
  picked peaks.
- **Pixel-level statistics say so.** Multi-group, A-vs-B and discriminating-feature summaries
  note when their p/q values describe pixels rather than replicates (pseudoreplication), and
  the MCP `run_analysis` result carries the step's warning.
- **Discriminating features filter first, then cap.** The per-group `top_n` cut ran before the
  enriched/q filters, so a group's depleted ions used up its slots and real markers were
  dropped without a word, and a cut through a run of tied AUCs kept one of them arbitrarily.
  Ties at the cut are now kept, and the summary reports how many passed but were over the cap.
- **Scripted and assistant runs use the session's seed and tolerance.** Segmentation, PCA,
  NMF, embedding, DGMM, cross-validation and SHAP took a fixed seed instead of the active
  profile's, and `run(step_id, …)` extracted at the registry's default tolerance instead of the
  session's.
- **Segmentation reports what it clustered.** `segment()` takes `mask=`; the MCP result echoes
  the mask and, for a whole-slide run, names the clusters that look like off-tissue background.
  `threshold_mask` keeps filling holes by default but logs when filling closed a ring, and the
  scripting guide documents it, the percentile semantics, `print()` → `stdout` and
  `ds.ratio_image`. `mean_spectrum` reports the same centroid m/z as `find_peaks`.
- **Ticking a feature list in the Export Studio no longer hangs the app.** Ticking a list's parent
  row cascades `itemChanged` over every ion under it, and both listeners — the All/None/Invert
  bar's live count and the dialog's `n sections × n ions` recount — walked the whole tree on every
  one of those signals. That is O(N²): measured 0.3 s for a 200-ion list, 7.8 s for 1000 and 70 s
  for 3000, which is why removing the 500-peak cap turned it into an apparent freeze. The recount
  is now coalesced (`common.coalesce`) — it runs once at the start of a cascade and once when the
  event loop next idles, never once per ion. Ticking a 3000-ion list went from ~70 s to **0.18 s**,
  with the same final count. Every All/None/Invert bar in the app (Export hub, co-localization,
  feature export, cohort add-samples/regions, cohort segmentation, find-peaks regions) shares the
  fix.
- **Exporting a table from a dialog no longer crashes after writing the file.** The result-table
  `⤓ CSV` button and the Analysis-script console passed their own widget as the export parent, and
  the status-bar report at the end of the write raised `AttributeError: statusBar` on a completed
  export. The export now finds the nearest window that owns a status bar, which also restores the
  `# …` provenance header block on exports launched from a dialog.
- **Feature lists no longer stop at a hidden cap of 500.** `pick_peaks` truncated to the 500
  most intense peaks with no control anywhere in the app and no record of what was cut, so
  three regions that detect 511, 670 and 636 peaks all reported 500 — which reads as the
  media agreeing when it is only the ceiling. Max peaks is now a visible setting defaulting
  to **no limit**, the spatial finder's candidate cap (2000) is visible too, and both the
  status bar and the audit trail report the pre-cap detection count and say when a cap bound.
  `pick_peaks` returns a `PeakList` carrying `.n_detected`; `SpatialFeatures` /
  `CoherentFeatures` carry `.n_detected` and `.candidates_capped`.
- The MCP `find_peaks` tool raised `TypeError: 'SpatialFeatures' object is not iterable` with
  `spatial=True`, and took no region argument. It now accepts `region=`, returns the funnel
  counts and `n_detected`, and defaults to no cap.
- **An AI assistant can drive the engine** — `python -m smile_msi.mcpserver` is a Model
  Context Protocol (MCP) server exposing 15 tools over the *same* registry steps the app
  runs: list the slides you have analysed, read the regions, groups, feature lists and past
  results on one, open a slide, pick peaks, run any registered analysis, render an ion image
  or a mean spectrum, identify lipids, run a full analysis script, and export the wide
  feature table. It reads the app's saved sessions and never writes to them — tables and
  figures go to `~/.smile-msi/mcp-results` and the tool returns the path. Tables come back as
  a short preview plus that path, so a 700-ion result costs one glance rather than a context
  window. A project `.mcp.json` is committed; the SDK is the optional `[mcp]` extra and stays
  out of the frozen app (`mcpserver.py`, `tests/test_mcpserver.py`).
- **Reopen a saved session without the GUI** (`headless.py`) — the app's live state, rebuilt
  from the session file: `open_slide("nerve-01.imzML")` returns the dataset with its
  preprocessing re-applied, the fast cube restored from the sidecar, every named region
  resolved to pixels (drawn ROIs, cluster-backed regions, aggregate parents, and the
  region↔slide tie honoured) and the group tags turned into the masks a comparison runs on —
  bound to the same `ScriptAPI` the in-app Script Console builds. `managed_sessions()` /
  `session_summary()` answer "what have I worked on and what did it find" from JSON alone,
  with no dataset opened. Extracted from the GUI load path, which keeps its own copy for now
  (`tests/test_headless.py`).
- **Regions by construction** (plan 25): the Draw-ROI bar is a small builder. The shape picker
  gained two sources — **Threshold (signal)** (the active feature, a lipid class, or a sum of
  features cut at a percentile of its signal pixels, holes filled, islands dropped, painted live)
  and **Existing region** — and the hidden Border popup became a visible **Refine** pill (inner
  rim / outer collar / band with the width in px *and* µm, plus **Everything else**) with a
  **Save as region** button that pre-fills the name from the build. **Compartments…** runs the
  nerve recipe in one dialog (threshold → endoneurium, outer collar → perineurium, everything
  else → epineurium) with an RGB preview and creates the three as grouped regions. The Regions
  panel's context menu gained **Derive ▸** (rim / collar / band / everything else / fill holes),
  nested under the source. Engine: `spatial.threshold_mask` / `fill_holes` / `drop_small` /
  `invert_mask`; script console: `composite`, `threshold_mask`, `ring`, `invert`, `add_region`
  and *Apply to app ▸ Add regions to the slide*; a `region_derive` provenance step records the
  recipe (`gui/ion.py`, `gui/compartments.py`, `gui/segment.py`, `scripting.py`, `spatial.py`).
- **Calibration is now a visible, saved state** (`gui/main.py` calibration dialog,
  `featurepanel.py` banner, `session.py`). *Apply lock-mass recalibration* used to fill the
  Preprocessing field, re-prime and re-measure with nothing that said "applied". The dialog
  now shows a state line (*Applying…* → *✓ Calibrated — ×factor from N anchors; residual
  +0.3 ppm, was −3.3 ppm, verified 14:24*), the Apply button turns into a disabled *Applied ✓*,
  the dataset banner in the right dock reads *✓ calibrated (lock-mass, residual …)* or
  *not calibrated*, and the status bar confirms the verified residual. The verify pass now
  runs when the re-prime finishes instead of being queued beside it. Clearing the reference
  m/z and applying drops the state.
- **The preprocessing pipeline is saved with the session and re-applied on reopen**
  (`build_session(preprocessing=…, calibration=…)`, `_prepare_imzml`). Before, a reopened
  slide silently ran on raw spectra while the panel still showed the lock-mass values. The
  pipeline is applied in the load worker before any pass, the panel widgets are restored
  from it, and the cube sidecar is keyed by slide fingerprint **plus** pipeline hash so a
  raw-spectra cube is never served to a recalibrated session (it rebuilds instead).
- **Data ▸ Manage caches…** (`gui/cachedialog.py`, `session.cache_inventory`) — lists every
  session, cube, legacy npz cube, run store, thumbnail folder and interrupted-build leftover in
  the managed store with its size and status, marks what is safe to delete (orphaned companions,
  older-fingerprint duplicates of the same slide, npz cubes already superseded by a Zarr cube,
  temporary files) and deletes the selection. *Change folder…* moves the whole store to another
  disk through a redirect file (`library.set_home_redirect`; `$SMILE_MSI_HOME` still wins when
  set), for machines whose system disk is nearly full.
- **Every whole-slide read is now in the perf log** (`MSIDataset.on_pass`, `PASS` lines in
  `~/smile_msi_perf.log`, shown in Help ▸ Show timeline…). Prime, the m/z bounds pass, skyline,
  ROI spectra without a cube, feature extraction, streamed ion images and both cube-build passes
  report their wall time and spectra count, flagged `GUI THREAD` when they ran on the event
  thread — the freezes that used to leave no trace.
- **Moved slides are found without asking** (`cohort.find_moved_source`) when a session's imzML
  is no longer at its recorded path: the unique same-named file (with its `.ibd`) in any folder
  this machine already reads slides from — roster samples, managed sessions, past manual
  relocations — is used, and the session JSON and roster entries are repointed. The manual
  *Locate dataset* prompt remains the fallback.

### Changed
- **A disk-backed slide is read once on open, not twice** (`MSIDataset.build_mz_cube(prime=True)`,
  `gui/main.py` `_prepare_imzml`). Prime (mean spectrum, per-pixel TIC/RMS/median) is folded
  into cube pass 1 for slides that do not fit in RAM, and the load bar now covers the whole
  build; the background cube build after first paint is gone for them. The same fused pass
  serves *Apply preprocessing*, which used to re-prime and then rebuild the cube in two reads
  (and, until now, only rebuilt it on the next ROI click).
- **Cube bins stay at 15 ppm across the full range and no peak is dropped** (`FULL_CUBE_MAX_BINS`,
  `_build_cube`). The app's cube builds pass a 200,000-bin cap (the library default of 60,000
  widened bins to ~38 ppm over 200–2000 m/z, so a 10 ppm window really integrated ±19–38 ppm)
  and no intensity floor (the old 0.1 %-of-mean-max floor removed ~2 % of peaks). Existing cube
  sidecars keep their own axis; new builds use the finer one.
- **File ▸ Open imzML reuses the slide's saved cube and prime stats** (`gui/main.py`
  `_prepare_imzml`). It used to re-prime and rebuild the cube even when the managed session and
  its Zarr sidecar existed; only the Samples panel, launcher and File ▸ Open analysis took the
  fast path. The managed session is now resolved by fingerprint before the load on every path.
- **Out-of-core cube build bins chunks on a small thread pool** (`msi.py`
  `_build_mz_cube_streaming`, `MSIDataset.n_bin_workers`): the COO→CSR build and grouping of
  each pixel chunk run on up to 4 threads while the consumer keeps draining the reader; bucket
  writes stay in pixel order, so the spill is byte-identical. The read pattern is unchanged.
- **Out-of-core cube build no longer re-scans its spill once per column band** (`msi.py`
  `_build_mz_cube_streaming`). Pass 1 now bins pixels in chunks (one block CSR per chunk, the
  same construction as the in-RAM build) and spills entries into 32 column-range bucket files;
  pass 2 sorts one bucket at a time, falling back to sub-band scans of *that bucket* only when a
  bucket exceeds the RAM budget. Spill I/O is now a small constant multiple of the entry count
  instead of `bands × spill`: a 5 µm centroid slide (143k pixels, 595M entries, 7 GB spill) used
  to scan the whole spill ~200 times. The build's RAM budget also scales with the machine
  (a sixteenth of physical RAM, 128 MB–4 GB) instead of a fixed 128 MB. Output is byte-identical
  (existing and new bit-identity tests, shared-axis and processed-mode).
- **Processed-mode prime and skyline accumulate in batches of 64 pixels** (`msi.py` `prime`,
  `max_spectrum`) instead of allocating an axis-long (up to 200k-bin) temporary per spectrum.
  The skyline is exactly unchanged; the mean differs only by float rounding order.

### Fixed
- **Analyses ignored the "Data in this analysis" bar** (`gui/analysisdialog.py` `_state`,
  `gui/scope.py`). Multi-group features (Kruskal-Wallis / ANOVA) and Discriminating features let
  you *Choose groups…* but ran on the window default — the segmentation clusters, or the Setup
  group tags — so the test "bound to segments". The dialog now builds its inputs from the bar:
  the chosen groups replace the segmentation fallback, the chosen feature set is the one used,
  and the Region-comparison A/B pickers (built but never filled or read before) list every
  ROI and name the pair. Readiness and the stale-run check use the same state, and the run
  record now carries the chosen A/B pair.
- **`MSIDataset.mz_range` re-read every spectrum on each call for processed-mode data**
  (`msi.py`, `session.py`). Without a shared m/z axis the bounds need a pass over the whole
  file, and nothing cached them — so every log-axis build (prime, each ROI/region spectrum
  without a cube, the skyline), the cube size estimate, the cube build and the dataset
  fingerprint each paid a full *serial* read. On a 7.3 GB centroid slide that was 3–5 minutes
  per call. The bounds are now computed once per dataset through the parallel reader,
  remembered on the store, and written into the `.parse.npz` sidecar (`amend_parse_cache`) so
  a reopen starts with them known.
- **Opening a large disk-backed slide could freeze the window for minutes after the load bar
  closed** (`gui/ion.py`, `gui/main.py`). With the spectrum view on *skyline*, the first paint
  streamed every spectrum off disk on the GUI thread, and clicking a peak before the fast cache
  existed did the same for each ion image. The fast-cache build now starts before the first
  paint; while it runs the spectrum view shows the already-primed mean (labelled *skyline
  pending fast cache*) and an ion-image request with no extracted column is parked with a
  status message. Both redraw from the cube the moment the build finishes. A picked feature
  whose apex-snapped m/z misses its extracted column now shows that exact column instead of
  streaming the slide.
- **Per-group analyses showed raw integer region ids instead of the assigned group names**
  (`spatial.py`, `registry.py`) — *Multi-group features* wrote a `top_region` column of raw
  integer cluster ids (e.g. `0 1 2 3`) that never reached a group name, and *Markers (shrunken
  centroids)* labelled its per-group table `region 0` because the run never carried the group
  names through. Both now surface the names the user assigned in Setup (`Cortex`, `Medulla`, …)
  in their result tables and exported CSVs; a genuine segmentation fallback with no assigned
  groups still reads `region <n>`. `multigroup_features` gained an optional `names=` argument and
  keeps the raw integer ids when it is omitted.

## [2.0.0] — Linear workflow — 2026-07-10

### Added
- **The feature-list export carries every analysis you have run** (`featuretable.py`,
  `gui/featureexport.py`) — *Feature list ▾ → Export CSV (with analyses)…* (or right-click the
  feature table) folds every completed run whose result is keyed by m/z into the exported table as
  a namespaced block of columns — `roi_comparison.AUC`, `discriminating_features.endo.q_value`,
  `pca.component0.loading`, `colocalize.score` — so one CSV opens in R / Excel / pandas with the
  annotation *and* the statistics already joined. Tick as few or as many analyses as you like (all,
  by default). Long-form results (one row per region or component × m/z) pivot to one column per
  key; two runs of the same analysis are named by what made them different
  (`roi_comparison_endo_vs_peri.AUC` beside `roi_comparison_peri_vs_epi.AUC`), falling back to an
  ordinal only when nothing distinguishes them; analyses that are not per-feature (lipid-class
  comparison, cross-validation, per-ion segmentation) are listed greyed with the reason rather
  than silently dropped. The join is nearest-m/z within an adjustable ppm window, so an analysis
  run before a recalibration still lands on its features, and an ion an analysis never scored
  gets a blank cell rather than losing its row.
- **Analyze gallery + run History (plan 24)** — every downstream analysis is now reached from a
  single **Analyze** card gallery and runs in one generic **Configure & Run** popup. Each run is
  recorded to a re-openable **Analyses → History** log (Open without recompute, Re-run, Duplicate,
  Add to report, Delete), with a staleness banner when the data has changed since. Run payloads
  live in a `<session>.runs/` sidecar; only a lightweight index is embedded in the session
  (`runs.py`, `gui/analysisdialog.py`, `gui/analyses.py`).

### Fixed
- **Analyses whose result is a dict of DataFrames persisted as garbage** — `RunStore.save_result`
  JSON-dumps a dict payload, and `json`'s `default=` hook is total (its last line is `str(o)`), so
  `filter_auc`, `marker_panel` and `region_membership` stored the *repr* of their tables: reopening
  such a run from History showed a meaningless string, and it could contribute nothing to an
  export. Those steps now persist their flat table instead — `runs.json_safe` decides — while
  dicts of arrays (ion segmentation, spectra overlay, classifier map) round-trip as JSON exactly
  as before.

### Changed
- **An A-vs-B run is titled by its contrast** — `Region comparison (A vs B) — endo vs peri`, and a
  target-ion analysis by its m/z, instead of the bare analysis name. `_scope_descriptor` records
  *every* group on the slide rather than the chosen pair, so two A/B runs were otherwise
  indistinguishable in History and in an export. Title only: a run's `inputs` digest — and so
  staleness detection — is unchanged.
- **The Flow designer is retired (plan 24)** — the analysis registry it wrapped lives on
  (`smile_msi/flow.py` renamed to `smile_msi/registry.py`), and the batch/replay surface is the
  scripting API (*Data → Analysis script…*). Saved flows under `~/.smile-msi/flows/` convert to
  scripting presets with `python scripts/migrate_flows_to_presets.py` — the migration preserves
  replicate-aware (`unit == 'sample'`) `roi_comparison` for `@grouped` scopes rather than silently
  regressing to per-pixel p-values.

## [1.0.1]

Closes the 2024–2026 MSI literature gaps catalogued in `LITERATURE_GAPS_2026.md`
(see `plans/` for the per-gap design docs) and the engine consistency / performance
audit summarized in `AUDIT_2026-06-24.md`.

### Added
- **Multi-sample batch correction (plan 01)** — ComBat harmonization wired into the
  Cohort tab and the pooled embedding, with an auto-batch-assignment menu and a
  batch-effect QC readout (`batchfx.py`).
- **Absolute quantification (plan 03)** — on-tissue calibration-curve workflow
  (define standard levels → fit response per analyte → apply to tissue pixels, with
  an internal-standard ratio option) and concentration-unit ion maps (`quantify.py`).
- **Community-standards round-trip (plan 02)** — richer imzML acquisition metadata +
  reporting dialog (`standards.py`).
- **MS/MS spectral-library matching (plan 04)** — cosine / modified-cosine / spectral-
  entropy scoring against an importable `.msp` / `.mgf` library, surfaced in the
  Features tab (`specmatch.py`).
- **Multimodal registration (plan 05)** — landmark optical↔MSI registration dialog
  (manual landmarks → affine) (`registration.py`).
- **Learned ion-image embeddings (plan 06)** — self-supervised colocalization /
  search embeddings (CPU inference) (`ionembed.py`).
- **Single-cell, 3D reconstruction, spatial multi-omics (plans 07 / 09 / 10)** —
  per-cell metabolite aggregation, serial-section stacking, and second-modality
  co-mapping dialogs (`singlecell.py`, `volume3d.py`, `comap.py`).

### Fixed
- **Cohort feature m/z now snap to a real apex, not a synthetic centroid
  (`cohort.consensus_targets`).** The shared cohort feature axis clustered each sample's
  picked peaks and emitted one **intensity-weighted centroid** per cluster — a synthetic
  m/z that need not sit on any slide's real peak. Combined comparison lists could therefore
  carry ions that land *between* two peaks and render as blank ion images, worst for the
  lowest-signal group (the "epi blanks" report; e.g. an annotated `886.5503` sitting in a
  valley). `consensus_targets` now emits the **real apex** (the most-intense pooled peak's
  m/z — a measured local maximum, computed once and shared across the cohort) and
  **deisotopes** the list by default, so a ¹³C M+1 like the PI 38:4 satellite near 886.55
  never becomes a standalone target. The clustering width is now the named
  `constants.CONSENSUS_TOL_PPM` (20 ppm). Extracted intensities shift (now integrated
  on-apex); `center="centroid"` reproduces the legacy values bit-for-bit. New
  `scripts/resnap_feature_lists.py` retrofits already-saved ★ lists — it snaps each stored
  m/z to the nearest real apex in its session (leaving genuinely-absent ions untouched),
  with an optional `--deisotope`.

### Changed
- **Chunked, lazily-read m/z cube cache (plan 17, `cubestore.py`).** The fast-ion cube
  sidecar is now a compressed, chunked **Zarr** store (`<stem>.cube.zarr`, Blosc/zstd)
  read on demand: reopening a slide pulls only the chunks an ion window touches instead of
  materialising the whole sparse cube in RAM. Warm-reopen resident memory for the cube goes
  from "the whole cube" (scales with non-zeros — e.g. +450 MB on a 0.5 GB cube) to a flat
  ~10 MB, while ion-image reads stay interactive (~7 ms warm); out-of-core makes slides
  larger than RAM browsable. `save_cube` dual-writes both the Zarr store and the legacy
  `.cache.npz` (one-release rollback path), and `load_cube` prefers the Zarr store and
  falls back to the npz; the `fingerprint` + pixel-count staleness guard and corrupt-cache
  safety are preserved bit-for-bit, and ion images / mean & max spectra are numerically
  identical to the in-RAM path. Adds `zarr>=2.17,<3` + `numcodecs`. The cube cache is now
  Zarr-only: the legacy uncompressed `.cache.npz` sidecar is no longer written, and an
  existing one is read and transparently **migrated** to a Zarr store on first reopen (then
  removed) — a one-time, best-effort upgrade that never fails a load. The `fingerprint` +
  pixel-count staleness guard and corrupt-cache safety are preserved on both paths.
- **Out-of-core cube build (plan 17 Phase 2).** `build_mz_cube(out_path=…)` assembles the
  cube straight to its on-disk store in two memory-bounded passes (stream+spill, then a
  column-band counting-sort to canonical CSC), so the whole cube never sits in RAM — peak
  build memory drops from ~4× the cube (the `parts`+`vstack` spike, e.g. 1.8 GB for a 0.43 GB
  cube) to a fixed ~0.3–0.4 GB regardless of cube size, making slides larger than RAM
  buildable. The result is byte-identical to the in-RAM build. The desktop app routes large
  disk-backed slides (estimated cube > 1.5 GB) through it automatically; smaller cubes keep
  the faster in-RAM build.
- **Engine consistency — named extraction tolerances (audit plan 18, Issue B).** The
  m/z extraction window is now a single documented `DEFAULT_TOL_PPM = 50.0`
  (`smile_msi/constants.py`); absolute quantification keeps its intentionally tighter
  `QUANT_TOL_PPM = 10.0`. Replaces ~60 scattered `tol_ppm` literals — behaviour
  unchanged, the intent is now explicit in one place.

### Fixed / performance (audit, see `AUDIT_2026-06-24.md`)
- **`comap.joint_embedding` block-balance weight was silently inert** — `weight` is now
  honoured (a `standardize=` flag stops the downstream `StandardScaler` from erasing it).
- **Normalization consistency** — `features_for_rows` median-norm uses the positive-only
  median, matching the other normalization paths.
- **Performance** — KD-tree cell-centroid mapping; hoisted FDR decoy-sort; debounced
  cohort-UMAP / joint-seg sliders; compact `blake2b` mask cache key for `mean_spectrum`
  (audit plan 23 E); shared normalize+align across MS/MS metrics (plan 23 A); optional
  shared isotope-score cache between feature-list build and FDR (plan 23 B).
- **Duplication removed** — shared helpers across the engine and GUI, including the
  export provenance-section assembly (plan 23 D) and `msi._finalize_norm` (plan 23 C).

### Scale-bar accuracy (2026-06-24)
- **Editable pixel size** — the Export dialog now exposes a *Pixel size (µm/px)* field,
  auto-filled from the imzML when recorded but overridable, so a file that omits the
  spacing (no bar otherwise) or records it wrongly can still be measured accurately.
  Writes back onto the dataset (`MSIDataset.set_pixel_size`) so every renderer uses it,
  and survives `to_ram`.
- **Reads `pixel size y`** — the imzML's vertical spacing is now read alongside x
  (`ImzMLStore`), and `MSIDataset.pixel_size_warning()` flags **non-square pixels** (the
  horizontal bar uses x; the image is vertically distorted) and a **non-unit coordinate
  grid** (`coordinate_grid_step()` — the displayed grid is coarser than the acquired one,
  so a naive bar is off by that factor). Surfaced as a caution in the Export dialog.
- **One rounding/clamp path** — `imaging._draw_scale_bar` (the figure exporters) now uses
  the same `annotations.nice_scalebar_um` / `format_scalebar_label` single source of truth
  as every other renderer: an over-long request snaps to a fitting `1-2-5` length and
  relabels instead of overflowing the frame with a caption that lies.

### Lipid-matching audit (2026-06-24)
- **Evidence-based candidate ranking** — when a dataset is present,
  `annotate.build_feature_list` now re-ranks each peak's candidates by their MSM score
  (mass × spectral-isotope × spatial-isotope) instead of by mass + class/adduct priors
  alone, so an isobaric runner-up the image clearly supports becomes the reported ID
  (METASPACE's ranking criterion). `ds=None` keeps the mass+prior order.
- **Target–decoy FDR corrected** — decoy elements are now a seeded *random* sample (not
  the first `n_decoy` alphabetically), each sample is an independent per-peak null, and
  the FDR is the **median over decoy samples** rather than one pooled ranking (which
  over-counted a peak coinciding with many decoy elements). The FDR target per peak is
  the best-MSM candidate, not merely the closest mass.
- **`annotate_mz` guards non-finite/zero m/z** — a NaN/0 m/z previously returned the
  *entire* database as NaN-scored candidates, leaking an arbitrary class label into
  `class_of` and class-level rollups; it now returns no candidates.
- **`build_database` returns a fresh list** — the `lru_cache`d enumeration is held as an
  immutable tuple and copied per call, so a caller sorting/filtering the result can no
  longer corrupt the shared database other callers see.
- **Deterministic isobaric tie-break** — equal-score candidates are ordered by
  class/name/adduct, so `class_of` no longer depends on DB insertion order.
- **Tolerance-aware priors** — class/adduct priors are scaled to `ppm_tol` so their
  combined tie-break swing is a fixed fraction of the mass window (unchanged at the
  default 10 ppm) instead of a constant ~0.95 ppm that dominated a tight window.
- **`[M-CH3]-` enumerated only for choline lipids** — headgroup demethylation is no
  longer generated for PE/FA/etc. (chemically impossible there), trimming spurious
  candidates and FDR-decoy noise.

### Annotation calibration & hardening (2026-07-01)

Fixes confidently-wrong lipid labels caused by an uncorrected m/z calibration offset (a
systematic ppm shift pushes the true `[M−H]⁻` out of the window so a contrived isobar wins —
e.g. a sulfatide read as a formate adduct of an odd-chain ether lipid).

- **Calibration check tool (new)** — *Data → Calibration check…* measures the dataset's
  **absolute** m/z offset against known reference ions (`intake.measure_calibration_offset` /
  `default_calibration_anchors`, anchors computed from the in-silico DB, not hardcoded), reports
  the median ppm / mass-slope / correction factor, and offers **one-click lock-mass
  recalibration** (fills the Preprocessing lock-mass field, applies `preprocess.recalibrate`, and
  re-verifies to < 2 ppm). Closes the gap between the spread-only QC (`intake.mass_drift`) and the
  fix. `METHODS.md` §1 / `USER_GUIDE.md` §3 document the recalibrate → tighten → verify workflow.
- **Exotic adducts gated by chemistry** — `[M+Cl]-` / `[M+HCOO]-` / `[M+CH3COO]-` are enumerated
  only for choline (PC/SM/PC-O/LPC) and neutral ceramide (HexCer/Cer) classes, never for
  acidic-headgroup lipids (PE/PS/PI/PG/FA/sulfatides), which ionize as `[M−H]⁻`. Removes the
  exotic-adduct false hits at the source (`match.py` `_ADDUCT_CLASS_WHITELIST`).
- **Database plausibility tightened** — odd total-carbon and >6-total-double-bond glycerolipids /
  free fatty acids are no longer generated (rare mass-coincidence species); sphingolipids and
  gangliosides are exempt (myelin has genuine odd-chain / 2-OH sulfatides). Reversible via
  `build_database(even_chain_only=…, max_total_db=…)` (`lipiddb.py`).
- **Base-ion tie-break** — on an exact score tie, the base `[M−H]⁻`/`[M+H]⁺` ion is preferred
  over an adduct form (then even-chain, then fewer double bonds); a sort-only rule that never
  overrides a closer mass.
- **Default identification tolerance 10 → 5 ppm** — brings the GUI, feature list, FDR, flow, and
  profile defaults in line with the already-5 ppm `Annotator`/`pipeline` defaults. Recalibrate
  before relying on it (see the workflow above).

## [1.0.0]

<!-- Set the release date here (YYYY-MM-DD) when you cut the public v1.0.0 tag. -->

### Added
- **Cohort-scale Samples roster** — the Samples panel now scales past a handful of slides: a
  **filter box** narrows the roster live by name / group / region / metadata; **Select matching**
  ticks the whole filtered set so one *Set group ▾* labels them all; **Auto-group** derives groups
  in one pass (by folder, by a name pattern, or by a metadata field); and **Import metadata…** loads
  a CSV that attaches per-sample metadata (and optionally the group), matched by sample name or
  imzML filename. Metadata shows inline on each row.
- **Regions scale too** — the Regions panel gains a **filter box** to find a region fast when a
  slide carries many, and the A/B region pickers (region comparison, ROI stats, lipid classes)
  gain **one-click group selection** — pick *all of Group A* vs *all of Group B* in a click each
  (regions carry the cohort group tag) instead of ticking every region — plus **All / None** and a
  filter inside the picker for long lists. Right-click a region selection for **batch rename**
  (prefix/suffix) and **tag with group**; the group tag shows inline on each row.

### Changed
- **Package renamed `lipidmatch` → `smile_msi`.** The Python import is now `import smile_msi`;
  the distribution is `smile-msi` and the console commands are `smile-msi` / `smile-msi-gui`.
  The desktop app, brand, and on-disk state (`~/.smile-msi/`, `$SMILE_MSI_HOME`) were already
  "SMILE MSI"; the old package name collided with the unrelated LipidMatch LC–MS/MS tool
  (Koelmel et al. 2017). (The `$LIPIDMATCH_HOME` / `$LIPIDMATCH_RAM_BUDGET_GB` env vars are
  renamed to `$SMILE_MSI_HOME` / `$SMILE_MSI_RAM_BUDGET_GB`.)
- **Cohort UMAP · compare selected regions across samples** — the *Region means* unit is now the
  clear path to compare named regions across the cohort: pick which named regions with **Regions ▾**
  and every matching region from every sample becomes a point (colour by Region / Sample / Group).
  Per-unit help text spells out what a point *is* so the region-vs-sample distinction stops tripping
  people up, and Region means now also runs on a **single slide** with ≥2 named regions (Pixels /
  Sample means still need ≥2 slides, since they pool *across* samples). A **modal loading popup**
  (with a per-sample progress bar, cancelable) covers the pooled run so a long embedding over many
  slides is unmissable instead of a quiet status-bar bar.
- **Faster bulk region creation** — *Regions from all segments* and *Auto-detect samples* now
  rebuild the region list / A/B pickers / segmentation map **once** for the whole batch instead of
  once per region (O(n) not O(n²)), so creating many regions at once no longer stutters.
- **SHAP biomarker discovery** — a supervised workflow (RandomForest + TreeExplainer) that ranks the
  molecular species driving a region/FTU classification and *which direction* each pushes, shown as a
  bubble plot (size = SHAP importance, colour = Spearman sign of intensity vs SHAP). Runs on one
  slide's segmentation clusters or drawn regions, or **per donor across a cohort** (one panel per
  region, donor rows labelled with optional age/sex/BMI). New **SHAP biomarkers** tab (Advanced),
  an Analysis-Flow/Script step, and report logging. `shap` is an optional dependency
  (`uv sync --extra shap`). See `METHODS.md` §5.
- **Export Studio** — one place to batch-export across feature lists *and* sections at once, instead
  of switching lists + sections and re-exporting per group. Pick ions across any number of saved ★
  lists, ◆ lipid-class lists (one composite image per class, or expand to member ions) and working
  scopes; choose which sections to render them across — the live slide, **any region you've drawn on
  it** (each rendered as a close-up, with its own “Edit crop…” to set the framing), and any cohort
  samples; and tick the outputs you want — per-ion photos in an organised folder tree
  (`<list>/<section>/ion_…`), a
  combined **ion × section matrix** figure, the feature-list CSVs, and the analyses bundle. Cross-
  section rendering loads each slide's cube once and evicts it before the next (peak RAM ≈ one cube),
  with a **shared intensity scale per ion** by default so the same lipid is honestly comparable
  between slides (toggle to per-section auto-contrast). Writes a `manifest.csv` recording every
  section × ion cell. Reachable from *File → Export Studio…* (⇧⌘E), the ⌘E export hub, the Report
  tab, and the Samples panel; the assembled plan is saved per-sample like the report contents.
- **Analysis Script console** — drive the analysis engine with Python. Write a short script against
  pre-bound bare-name functions (`find_peaks`, `segment`, `compare`, `colocalize`, `annotate`, `pca`,
  …) plus `ds` / `np`, run it on the loaded slide (⌘/Ctrl+Return), and see tables + ion images + a
  log. Output helpers (`log` / `table` / `image` / `record`) surface results; the run can apply its
  features / segmentation back to the app. Scripts save as reusable **workflows** in
  `~/.smile-msi/workflows/`. A built-in **AI guide** button (and [`SCRIPTING.md`](SCRIPTING.md))
  emits a self-describing reference of every function + the runtime context, to hand to an AI so it
  can write a correct workflow. Shares the Analysis Flow registry, so scripted and designed analyses
  behave identically. (*Data → Analysis script…*, ⇧⌘J)
- **Cluster tree (HCA)** — the Segmentation tab now draws the hierarchical-cluster-analysis tree
  the Detail slider cuts, with a draggable cut line: drag it down for more/finer segments, up for
  fewer/coarser. Branch colours match the segmentation map, and the line and the Detail slider stay
  in sync (each is a handle on the same cut). The tree is truncated to readable cluster blocks and
  drawn on a √ distance scale, so a big vertical gap shows a natural place to cut.
- **Analysis Flow** — a saveable / replayable analysis pipeline. Assemble an ordered recipe of
  steps (find peaks → segment → discriminating / region comparison → co-localization → identify)
  over labelled ROIs once, then re-run it on a single slide or across the whole cohort; results
  route live to their views, log to the Report, and export as an ion-image + table bundle.
  Presets save to `~/.smile-msi/flows/`. (*Data → Analysis flow…*)

### Changed
- **License is now Apache-2.0** (was MIT) — adds an express patent grant + patent-retaliation
  clause, a better fit given the MSI patent landscape. Added `NOTICE` and
  `THIRD_PARTY_LICENSES.md` (the optional desktop GUI links PySide6 / Qt under LGPL-3.0).
- **Segmentation defaults to Ward** (agglomerative) instead of bisecting k-means in both the
  single-slide and joint/cohort views; bisecting k-means is still selectable.
- **Per-spectrum "median" normalization removed from the UI** (the engine still supports
  `norm="median"` so older sessions keep loading) — an IP-hygiene change pending a patent review.
- **Renamed the user-facing spatial finder label** to "Spatial feature finder."
- **Decluttered ROI / brush toolbar** — the ion-image tool strip now reflows onto extra rows
  instead of overlapping when the panel is narrow, the freehand **Draw/Erase + brush-size**
  controls moved into the *ROI options ▾* popover (shown only for the Freehand shape), and the
  shape picker no longer truncates ("Freehand (brush)" reads in full).
- **Demo loads fully live** — *Load demo* now auto-finds peaks and opens on a real ion image
  (no need to click *Find peaks* first).

### Fixed
- Friendlier errors (tracebacks moved behind "Show Details"; clear messages for a missing `.ibd`
  and a corrupt session file); crash guards on the cohort batch comparison and feature-space lasso;
  empty-state guidance on the analysis tabs; grouped-tab-safe navigation from volcano plots.
- **Script Console robustness** — closing the console (window-close or Escape) mid-run now signals
  the worker to stop and restores the Run/Cancel/editor state instead of orphaning the run; a
  malformed table/image or a failing **AI guide** falls back to an inline note instead of crashing
  the dialog; a workflow that produces nothing now says so explicitly.
- **Undo for the optical backdrop** — ⌘Z now restores *Reset alignment* and *Remove image* on the
  optical/histology layer (previously outside the undo system).
- **Export data book** — a data bundle with every file unticked is refused with a hint instead of
  silently writing an empty folder.
- **Analysis Flow** — an empty flow shows a "click ➕ Add step" placeholder, and a failed preset
  save/load surfaces a clear message instead of failing silently.
- **macOS app version** — `scripts/make_macos_app.sh` now reads `__version__` from the package, so
  the `.app` bundle version can't drift from `pyproject` / the in-app update check (was hardcoded
  `0.3.0`).
- **Consistent Windows-safe export filenames** — every screen now derives save names from one shared
  helper (`studio.safe_name`) instead of five slightly different per-screen sanitisers. The feature-
  list export default no longer leaves Windows-forbidden characters (`: * ? " < > |`) in the name —
  so a working scope or list called e.g. `PG 46:2` exports as `PG_46-2` everywhere (it previously
  kept the `:` and could fail to write on the lab's Windows machines), and the same label now yields
  the same filename across Export, Stats, Export Studio, and Analysis-Flow bundles.
- **Analysis-Flow data bundle** — a table that fails to write now prints why (matching the ion-image
  branch) instead of vanishing silently, so a partial bundle no longer hides a write error.

### Tests & docs
- ~150 new tests: engine unit-test backfill for `spectrum_range`, `annotate`, `match`, `imaging`,
  `msms`, `provenance`; Flow `scope_region` serialization + mask resolution; Script Console ↔
  MainWindow integration; dendrogram ↔ Detail-slider edge cases; and the robustness fixes above.
- New `CONTRIBUTING.md` and `RELEASE.md`; runnable `examples/` (scripts + flow presets); documented
  the Flow **Restrict to:** scope control, the `~/.smile-msi/{flows,workflows}` preset formats, and
  that `SCRIPTING.md` is generated from the in-app AI guide (`scripting.capabilities_doc()`).

## [0.3.0] — Desktop MSI workspace

A ground-up expansion from a peak-table annotator into a
**desktop MALDI-MSI workspace** (PySide6 + pyqtgraph), backed by a streaming,
out-of-core engine. 50+ tests, CI, and packaging.

### Added
- **Ingestion** — lazy/out-of-core imzML/ibd; subsample quick-load; imaging-table
  import (long/wide CSV/TSV); convert any dataset → imzML; physical pixel size for scale bars.
- **Imaging** — ion images, TIC, hotspot/quantile contrast, colormaps, RGB overlay,
  multi-ion **montage**, single-pixel & region spectra, **optical/H&E overlay**,
  **lipid-class composite images** (total-sulfatide / total-ganglioside), publication PNG export.
- **Preprocessing** — SNIP baseline, Savitzky-Golay / Gaussian smoothing, m/z recalibration,
  vector / reference normalization.
- **Unsupervised** — peak picking; **spatial feature finding** (Moran's I); segmentation
  (k-means, auto-k, spatially-aware); PCA & NMF component images; t-SNE / UMAP embedding.
- **Co-localization** — target ranking, full matrix, and **module detection**.
- **Statistics** — exact rank ROC AUC + Mann-Whitney + BH-FDR between two regions
  (drawn ROIs or clusters), one-vs-rest discriminating features, Kruskal-Wallis multi-group,
  **volcano plot**, a **live per-ion ROC curve** (click any ion to see its curve, with a
  selective **CSV / PDF-with-graphs report**), side-by-side region comparison. Optional **parametric tests** —
  Welch's / Student's t-test for two-group (`roi_comparison`, cohort `group_comparison`) and
  one-way **ANOVA** for multi-group (`multigroup_features`) — via a `method=` argument.
- **Identification** — in-silico m/z→lipid across **29 classes** (glycerophospholipids,
  ether/plasmalogens, lyso, sphingolipids, **gangliosides**, **sterols & cholesteryl esters**,
  **acylcarnitines**, **oxylipins**, fatty acids, **neural metabolites**); feature list with
  alternatives, isotope-pattern + adduct corroboration, confidence, deisotoping, target-decoy
  **annotation FDR**, research links; **MS/MS confirmation** (diagnostic fragments / neutral
  losses, acyl-chain readout, unknown-spectrum classification).
- **Workflow** — session save/load, provenance + methods report (JSON / Markdown / Excel sheet),
  in-app Help + User Guide, one-click launchers, standalone-app PyInstaller spec, GitHub Actions CI.
- **Performance** — sparse m/z cube for instant arbitrary-m/z ion images, bincount binning,
  LRU ion cache, adaptive feature build; cooperative Cancel on long operations.
- **Large-RAM acceleration** — `MSIDataset.to_ram()` reads a continuous-mode dataset once
  into a dense in-RAM float32 matrix (budget-guarded by physical RAM /
  `$SMILE_MSI_RAM_BUDGET_GB`), making the mean-spectrum and feature passes vectorized and
  disk-free (feature extraction ~300× faster); imzML loads use it automatically when it fits.
- **Cube cache sidecar** — the fast m/z cube + prime stats are persisted next to the
  managed session (`<stem>.cache.npz`, fingerprint-guarded), so reopening a sample restores
  them instead of re-streaming the whole acquisition to rebuild.

- **ROI close-up exports + Crop Studio** — every image export (ion image, colour overlay,
  gallery, segmentation map) can zoom in on a region of interest, the publication convention:
  the panel crops to the ROI, traces its outline in the region colour, and fades the surround,
  while keeping the full-view contrast and a correctly-scaled bar. Crop to one region, or fan
  out one close-up per region. The crop is set in a dedicated **Crop Studio** — a focused
  dialog with the ion image, a draggable eight-handle crop box, and a **live WYSIWYG preview**
  of the exported panel. It opens automatically the first time a region is exported as a
  close-up, and is re-editable from the region list. A **project standard** (aspect ratio +
  optional fixed physical size) can be set as the default and applied to every region, so
  close-ups are reproducibly framed. The crop and the project standard persist in the session.

- **Find peaks scoping & auto-save** — the Find peaks dialog now has a **Region** selector
  (whole dataset, or any named region's pixels) instead of a single "selected region" toggle,
  plus an **Also save as a feature list (★)** option that snapshots the picked features into a
  named library list in one step.

### Changed
- **Spectra auto-crop to the signal** — every spectrum view (Find-peaks mean/skyline, region &
  ROI overlays, the region-comparison butterfly, and exported spectrum figures + ion-panel side
  spectra) now opens cropped to the m/z window that actually holds peaks, trimming the long empty
  tail past the last real peak so plots read like publication figures instead of a sliver of peaks
  lost in dead space. One shared `spectrum_range.signal_mz_range` helper backs both the live
  pyqtgraph views and the headless export engine; the Find-peaks view still pans/zooms to the full
  acquired range, and an explicit export `mz_range` still overrides the crop.
- **Instant skyline spectrum** — switching the Find-peaks spectrum to *skyline (max)* now serves
  the per-m/z maximum from the prebuilt m/z cube (like the mean backdrop) instead of streaming
  every pixel off disk on the GUI thread, so it no longer triggers a long freeze; the exact
  cache-less path also now streams through the shared thread pool.
- **GUI-only**: the `smile_msi` command now launches the desktop app (the old
  `analyze`/`annotate` CLI subcommands and the Streamlit UI were removed).
- **Slimmer data bundle** — the report's CSV companion was reworked to ship only reusable
  supplementary data (per the SMART / MIAMSIE reporting standards): a SMART-style feature
  table, per-region intensity statistics, discriminating-feature statistics, an **opt-in**
  raw-spectra dump, and one compact `methods.md` (folding in the dataset summary, provenance,
  references and file manifest). The verbose `provenance.json`, `README.txt` and redundant
  `dataset_summary.csv`/`ion_images.csv` are gone. Each file is a toggle in the Export hub,
  with a **live preview** of the exact columns/rows the bundle will contain.

## [0.2.0]
- First spatial engine + Streamlit imaging UI (superseded by the desktop app).

## [0.1.0]
- Offline in-silico m/z → lipid annotation and two-group AUC for exported peak tables.
