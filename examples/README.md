# Examples

Runnable starting points for the two ways to script SMILE MSI:

| Folder | What it is | Where it runs |
| --- | --- | --- |
| [`scripts/`](scripts) | Short **Python scripts** that drive the analysis engine | **Script Console** — *Data → Analysis script…* (⇧⌘J) |
| [`workflows/`](workflows) | **Analysis Flow** presets (declarative, click-to-run pipelines) | **Analysis Flow** — *Data → Analysis flow…* |

There's also [`run_demo.py`](run_demo.py) — the headless library demo (`python examples/run_demo.py`), which annotates [`sample_peaks.csv`](sample_peaks.csv) and writes an Excel report. That one runs from a terminal, not inside the app.

---

## Scripts (`scripts/`)

Each `.py` file is a self-contained Script Console snippet. The analysis functions are
**pre-bound bare names** — `find_peaks()`, `segment()`, `compare(...)`, `colocalize()`,
`annotate()`, … — plus `ds` (the loaded dataset), `np` (numpy), and the output helpers
`log()` / `table()` / `image()` / `record()`. There are no imports to add and nothing to
`return`; you surface results through those helpers. See [`../SCRIPTING.md`](../SCRIPTING.md)
for the full reference (same text as the console's **AI guide** button).

| File | Does |
| --- | --- |
| [`find_and_annotate.py`](scripts/find_and_annotate.py) | Find peaks, match them to the lipid database, table the confident IDs. |
| [`segment_then_markers.py`](scripts/segment_then_markers.py) | Auto-segment the tissue, then list automatic marker ions per region. |
| [`compare_two_groups.py`](scripts/compare_two_groups.py) | Per-ion A-vs-B comparison (AUC + FDR), then map the most enriched ion. |
| [`colocalization.py`](scripts/colocalization.py) | Rank features by spatial similarity to a target ion and map the top hit. |

### How to import a script

1. Open a slide, then **Data → Analysis script…** (⇧⌘J).
2. **Workflows → Open file…** and pick a `.py` from `scripts/` — *or* just open the
   file in any editor and paste its contents into the console.
3. Press **▶ Run** (⌘/Ctrl+Return). Tables and ion images appear as tabs; the **Log**
   tab shows prints, recorded values, and any error (with your line numbers).
4. **Apply to app** pushes the script's features or segmentation back into the app's views.
5. **Workflows → Save as…** stores your edited version under `~/.smile-msi/workflows/`
   so it shows up in the console's list on every slide.

> **Setup note** — `compare_two_groups.py` needs two **groups**: draw and name **Regions**
> on the *Ion image* tab and tag them **Group A / B**. The other three scripts run on any
> loaded slide as-is (with no ROIs, group/marker steps fall back to a fresh segmentation).

---

## Workflows (`workflows/`)

These JSON files are **Analysis Flow** presets — an ordered list of analysis steps with
their parameters, matching `smile_msi.flow.Flow.to_dict`. A flow holds **no data** (no
masks, no arrays), so a preset is portable across slides.

| File | Pipeline |
| --- | --- |
| [`segment_then_compare.json`](workflows/segment_then_compare.json) | Find spatial features → segment → compare Group A vs B → identify lipids. |
| [`peaks_pca_coloc.json`](workflows/peaks_pca_coloc.json) | Find peaks → identify lipids → PCA images → co-localization modules. |

### How to import a workflow

The Flow designer reads its presets from `~/.smile-msi/flows/`, so copy a preset there:

```sh
mkdir -p ~/.smile-msi/flows
cp examples/workflows/*.json ~/.smile-msi/flows/
```

Then in the app: **Data → Analysis flow…**, and the presets appear in the **Load preset**
list. Select one, choose **Run on → This slide** (or **Across slides (cohort)**), and run.
Results route live to their views; tick **Log to Report** to drop them into the PDF data
book and **Export bundle** to write the ion images + tables to a folder.

> **Setup note** — `segment_then_compare.json` includes a **Region comparison (A vs B)**
> step. Tag two groups in the flow's **Setup** table first; if you haven't, the comparison
> falls back to the most recent **segmentation** (which the same flow produces one step
> earlier), so it still runs.

---

## Scripts vs. flows — which one?

- **Flows** are declarative and shareable: assemble a recipe in a form, save it, replay it
  slide after slide (or across a whole cohort). Best when the built-in steps cover what you
  need.
- **Scripts** are imperative and open-ended: full Python over the same engine, for logic a
  flow can't express (conditionals, custom tables/images, anything ad-hoc). Hand the
  console's **AI guide** (or [`../SCRIPTING.md`](../SCRIPTING.md)) to an AI assistant and
  ask it to write one for you.

Both drive the **same** analysis engine (`smile_msi.flow.REGISTRY`), so a result never
differs between the two.
