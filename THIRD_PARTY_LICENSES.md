# Third-Party Licenses

SMILE MSI (package `smile_msi`) is released under the **Apache License 2.0** (see `LICENSE`).
It depends on the third-party packages below, each under its own license. Versions reflect
the resolved environment; run `pip show <pkg>` for the exact installed version/license.

## Runtime dependencies (core)

| Package | License | SPDX |
|---|---|---|
| numpy | BSD 3-Clause | `BSD-3-Clause` |
| pandas | BSD 3-Clause | `BSD-3-Clause` |
| scipy | BSD 3-Clause | `BSD-3-Clause` |
| scikit-learn | BSD 3-Clause | `BSD-3-Clause` |
| matplotlib | Matplotlib License (PSF-based, BSD-compatible) | `matplotlib` / `PSF-2.0` |
| pillow | Historical Permission Notice and Disclaimer (open, MIT-style) | `HPND` |
| openpyxl | MIT | `MIT` |
| pyimzml | Apache 2.0 | `Apache-2.0` |
| statsmodels | BSD 3-Clause | `BSD-3-Clause` |
| umap-learn | BSD 3-Clause | `BSD-3-Clause` |
| numba | BSD 2-Clause | `BSD-2-Clause` |
| llvmlite (numba backend) | BSD 2-Clause | `BSD-2-Clause` |
| zarr | MIT | `MIT` |
| numcodecs | MIT | `MIT` |

All of the above are permissive (BSD/MIT/Apache/HPND) and impose no copyleft obligation on
SMILE MSI or on works that depend on it.

## Optional desktop GUI (`pip install smile_msi[gui]`)

| Package | License | SPDX |
|---|---|---|
| PySide6-Essentials (Qt for Python) | **LGPL-3.0** (also offered under GPL-2.0 / GPL-3.0) — used here under LGPL-3.0 | `LGPL-3.0-only` |
| shiboken6 | **LGPL-3.0** (with PySide6) | `LGPL-3.0-only` |
| pyqtgraph | MIT | `MIT` |
| pyobjc-framework-Cocoa (macOS only) | MIT | `MIT` |
| qtawesome | MIT | `MIT` |
| qtpy (qtawesome dep) | MIT | `MIT` |

### Icon-font attribution (qtawesome)

The toolbar/button icons are rendered by **qtawesome** (MIT), which bundles several icon
fonts. SMILE MSI uses the **Font Awesome 6 Free — Solid** set (`fa6s.*`). The bundled fonts
are all permissive and compatible with this Apache-2.0 project:

- **Font Awesome Free** — icons under **CC BY 4.0** (attribution required), webfonts under
  **SIL OFL 1.1**, code under **MIT**. © Fonticons, Inc. — https://fontawesome.com
- Material Design Icons (`mdi6.*`, **Apache-2.0** / SIL OFL 1.1), Phosphor (`ph.*`, **MIT**),
  Remix Icon (`ri.*`, **Apache-2.0**), Elusive (`ei.*`, **SIL OFL 1.1**), and Microsoft
  Codicons (`msc.*`, **CC BY 4.0**) are also bundled by qtawesome but only used if referenced.

The built binaries bundle the required attribution and license text at
`licenses/Font-Awesome-CC-BY-4.0.txt` and `licenses/CC-BY-4.0.txt` (copied into the frozen
app by `smile_msi.spec`), which satisfies CC BY 4.0 for the Font Awesome icons shipped in any
built binary.

### LGPL-3.0 compliance note (PySide6 / shiboken6 / Qt)

Qt for Python is used under the **LGPL-3.0**. SMILE MSI links it **dynamically** (a normal
`pip` install of PySide6), which the LGPL permits from a project under any license, including
this Apache-2.0 project. To stay compliant when **distributing a built binary** (e.g. a
PyInstaller `.app` / `.exe`):

- Ship the Qt/PySide6 libraries as **replaceable shared libraries** (PyInstaller's default —
  do not statically link or modify Qt).
- Include the **LGPL-3.0 license text** with the distribution — bundled at
  `licenses/LGPL-3.0.txt` (with the incorporated `licenses/GPL-3.0.txt`) via `smile_msi.spec`.
- Allow end users to **relink/replace** the Qt libraries with a modified version (the
  one-folder PyInstaller bundle keeps PySide6 as replaceable `.dll`/`.dylib` files).

No changes are made to Qt or PySide6 source.

## Optional extras (installed only if you request the extra)

| Package | Extra | License | SPDX | Used for |
|---|---|---|---|---|
| shap | `[shap]` | MIT | `MIT` | SHAP biomarkers |
| colorcet | `[studio]` | CC-BY-4.0 / MIT | `CC-BY-4.0` | perceptual colormaps |
| scikit-image | `[register]` | BSD 3-Clause | `BSD-3-Clause` | image registration |
| SimpleITK | `[register]` | Apache 2.0 | `Apache-2.0` | image registration |
| cellpose | `[cells]` | BSD 3-Clause | `BSD-3-Clause` | cell segmentation |
| stardist | `[cells]` | BSD 3-Clause | `BSD-3-Clause` | cell segmentation |
| PyOpenGL | `[viz3d]` | BSD 3-Clause | `BSD-3-Clause` | 3-D volume view |
| anndata | `[spatialomics]` | BSD 3-Clause | `BSD-3-Clause` | AnnData / SpatialData export |
| metaspace2020 | `[metaspace]` | Apache 2.0 | `Apache-2.0` | METASPACE client |
| torch | `[ion-embed]` | BSD 3-Clause | `BSD-3-Clause` (+ `LicenseRef-NVIDIA-CUDA`) | learned ion embeddings |

> `[embed]` (`umap-learn`) is retained for backwards compatibility only — `umap-learn` is now a
> **core** dependency (see the core table above), so the extra installs nothing new.

> ⚠️ **`[ion-embed]` is not "all permissive."** A GPU `torch` wheel pulls in NVIDIA's CUDA
> runtime libraries, distributed under the proprietary **NVIDIA CUDA EULA** (not an OSI-approved
> license). If you redistribute a binary built with `[ion-embed]`, review the NVIDIA CUDA
> redistribution terms separately. The `[ion-embed]` extra is not part of the default GUI/build.

## Build-time only (not shipped as a library)

| Package | Extra | License | Note |
|---|---|---|---|
| PyInstaller | `[build]` | GPL-2.0 **with a bootloader exception** | A **build tool**, not a runtime dependency. PyInstaller's runtime bootloader carries an explicit exception that lets you ship the produced application under any license, so it imposes **no GPL obligation** on SMILE MSI or its binaries. |

---

*This file is a good-faith engineering summary, not legal advice. Verify the exact license of
each pinned version before a public/commercial release; consider generating it automatically
(e.g. `pip-licenses`) as part of the release checklist in `RELEASE.md`.*
