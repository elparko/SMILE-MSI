"""Provenance & traceability — a publishable record of exactly how a result was made.

Records the input file (with a checksum/fingerprint), the dataset's properties, every
processing step and its parameters and random seeds, the software/library versions, and
the outputs produced. Emits three forms:

* :meth:`Provenance.to_json` — machine-readable, complete record (for a supplement / re-run).
* :meth:`Provenance.to_markdown` — a human-readable methods/traceability report.
* :meth:`Provenance.methods_paragraph` — an auto-drafted Methods paragraph for a manuscript.

The engine functions stay pure; the orchestrator (CLI / GUI / script) records what it
ran. Pass an explicit ``now`` for deterministic output in tests.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass, field

_PACKAGES = ["numpy", "scipy", "scikit-learn", "pandas", "pyimzml", "matplotlib",
             "openpyxl", "pyqtgraph", "PySide6-Essentials"]

# Verified literature/software citations for the methods this app can run, keyed by a
# short tag. methods_paragraph() cites these inline and references_used() emits the
# subset that a given analysis actually touched. Full catalog: METHODS.md.
REFERENCES = {
    "snip": "Ryan, C.G., Clayton, E., Griffin, W.L., Sie, S.H. & Cousens, D.R. (1988). SNIP, a "
            "statistics-sensitive background treatment for the quantitative analysis of PIXE "
            "spectra. Nucl. Instrum. Methods Phys. Res. B, 34(3), 396–402. doi:10.1016/0168-583X(88)90063-8",
    "snip_lls": "Morháč, M. & Matoušek, V. (2008). Peak clipping algorithms for background "
                "estimation in spectroscopic data. Appl. Spectrosc., 62(1), 91–106. doi:10.1366/000370208783412762",
    "savgol": "Savitzky, A. & Golay, M.J.E. (1964). Smoothing and differentiation of data by "
              "simplified least squares procedures. Anal. Chem., 36(8), 1627–1639. doi:10.1021/ac60214a047",
    "tic": "Deininger, S.-O. et al. (2011). Normalization in MALDI-TOF imaging datasets of "
           "proteins: practical considerations. Anal. Bioanal. Chem., 401(1), 167–181. doi:10.1007/s00216-011-4929-z",
    "mad": "Rousseeuw, P.J. & Croux, C. (1993). Alternatives to the median absolute deviation. "
           "J. Am. Stat. Assoc., 88(424), 1273–1283. doi:10.1080/01621459.1993.10476408",
    "morans": "Moran, P.A.P. (1950). Notes on continuous stochastic phenomena. Biometrika, "
              "37(1–2), 17–23. doi:10.1093/biomet/37.1-2.17",
    "pca_pearson": "Pearson, K. (1901). On lines and planes of closest fit to systems of points "
                   "in space. Philos. Mag., 2(11), 559–572. doi:10.1080/14786440109462720",
    "pca_hotelling": "Hotelling, H. (1933). Analysis of a complex of statistical variables into "
                     "principal components. J. Educ. Psychol., 24(6), 417–441. doi:10.1037/h0071325",
    "kmeans": "Lloyd, S.P. (1982). Least squares quantization in PCM. IEEE Trans. Inf. Theory, "
              "28(2), 129–137. doi:10.1109/TIT.1982.1056489",
    "minibatch": "Sculley, D. (2010). Web-scale k-means clustering. Proc. 19th Int. Conf. on World "
                 "Wide Web (WWW '10), 1177–1178. doi:10.1145/1772690.1772862",
    "ward": "Ward, J.H. (1963). Hierarchical grouping to optimize an objective function. J. Am. "
            "Stat. Assoc., 58(301), 236–244. doi:10.1080/01621459.1963.10500845",
    "bisecting": "Steinbach, M., Karypis, G. & Kumar, V. (2000). A comparison of document clustering "
                 "techniques. KDD Workshop on Text Mining.",
    "silhouette": "Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation and "
                  "validation of cluster analysis. J. Comput. Appl. Math., 20, 53–65. doi:10.1016/0377-0427(87)90125-7",
    "combat": "Johnson, W.E., Li, C. & Rabinovic, A. (2007). Adjusting batch effects in microarray "
              "expression data using empirical Bayes methods. Biostatistics, 8(1), 118–127. "
              "doi:10.1093/biostatistics/kxj037",
    "recombat": "Adler, M. et al. (2022). reComBat: batch-effect removal in large-scale multi-source "
                "gene-expression data integration. Bioinformatics Advances, 2(1), vbac016. "
                "doi:10.1093/bioadv/vbac016",
    "kbet": "Büttner, M., Miao, Z., Wolf, F.A., Teichmann, S.A. & Theis, F.J. (2019). A test metric "
            "for assessing single-cell RNA-seq batch correction. Nature Methods, 16(1), 43–49. "
            "doi:10.1038/s41592-018-0254-1",
    "metaspace": "Palmer, A. et al. (2017). FDR-controlled metabolite annotation for imaging mass "
                 "spectrometry. Nature Methods, 14(1), 57–60. doi:10.1038/nmeth.4072",
    "miamsie": "McDonnell, L.A. et al. (2015). Discussion point: reporting guidelines for mass "
               "spectrometry imaging. Anal. Bioanal. Chem., 407(8), 2035–2045. doi:10.1007/s00216-014-8322-6",
    "imzml": "Schramm, T. et al. (2012). imzML — a common data format for the flexible exchange and "
             "processing of mass spectrometry imaging data. J. Proteomics, 75(16), 5106–5110. "
             "doi:10.1016/j.jprot.2012.07.026",
    "qmsi": "Quantitative MALDI imaging via on-tissue calibration / standard addition; e.g. Hamm, G. "
            "et al. (2012), J. Proteomics 75(16):4952, doi:10.1016/j.jprot.2012.07.035; absolute "
            "case Anal. Chem. (2016) doi:10.1021/acs.analchem.5b04409.",
    "entropy_sim": "Li, Y. et al. (2021). Spectral entropy outperforms MS/MS dot product similarity "
                   "for small-molecule compound identification. Nature Methods, 18, 1524–1531. "
                   "doi:10.1038/s41592-021-01331-z",
    "spectral_cosine": "Stein, S.E. & Scott, D.R. (1994). Optimization and testing of mass spectral "
                       "library search algorithms for compound identification. J. Am. Soc. Mass "
                       "Spectrom., 5(9), 859–866. doi:10.1016/1044-0305(94)87009-8",
    "umeyama": "Umeyama, S. (1991). Least-squares estimation of transformation parameters between two "
               "point patterns. IEEE Trans. Pattern Anal. Mach. Intell., 13(4), 376–380. doi:10.1109/34.88573",
    "skimage": "van der Walt, S. et al. (2014). scikit-image: image processing in Python. PeerJ, 2, "
               "e453. doi:10.7717/peerj.453",
    "simpleitk": "Lowekamp, B.C. et al. (2013). The design of SimpleITK. Front. Neuroinform., 7, 45. "
                 "doi:10.3389/fninf.2013.00045",
    "spacem": "Rappez, L. et al. (2021). SpaceM reveals metabolic states of single cells. Nature "
              "Methods, 18(7), 799–805. doi:10.1038/s41592-021-01198-0",
    "watershed": "Beucher, S. & Meyer, F. (1993). The morphological approach to segmentation: the "
                 "watershed transformation. In Mathematical Morphology in Image Processing, 433–481.",
    "vol_reg": "Reddy, B.S. & Chatterjee, B.N. (1996). An FFT-based technique for translation, "
               "rotation and scale-invariant image registration. IEEE Trans. Image Process., 5(8), "
               "1266–1271. doi:10.1109/83.506761",
    "vol_render": "Levoy, M. (1988). Display of surfaces from volume data. IEEE Comput. Graph. Appl., "
                  "8(3), 29–37. doi:10.1109/38.511",
    "comap_multiomics": "Mass spectrometry imaging for spatially resolved multi-omics. npj Imaging "
                        "(2024). doi:10.1038/s44303-024-00025-3",
    "spearman": "Spearman, C. (1904). The proof and measurement of association between two things. "
                "Am. J. Psychol., 15(1), 72–101. doi:10.2307/1412159",
    "deepion": "Wang, L. et al. (2024). DeepION: a deep learning-based low-dimensional representation "
               "model of ion images for mass spectrometry imaging. Anal. Chem., 96(9), 3829–3838. "
               "doi:10.1021/acs.analchem.3c05002",
    "simclr": "Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020). A simple framework for "
              "contrastive learning of visual representations (SimCLR). Proc. ICML, 1597–1607.",
    "spatial_seg": "Alexandrov, T. & Kobarg, J.H. (2011). Efficient spatial segmentation of large "
                   "imaging mass spectrometry datasets with spatially aware clustering. Bioinformatics, "
                   "27(13), i230–i238. doi:10.1093/bioinformatics/btr246",
    "nmf": "Lee, D.D. & Seung, H.S. (1999). Learning the parts of objects by non-negative matrix "
           "factorization. Nature, 401(6755), 788–791. doi:10.1038/44565",
    "tsne": "van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. J. Mach. Learn. "
            "Res., 9, 2579–2605.",
    "umap": "McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold Approximation and "
            "Projection for dimension reduction. arXiv:1802.03426. doi:10.48550/arXiv.1802.03426",
    "mwu": "Mann, H.B. & Whitney, D.R. (1947). On a test of whether one of two random variables is "
           "stochastically larger than the other. Ann. Math. Stat., 18(1), 50–60. doi:10.1214/aoms/1177730491",
    "auc": "Hanley, J.A. & McNeil, B.J. (1982). The meaning and use of the area under a receiver "
           "operating characteristic (ROC) curve. Radiology, 143(1), 29–36. doi:10.1148/radiology.143.1.7063747",
    "bh_fdr": "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a "
              "practical and powerful approach to multiple testing. J. R. Stat. Soc. B, 57(1), "
              "289–300. doi:10.1111/j.2517-6161.1995.tb02031.x",
    "kruskal": "Kruskal, W.H. & Wallis, W.A. (1952). Use of ranks in one-criterion variance "
               "analysis. J. Am. Stat. Assoc., 47(260), 583–621. doi:10.1080/01621459.1952.10483441",
    "ebayes": "Smyth, G.K. (2004). Linear models and empirical Bayes methods for assessing "
              "differential expression in microarray experiments. Stat. Appl. Genet. Mol. Biol., "
              "3(1), Article 3. doi:10.2202/1544-6115.1027",
    "meanstest": "Bemis, K.D., Harry, A., Eberlin, L.S., Ferreira, C.R., van de Ven, S.M., "
                 "Mallick, P., Stolowitz, M. & Vitek, O. (2016). Probabilistic segmentation of "
                 "mass spectrometry (MS) images helps select important ions and characterize "
                 "confidence in the resulting segments. Mol. Cell. Proteomics, 15(5), 1761–1772. "
                 "doi:10.1074/mcp.O115.053918",
    "lmm": "Laird, N.M. & Ware, J.H. (1982). Random-effects models for longitudinal data. "
           "Biometrics, 38(4), 963–974. doi:10.2307/2529876",
    "camera": "Wu, D. & Smyth, G.K. (2012). Camera: a competitive gene set test accounting for "
              "inter-gene correlation. Nucleic Acids Research, 40(17), e133. "
              "doi:10.1093/nar/gks461",
    "pls": "Wold, S., Sjöström, M. & Eriksson, L. (2001). PLS-regression: a basic tool of "
           "chemometrics. Chemom. Intell. Lab. Syst., 58(2), 109–130. doi:10.1016/S0169-7439(01)00155-1",
    "coloc": "Ovchinnikova, K. et al. (2020). ColocML: machine learning quantifies co-localization "
             "between mass spectrometry images. Bioinformatics, 36(10), 3215–3224. doi:10.1093/bioinformatics/btaa085",
    "isotope": "Senko, M.W., Beu, S.C. & McLafferty, F.W. (1995). Determination of monoisotopic "
               "masses and ion populations for large biomolecules from resolved isotopic distributions. "
               "J. Am. Soc. Mass Spectrom., 6(4), 229–233. doi:10.1016/1044-0305(95)00017-8",
    "target_decoy": "Elias, J.E. & Gygi, S.P. (2007). Target-decoy search strategy for increased "
                    "confidence in large-scale protein identifications by mass spectrometry. Nat. "
                    "Methods, 4(3), 207–214. doi:10.1038/nmeth1019",
    "masses": "Wang, M., Huang, W.J., Kondev, F.G., Audi, G. & Naimi, S. (2021). The AME 2020 atomic "
              "mass evaluation (II). Chin. Phys. C, 45(3), 030003. doi:10.1088/1674-1137/abddaf",
    "numpy": "Harris, C.R. et al. (2020). Array programming with NumPy. Nature, 585(7825), "
             "357–362. doi:10.1038/s41586-020-2649-2",
    "scipy": "Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing "
             "in Python. Nat. Methods, 17(3), 261–272. doi:10.1038/s41592-019-0686-2",
    "sklearn": "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. J. Mach. "
               "Learn. Res., 12, 2825–2830.",
    "matplotlib": "Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. Comput. Sci. Eng., "
                  "9(3), 90–95. doi:10.1109/MCSE.2007.55",
}


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_fingerprint(path: str, full_max_mb: int = 64) -> dict:
    """Checksum a file. Small files get a full SHA-256; large files (e.g. a multi-GB
    .ibd) get a fast fingerprint (SHA-256 of head+tail+size) so traceability does not
    require reading gigabytes."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    if size <= full_max_mb * 1024 * 1024:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {"name": os.path.basename(path), "bytes": size, "sha256": h.hexdigest(),
                "method": "sha256"}
    with open(path, "rb") as f:
        head = f.read(1 << 20)
        f.seek(-(1 << 20), os.SEEK_END)
        tail = f.read(1 << 20)
    h.update(head); h.update(tail); h.update(str(size).encode())
    return {"name": os.path.basename(path), "bytes": size, "sha256": h.hexdigest(),
            "method": "sha256(head+tail+size)"}


def _versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        out["smile_msi"] = version("smile-msi")
    except Exception:  # noqa: BLE001
        from . import __version__
        out["smile_msi"] = __version__
    for pkg in _PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------- #
# ROI / region descriptors — a compact, *stable* record of which pixels fed an
# analysis. A region's name alone is unstable (renamable, mutable), so we also
# stamp a content hash of its pixel mask + how it was made, so the audit trail
# survives a later rename or edit (a changed mask gives a different hash).
# --------------------------------------------------------------------------- #
def mask_fingerprint(mask) -> str:
    """An 8-char SHA-1 of a boolean pixel mask's contents — a stable region identity
    that detects later edits (a changed selection yields a different fingerprint)."""
    import numpy as np
    a = np.asarray(mask, dtype=bool)
    return hashlib.sha1(a.tobytes()).hexdigest()[:8]


def region_descriptor(name, *, created_via="", mask=None, n_pixels=None,
                      bbox=None, role="") -> dict:
    """A compact, JSON-safe descriptor of one ROI/region that fed an analysis:
    its name, how it was made (drawn ROI / segmentation clusters / dendrogram branch /
    lasso), its pixel count, a content fingerprint of the mask, and (when known) its
    display bounding box. ``role`` tags the part it played (e.g. ``"A"``/``"B"``)."""
    d = {"name": str(name)}
    if role:
        d["role"] = str(role)
    if created_via:
        d["via"] = str(created_via)
    if mask is not None:
        import numpy as np
        a = np.asarray(mask, dtype=bool)
        d["n_pixels"] = int(a.sum())
        d["mask_sha1"] = mask_fingerprint(a)
    elif n_pixels is not None:
        d["n_pixels"] = int(n_pixels)
    if bbox is not None:
        d["bbox"] = [int(v) for v in bbox]
    return d


# --------------------------------------------------------------------------- #
# Compact one-line renderers — used for the leading ``# …`` audit header written
# at the top of every exported CSV/TSV and for the Report tab's source block.
# --------------------------------------------------------------------------- #
def format_settings(params: dict) -> str:
    """Render an analysis parameter dict as a compact ``k=v k2=v2`` string (skips the
    ``regions`` key, which is rendered separately, and empty/None values)."""
    bits = []
    for k, v in (params or {}).items():
        if k == "regions" or v is None or v == "":
            continue
        if isinstance(v, float):
            v = f"{v:g}"
        elif isinstance(v, dict):
            v = "{" + ", ".join(f"{a}={b}" for a, b in v.items()) + "}"
        bits.append(f"{k}={v}")
    return " ".join(bits)


def format_regions(descriptors) -> str:
    """Render a list of :func:`region_descriptor` dicts as a compact one-liner, e.g.
    ``A=nerve(3,204px,drawn ROI)  B=background(5,991px,clusters 2+5)``."""
    parts = []
    for r in (descriptors or []):
        if not isinstance(r, dict):
            parts.append(str(r))
            continue
        role = r.get("role")
        extra = []
        if r.get("n_pixels") is not None:
            extra.append(f"{int(r['n_pixels']):,}px")
        if r.get("via"):
            extra.append(str(r["via"]))
        s = (f"{role}=" if role else "") + str(r.get("name", "?"))
        if extra:
            s += "(" + ", ".join(extra) + ")"
        parts.append(s)
    return "  ".join(parts)


@dataclass
class Provenance:
    title: str = "SMILE MSI analysis"
    started: str = field(default_factory=_utc_now)
    inputs: list = field(default_factory=list)
    dataset: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    environment: dict = field(default_factory=_versions)

    # ----- recording ------------------------------------------------------- #
    def set_input(self, path: str, also: list | None = None):
        """Fingerprint the imzML (and, by default, its sibling .ibd)."""
        self.inputs.append(file_fingerprint(path))
        ibd = os.path.splitext(path)[0] + ".ibd"
        for extra in (also or ([ibd] if os.path.exists(ibd) else [])):
            if os.path.exists(extra):
                self.inputs.append(file_fingerprint(extra))
        return self

    def set_dataset(self, ds, mode: str = ""):
        lo, hi = ds.mz_range
        self.dataset = {
            "pixels": int(ds.n_pixels), "width": int(ds.width), "height": int(ds.height),
            "mz_min": round(float(lo), 4), "mz_max": round(float(hi), 4),
            "polarity": ds.polarity or mode, "mode": mode or ds.polarity,
            "spec_mode": ds.spec_mode, "source": ds.source,
            "pixel_size_um": getattr(ds, "pixel_size_um", None),
        }
        return self

    def step(self, name: str, now: str | None = None, **params):
        self.steps.append({"step": name, "time": now or _utc_now(),
                           "params": {k: _clean(v) for k, v in params.items()}})
        return self

    def output(self, path: str, description: str = ""):
        self.outputs.append({"file": os.path.basename(path), "description": description})
        return self

    # ----- serialization --------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"title": self.title, "started": self.started, "inputs": self.inputs,
                "dataset": self.dataset, "steps": self.steps, "outputs": self.outputs,
                "environment": self.environment}

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        """Rebuild a Provenance from :meth:`to_dict` output — restores the full audit
        trail (inputs, dataset, steps, outputs, environment) when a session is reopened,
        so the methods record is not lost on close. Tolerant of missing keys."""
        d = d or {}
        p = cls(title=d.get("title", "SMILE MSI analysis"),
                started=d.get("started") or _utc_now())
        p.inputs = list(d.get("inputs") or [])
        p.dataset = dict(d.get("dataset") or {})
        p.steps = list(d.get("steps") or [])
        p.outputs = list(d.get("outputs") or [])
        if d.get("environment"):
            p.environment = dict(d["environment"])
        return p

    def csv_header_lines(self, *, analysis: str = "", settings=None, regions=None,
                         now: str | None = None, brand: str = "SMILE MSI") -> list:
        """The leading ``# …`` audit block written at the top of an exported CSV/TSV:
        when it was made, the dataset (+ checksum, geometry, polarity), the analysis and
        every setting that produced the table, the ROIs that fed it, and the software
        stack. ``settings``/``regions`` accept either a pre-rendered string or the raw
        ``params`` dict / list of :func:`region_descriptor` dicts."""
        d = self.dataset or {}
        L = [f"{brand} — audit trail / data provenance",
             f"generated: {now or _utc_now()}"]
        if self.inputs:
            i0 = self.inputs[0]
            ds_line = f"dataset: {i0.get('name', '?')} (sha256 {str(i0.get('sha256', ''))[:12]}…)"
        elif d.get("source"):
            ds_line = f"dataset: {os.path.basename(str(d['source']))}"
        else:
            ds_line = "dataset:"
        if d:
            ds_line += (f"  {int(d.get('pixels', 0)):,} px · m/z {d.get('mz_min', '?')}–"
                        f"{d.get('mz_max', '?')} · {d.get('polarity') or d.get('mode') or '?'}")
        L.append(ds_line)
        if analysis:
            L.append(f"analysis: {analysis}")
        if settings:
            L.append("settings: " + (settings if isinstance(settings, str)
                                     else format_settings(settings)))
        if regions:
            L.append("ROIs: " + (regions if isinstance(regions, str)
                                 else format_regions(regions)))
        env = self.environment or {}
        libs = "; ".join(f"{k} {env[k]}"
                         for k in ("smile_msi", "numpy", "scipy", "scikit-learn") if k in env)
        if libs:
            L.append("software: " + libs)
        return ["# " + s for s in L]

    def to_json(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def to_markdown(self, path: str | None = None) -> str:
        L = [f"# {self.title} — Provenance & methods", "", f"_Generated {self.started}_", ""]
        if self.inputs:
            L += ["## Input files", "", "| file | bytes | checksum |", "|---|---|---|"]
            for i in self.inputs:
                L.append(f"| {i['name']} | {i['bytes']:,} | `{i['sha256'][:16]}…` ({i['method']}) |")
            L.append("")
        if self.dataset:
            d = self.dataset
            L += ["## Dataset", "",
                  f"- {d['pixels']:,} pixels ({d['width']}×{d['height']}), m/z "
                  f"{d['mz_min']}–{d['mz_max']}, {d.get('polarity') or '?'} mode", ""]
        if self.steps:
            L += ["## Processing steps", ""]
            for k, s in enumerate(self.steps, 1):
                params = dict(s["params"])
                rois = params.pop("regions", None)          # render ROIs compactly, not as raw dicts
                p = ", ".join(f"{a}={b}" for a, b in params.items())
                line = f"{k}. **{s['step']}**" + (f" — {p}" if p else "")
                if rois:
                    line += f" · ROIs: {format_regions(rois)}"
                L.append(line)
            L.append("")
        if self.outputs:
            L += ["## Outputs", ""] + [f"- `{o['file']}`" + (f" — {o['description']}" if o['description'] else "")
                                       for o in self.outputs] + [""]
        L += ["## Software", "", "| package | version |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in self.environment.items()]
        L += ["", "## Methods (draft)", "", self.methods_paragraph()]
        refs = self.references_used()
        if refs:
            L += ["", "## References", ""] + [f"{i}. {r}" for i, r in enumerate(refs, 1)]
        text = "\n".join(L)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    # ----- reporting checklist --------------------------------------------- #
    REQUIRED = "⟨required from user⟩"

    def miamsie_checklist(self) -> dict:
        """A MIAMSIE / MSIcheck minimum-information reporting checklist for an MSI
        experiment. Auto-fills every field the app actually observes (geometry, m/z range,
        polarity, centroid/profile mode, pixel size, processing steps + parameters,
        software versions, input checksums) and flags the acquisition / sample-prep fields
        it CANNOT know (matrix, instrument, laser diameter, …) as 'required from user'.

        Turns a custom tool's usual liability ('how did you do it?') into a strength: most
        of the checklist is filled automatically and the gaps are made explicit. Pair each
        annotation with its MSI confidence level (:func:`annotate.msi_confidence_level`,
        Schymanski et al. 2014). Refs: MIAMSIE / MSIcheck MSI reporting standards."""
        d = self.dataset or {}
        R = self.REQUIRED
        px = d.get("pixel_size_um")
        steps = [f"{s['step']}(" + ", ".join(f"{a}={b}" for a, b in s["params"].items()) + ")"
                 for s in self.steps] or [R]
        sw = "; ".join(f"{k} {v}" for k, v in self.environment.items())
        checksums = [f"{i['name']}: {i['sha256'][:16]}…" for i in self.inputs] or [R]
        return {
            "Sample & preparation": {
                "Organism / tissue / condition": R,
                "Sample preparation & storage": R,
                "Matrix & application method": R,
                "Section thickness": R,
            },
            "Instrument & acquisition": {
                "Instrument / mass analyzer": R,
                "Ionization (e.g. MALDI)": R,
                "Polarity": d.get("polarity") or R,
                "m/z range": (f"{d.get('mz_min', '?')}–{d.get('mz_max', '?')}"
                              if d else R),
                "Laser / spot size": R,
                "Spatial resolution (pixel size)": (f"{px} µm" if px else R),
                "Mass resolution": R,
                "Mass calibration / lock mass": R,
            },
            "Data & preprocessing": {
                "Spectrum type": d.get("spec_mode") or R,
                "Pixels / grid": (f"{d.get('pixels', '?')} ({d.get('width', '?')}×"
                                  f"{d.get('height', '?')})" if d else R),
                "Processing steps & parameters": steps,
            },
            "Annotation & confidence": {
                "Annotation method": "in-silico lipid database + isotope/adduct corroboration"
                                     " + MS/MS (where available)",
                "FDR control": "target–decoy MSM FDR (METASPACE-style); report q-value per ID",
                "Identification confidence": "MSI levels 1–5 (Schymanski et al. 2014)",
            },
            "Reproducibility": {
                "Software & versions": sw or R,
                "Input file checksums (SHA-256)": checksums,
                "Provenance record": "attached (JSON + auto-drafted Methods paragraph)",
            },
        }

    def miamsie_markdown(self, path: str | None = None) -> str:
        """Render :meth:`miamsie_checklist` as a fill-in Markdown checklist."""
        L = [f"# {self.title} — MSI reporting checklist (MIAMSIE / MSIcheck)", ""]
        for section, fields in self.miamsie_checklist().items():
            L.append(f"## {section}")
            for k, v in fields.items():
                if isinstance(v, list):
                    L.append(f"- **{k}:**")
                    L += [f"    - {item}" for item in v]
                else:
                    mark = " ⚠️" if v == self.REQUIRED else ""
                    L.append(f"- **{k}:** {v}{mark}")
            L.append("")
        text = "\n".join(L)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def methods_paragraph(self) -> str:
        """Auto-draft a Methods paragraph from the recorded steps, with inline
        author–year citations for each method (full list: :meth:`references_used`)."""
        d = self.dataset
        by = {s["step"]: s["params"] for s in self.steps}
        sent = []
        if d:
            sent.append(f"MALDI mass spectrometry imaging data ({d['pixels']:,} pixels, "
                        f"{d['width']}×{d['height']}; m/z {d['mz_min']}–{d['mz_max']}; "
                        f"{d.get('polarity') or 'unspecified'} ion mode) were loaded from imzML.")
        pp = by.get("preprocess", {})
        if pp:
            bits = []
            if pp.get("baseline"):
                bits.append("baseline-corrected (SNIP; Ryan et al. 1988; Morháč & Matoušek 2008)")
            if pp.get("smooth"):
                bits.append("smoothed (Savitzky–Golay 1964)")
            if pp.get("recalibrate"):
                bits.append("m/z-recalibrated to reference masses")
            if bits:
                sent.append("Spectra were " + ", ".join(bits) + ".")
        norm = by.get("normalize", {}).get("method") or by.get("peak_picking", {}).get("norm")
        if norm and norm != "none":
            sent.append(f"Intensities were {norm.upper()}-normalized per pixel (Deininger et al. 2011).")
        pk = by.get("peak_picking", {})
        if pk:
            sent.append(f"Peaks were detected on the mean spectrum (S/N ≥ {pk.get('snr', '?')}; "
                        "SciPy find_peaks, Virtanen et al. 2020) with noise estimated by the median "
                        "absolute deviation (Rousseeuw & Croux 1993).")
        sf = by.get("spatial_feature_finding", {})
        if sf:
            sent.append(f"Spatially-structured features were retained (Moran's I ≥ "
                        f"{sf.get('min_morans')}; Moran 1950).")
        seg = by.get("segmentation", {})
        if seg:
            n = seg.get("n_clusters", "?")
            method = seg.get("method")
            if method in ("agglomerative", "ward", "bisecting"):
                algo = ("two-stage bisecting k-means clustering (MiniBatch k-means "
                        "micro-clustering, Sculley 2010; divisive bisecting k-means, "
                        "Steinbach et al. 2000)") if method == "bisecting" else \
                       ("two-stage agglomerative clustering (MiniBatch k-means, Sculley "
                        "2010; Ward linkage, Ward 1963)")
                if seg.get("metric") == "correlation":   # no PCA: cluster spectral shape
                    basis = (f"the correlation distance (1 − Pearson r) between pixel spectra "
                             f"using {algo}")
                else:
                    basis = f"PCA (Pearson 1901; Hotelling 1933) followed by {algo}"
                scope = " within a selected region" if seg.get("scoped") else ""
                sent.append(f"The tissue was segmented{scope} into {n} regions by {basis}, "
                            "cut at a chosen detail level.")
            else:
                aware = "spatially-aware " if seg.get("spatial") else ""
                kdesc = "k chosen by silhouette (Rousseeuw 1987)" if seg.get("auto") \
                    else f"k = {seg.get('k', n)}"
                cite = "; Alexandrov & Kobarg 2011" if seg.get("spatial") else ""
                sent.append(f"The tissue was segmented into {n} regions by {aware}PCA (Pearson 1901; "
                            f"Hotelling 1933) + k-means (Lloyd 1982{cite}) ({kdesc}).")
        st = by.get("statistics") or by.get("roi_comparison")   # GUI records the latter
        if st:
            sent.append("Discriminating ions between regions were identified by rank-based ROC AUC "
                        "(equivalent to the Mann–Whitney U statistic; Mann & Whitney 1947; Hanley & "
                        "McNeil 1982) with Benjamini–Hochberg FDR control (Benjamini & Hochberg 1995).")
        if by.get("multigroup"):
            sent.append("Ions differing across multiple regions were identified per feature by the "
                        "Kruskal–Wallis H-test (Kruskal & Wallis 1952) with Benjamini–Hochberg FDR "
                        "control (Benjamini & Hochberg 1995).")
        if by.get("region_membership"):
            sent.append("Region-distinctive and shared ions were determined from per-region "
                        "detection prevalence (Venn compartments).")
        cc = by.get("cohort_comparison")
        if cc:
            sent.append(f"Across slides, {cc.get('group_a', 'group A')} and "
                        f"{cc.get('group_b', 'group B')} were compared at the sample level "
                        "(each replicate summarized to one mean profile), avoiding pixel-level "
                        "pseudoreplication, with Benjamini–Hochberg FDR control (Benjamini & "
                        "Hochberg 1995).")
        nc = by.get("cohort_nested_comparison")
        if nc:
            comps = nc.get("compartments") or []
            comp_txt = (f" across {len(comps)} compartments ({', '.join(str(c) for c in comps)})"
                        if comps else "")
            unit = nc.get("subject_by") or "subject"
            summ = nc.get("summary")
            summ_txt = {"median": "the median", "mean": "the mean"}.get(
                summ, "the saved per-region mean")
            sent.append(
                f"For the nested design{comp_txt}, each region's pixels were summarized to one "
                f"profile per {unit} per compartment using {summ_txt} of the normalized "
                f"per-pixel intensities (pseudobulk; Bemis et al. 2016), so the {unit} — not "
                f"the pixel — is the unit of replication.")
            if nc.get("model") == "lmm":
                dfm = ("between-within denominator degrees of freedom"
                       if nc.get("df_method") == "bw" else "a normal (Wald z) reference")
                sent.append(
                    f"Each feature was then fitted with a linear mixed model, "
                    f"intensity ~ group * compartment + (1 | {unit}) (Laird & Ware 1982), whose "
                    f"random intercept accounts for the correlation among a {unit}'s "
                    f"compartments; the group × compartment interaction, the group effect "
                    f"averaged over compartments, and the group's simple effect within each "
                    f"compartment were tested by Wald contrasts on {dfm}.")
            else:
                sent.append(
                    f"Each compartment was then tested separately for a "
                    f"{nc.get('group_a', 'group A')}-versus-{nc.get('group_b', 'group B')} "
                    f"difference by {nc.get('test', 'a two-group test')}.")
            if str(nc.get("test", "")).startswith("moderated"):
                sent.append(
                    "The moderated t-statistic shrinks each feature's residual variance toward "
                    "a prior fitted across all features by empirical Bayes (Smyth 2004), which "
                    "is what makes inference tractable at this number of replicates.")
            sent.append("Each contrast was Benjamini–Hochberg FDR-corrected across features "
                        "within its own family (Benjamini & Hochberg 1995).")

        bc = by.get("batch_correction")
        if bc:
            prot = bc.get("protected") or []
            prot_txt = (f", with the {', '.join(str(p) for p in prot)} contrast protected via a "
                        "covariate model matrix (Johnson et al. 2007)") if prot else ""
            if bc.get("applied"):
                sent.append("Cross-batch technical variation was removed by empirical-Bayes ComBat "
                            f"adjustment (Johnson et al. 2007){prot_txt}; residual batch effect was "
                            "assessed by a kBET-like neighbourhood mixing test (Büttner et al. 2019) and "
                            "silhouette-by-batch versus -by-biology (Rousseeuw 1987).")
            else:
                # Diagnostic-only run: the ComBat-corrected matrix was NOT applied to the reported
                # comparison (see gui/cohortview.py), so we must not claim the variation was "removed".
                sent.append("Cross-batch technical variation was assessed as a diagnostic — not applied "
                            "to the reported comparison — using empirical-Bayes ComBat "
                            f"(Johnson et al. 2007){prot_txt}: batch mixing was quantified by a kBET-like "
                            "neighbourhood test (Büttner et al. 2019) and silhouette-by-batch versus "
                            "-by-biology (Rousseeuw 1987).")
        if by.get("quantification"):
            sent.append("Absolute concentrations were estimated from on-tissue calibration curves "
                        "fitted on concentration-defined standard regions (with LOD/LOQ reporting and "
                        "out-of-range pixels flagged), i.e. quantitative MSI.")
        if by.get("spectral_match"):
            sent.append("MS/MS identities were corroborated against a spectral library by entropy and "
                        "(modified) cosine similarity (Li et al. 2021; Stein & Scott 1994).")
        if by.get("registration"):
            sent.append("The ion image was co-registered to the optical/histology image by a "
                        "landmark-estimated transform (Umeyama 1991; scikit-image, van der Walt et al. 2014).")
        if by.get("imzml_export"):
            sent.append("Data were exported as imzML with reporting metadata validated against minimum-"
                        "reporting guidelines (Schramm et al. 2012; McDonnell et al. 2015) for deposition / "
                        "METASPACE annotation (Palmer et al. 2017).")
        if by.get("single_cell"):
            sent.append("Single-cell metabolite profiles were obtained by segmenting cells on the "
                        "co-registered microscopy image (watershed; Beucher & Meyer 1993) and area-"
                        "weighting MSI pixels onto cells (the SpaceM approach; Rappez et al. 2021).")
        if by.get("volume3d"):
            sent.append("Serial sections were registered (FFT phase correlation; Reddy & Chatterjee "
                        "1996) and stacked into a 3-D molecular volume (volume rendering; Levoy 1988).")
        if by.get("cross_modality_correlation"):
            sent.append("The MSI data were co-registered to a second spatial modality and analysed "
                        "jointly by per-feature spatial correlation (Spearman 1904; spatial multi-omics, "
                        "npj Imaging 2024).")
        if by.get("ion_embedding"):
            sent.append("Ion images were embedded by a self-supervised contrastive encoder "
                        "(DeepION-style; Wang et al. 2024; SimCLR, Chen et al. 2020), and "
                        "colocalization was measured in the learned embedding space.")
        cls = by.get("classification")
        if cls:
            sent.append(f"Regions were classified by {cls.get('method', 'PLS-DA')} "
                        "(Wold et al. 2001) with variable-importance-in-projection (VIP) feature "
                        "ranking; generalization was assessed by leave-one-sample-out "
                        "cross-validation to avoid pixel-level leakage.")
        an = by.get("annotation", {})
        if an:
            sent.append(f"Lipids were annotated in silico within {an.get('match_ppm', '?')} ppm "
                        "(sum-composition level) using monoisotopic masses (Wang et al. 2021), with "
                        "isotope-pattern (Senko et al. 1995) and adduct corroboration; annotation "
                        "confidence was estimated by a target–decoy approach (Elias & Gygi 2007).")
        env = self.environment
        ver = f"SMILE MSI {env.get('smile_msi', '')}".strip()
        libs = ", ".join(f"{k} {env[k]}" for k in ("numpy", "scipy", "scikit-learn") if k in env)
        sent.append(f"Analyses were performed with {ver} ({libs}; Harris et al. 2020; Virtanen et al. "
                    "2020; Pedregosa et al. 2011).")
        return " ".join(sent)

    def _used_ref_tags(self) -> list:
        """Citation tags for exactly the methods the recorded steps used, in report
        order — keeps :meth:`references_used` in lock-step with the methods paragraph."""
        by = {s["step"]: s["params"] for s in self.steps}
        tags = []
        pp = by.get("preprocess", {})
        if pp.get("baseline"):
            tags += ["snip", "snip_lls"]
        if pp.get("smooth"):
            tags.append("savgol")
        norm = by.get("normalize", {}).get("method") or by.get("peak_picking", {}).get("norm")
        if norm and norm != "none":
            tags.append("tic")
        if by.get("peak_picking"):
            tags += ["scipy", "mad"]
        if by.get("spatial_feature_finding"):
            tags.append("morans")
        seg = by.get("segmentation", {})
        if seg:
            method = seg.get("method")
            if seg.get("metric") != "correlation":       # correlation path skips PCA
                tags += ["pca_pearson", "pca_hotelling"]
            if method in ("agglomerative", "ward", "bisecting"):
                tags += ["minibatch", "bisecting" if method == "bisecting" else "ward"]
            else:
                tags += ["kmeans", "silhouette"]
                if seg.get("spatial"):
                    tags.append("spatial_seg")
        if by.get("statistics") or by.get("roi_comparison"):
            tags += ["mwu", "auc", "bh_fdr"]
        if by.get("multigroup"):
            tags += ["kruskal", "bh_fdr"]
        if by.get("cohort_comparison"):
            tags += ["mwu", "bh_fdr"]
        nc = by.get("cohort_nested_comparison")
        if nc:
            tags += ["bh_fdr", "meanstest"]
            tags += ["lmm"] if nc.get("model") == "lmm" else []
            tags += ["ebayes"] if nc.get("test", "").startswith("moderated") else []
        if by.get("batch_correction"):
            # covariate protection is ComBat's own model matrix (Johnson 2007), not
            # reComBat — so the generated methods text cites ComBat, not Adler 2022.
            tags += ["combat", "kbet", "silhouette"]
        if by.get("quantification"):
            tags += ["qmsi"]
        if by.get("spectral_match"):
            tags += ["entropy_sim", "spectral_cosine"]
        if by.get("registration"):
            tags += ["umeyama", "skimage"]
        if by.get("imzml_export"):
            tags += ["imzml", "miamsie", "metaspace"]
        if by.get("single_cell"):
            tags += ["watershed", "spacem"]
        if by.get("volume3d"):
            tags += ["vol_reg", "vol_render"]
        if by.get("cross_modality_correlation"):
            tags += ["spearman", "comap_multiomics"]
        if by.get("ion_embedding"):
            tags += ["deepion", "simclr"]
        if by.get("classification"):
            tags += ["pls"]
        if by.get("annotation"):
            tags += ["masses", "isotope", "target_decoy"]
        tags += ["numpy", "scipy", "sklearn", "matplotlib"]   # software stack, always
        seen, out = set(), []
        for t in tags:
            if t not in seen and t in REFERENCES:
                seen.add(t)
                out.append(t)
        return out

    def references_used(self) -> list:
        """Full citations (strings) for the methods this analysis actually ran — drop
        straight into a manuscript's reference list. See METHODS.md for the catalog."""
        return [REFERENCES[t] for t in self._used_ref_tags()]


def _clean(v):
    """JSON-friendly parameter value."""
    if isinstance(v, (range,)):
        return f"{v.start}..{v.stop - 1}"
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    return v
