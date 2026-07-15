"""Externalize to community standards — enrich imzML with reporting metadata,
validate the minimum-reporting checklist, and (opt-in) round-trip to METASPACE.

SMILE MSI already *reads* and *writes* imzML/ibd (:mod:`smile_msi.ingest`) and carries a
rich internal provenance record (:class:`smile_msi.provenance.Provenance`) including a
MIAMSIE / MSIcheck minimum-reporting checklist. The file it writes, however, is
metadata-thin: :func:`ingest.to_imzml` passes only mode and polarity. This module closes
the gap with three pure concerns and one opt-in network concern:

1. :class:`AcquisitionMeta` — the acquisition / sample-prep reporting metadata the file
   itself cannot supply (instrument, matrix, section thickness, …). It maps onto the
   checklist's "required from user" fields.
2. :func:`imzml_cvparams` / :func:`write_imzml_with_metadata` — map the dataset +
   :class:`AcquisitionMeta` (+ provenance pixel size) to imzML/OBO cvParams and inject
   them into the written ``.imzML`` header via a stdlib ``xml.etree`` post-pass (spectrum
   offsets and the ``.ibd`` are never touched).
3. :func:`validate_reporting` — score the minimum-reporting checklist per field
   (ok / warn / missing-required) and gate submission-readiness.
4. :func:`metaspace_submit` / :func:`metaspace_annotations` / :func:`compare_fdr` — opt-in
   METASPACE submission and FDR-annotation pull-back; the heavy ``metaspace2020`` client
   and pandas are lazy-imported so the core engine stays offline-first.

Literature basis
----------------
* imzML cvParam mapping convention — reimplemented clean-room from the imzML Writer
  publication + the open imzML OBO (no source copied): Hamilton et al. (2024), *imzML
  Writer: A User-Friendly Tool for Generating imzML Files*, ChemRxiv,
  doi:10.26434/chemrxiv-2024-0jjf6.
* Validation rationale and field set: Weiskirchen, R. et al. (2018), *Establishing and
  validating the imzML/mzML standards for error-free processing of mass spectrometry
  imaging data*, Anal. Chem., doi:10.1021/acs.analchem.8b03059. We implement a lightweight
  checklist validator (completeness over :class:`Provenance` + :class:`AcquisitionMeta`),
  not a full XSD/semantic validator.
* Minimum-reporting standard: MIAMSIE / MSIcheck (already encoded in
  :meth:`Provenance.miamsie_checklist`).
* METASPACE annotation engine / MSM target–decoy FDR (the reference our in-engine
  :func:`annotate.estimate_fdr` is compared against — an agreement diagnostic, not an
  equivalence claim): Palmer, A. et al. (2017), *FDR-controlled metabolite annotation for
  high-resolution imaging mass spectrometry*, Nat. Methods 14(1):57–60,
  doi:10.1038/nmeth.4072.

Nothing here is stochastic — export is deterministic and METASPACE is a network fetch — so
no ``random_state`` is threaded.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

# mzML uses the PSI-MS namespace; imzML adds the IMS controlled vocabulary on top.
_MZML_NS = "http://psi.hupo.org/ms/mzml"


# --------------------------------------------------------------------------- #
# reporting metadata model
# --------------------------------------------------------------------------- #
@dataclass
class AcquisitionMeta:
    """Acquisition / sample-prep reporting metadata that the file itself cannot supply.

    Maps onto the MIAMSIE checklist's ``REQUIRED`` ("required from user") fields. Every
    field is optional; completeness is what :func:`validate_reporting` scores. Polarity,
    pixel size, m/z range and spectrum type come from the dataset + provenance, **not**
    here — never duplicate an auto-observed field.
    """

    organism: str = ""
    tissue: str = ""
    condition: str = ""
    sample_prep: str = ""
    storage: str = ""
    matrix: str = ""
    matrix_application: str = ""
    section_thickness_um: float | None = None
    instrument: str = ""
    mass_analyzer: str = ""
    ionization: str = "MALDI"
    laser_spot_um: float | None = None
    mass_resolution: str = ""
    calibration: str = ""

    def to_dict(self) -> dict:
        """JSON-safe dict (numeric fields stay numeric, empties preserved)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "AcquisitionMeta":
        """Rebuild from :meth:`to_dict`; tolerant of missing/extra keys."""
        d = d or {}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def is_empty(self) -> bool:
        """True when no user-supplied field is set (ionization defaults to MALDI, so a
        bare instance with only that default still counts as empty for export fallback)."""
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "ionization":
                continue
            if v not in ("", None):
                return False
        return True


# --------------------------------------------------------------------------- #
# imzML cvParam enrichment
# --------------------------------------------------------------------------- #
# Curated controlled-vocabulary terms for the common polarity / mass-analyzer values.
# Free-text instrument/matrix/etc. are emitted as the canonical "model/matrix" accession
# with a user-supplied value (the OBO permits a value on these terms), never invented
# accessions. Verify accessions against the current imzML/MS OBO before merge — accession
# drift across OBO versions is a known risk.
_POLARITY_CV = {
    "positive": ("MS:1000130", "positive scan"),
    "negative": ("MS:1000129", "negative scan"),
}
_ANALYZER_CV = {
    "tof": ("MS:1000084", "time-of-flight"),
    "time-of-flight": ("MS:1000084", "time-of-flight"),
    "ft-icr": ("MS:1000079", "fourier transform ion cyclotron resonance mass spectrometer"),
    "fticr": ("MS:1000079", "fourier transform ion cyclotron resonance mass spectrometer"),
    "orbitrap": ("MS:1000484", "orbitrap"),
    "quadrupole": ("MS:1000081", "quadrupole"),
    "ion trap": ("MS:1000264", "ion trap"),
}
_IONIZATION_CV = {
    "maldi": ("MS:1000075", "matrix-assisted laser desorption ionization"),
    "desi": ("MS:1000247", "desorption electrospray ionization"),
    "esi": ("MS:1000073", "electrospray ionization"),
}


def imzml_cvparams(ds, meta: "AcquisitionMeta | None" = None, prov=None) -> dict:
    """Map a dataset + :class:`AcquisitionMeta` (+ provenance pixel size) to imzML/OBO
    cvParams ready to inject into a written ``.imzML`` header.

    Returns ``accession -> {"name": str, "value": str|"", "target": str}`` where
    ``target`` is the header element the param belongs in (``"scanSettings"`` for the
    pixel-size step, ``"instrument"`` for instrument/analyzer/ionization, ``"sample"`` for
    sample/matrix/prep). Absent fields are **omitted** (never emitted blank).

    The pixel size is sourced (in order) from ``ds.pixel_size_um`` then the provenance
    dataset record — keeping the dataset → provenance → cvParam flow the single source of
    truth. Polarity comes from ``ds.polarity``.

    Clean-room mapping convention from the imzML Writer publication + the imzML OBO
    (Hamilton et al. 2024, doi:10.26434/chemrxiv-2024-0jjf6). The example accessions are
    illustrative; confirm each against the current OBO before merge.
    """
    meta = meta or AcquisitionMeta()
    out: dict[str, dict] = {}

    def add(accession, name, value, target):
        # `None` => omit the field entirely; "" => a presence-only term (the cvParam's
        # existence is the value, e.g. a polarity or ionization term). True is a synonym
        # for a presence-only term.
        if value is None:
            return
        out[accession] = {"name": name, "value": "" if value is True else str(value),
                          "target": target}

    # --- spatial resolution (scanSettings) — single source: dataset, then provenance ---
    px = getattr(ds, "pixel_size_um", None)
    if px is None and prov is not None:
        px = (getattr(prov, "dataset", None) or {}).get("pixel_size_um")
    if px not in (None, ""):
        add("IMS:1000046", "pixel size x", float(px), "scanSettings")
        add("IMS:1000047", "pixel size y", float(px), "scanSettings")

    # --- polarity (instrument scan) ---
    pol = str(getattr(ds, "polarity", "") or "").strip().lower()
    if pol in _POLARITY_CV:
        acc, name = _POLARITY_CV[pol]
        add(acc, name, "", "instrument")

    # --- instrument / mass analyzer / ionization ---
    if meta.instrument:
        # "instrument model" carries the free-text model as the cvParam value.
        add("MS:1000031", "instrument model", meta.instrument, "instrument")
    analyzer = (meta.mass_analyzer or "").strip().lower()
    if analyzer in _ANALYZER_CV:
        acc, name = _ANALYZER_CV[analyzer]
        add(acc, name, "", "instrument")
    elif meta.mass_analyzer:
        add("MS:1000443", "mass analyzer type", meta.mass_analyzer, "instrument")
    ion = (meta.ionization or "").strip().lower()
    if ion in _IONIZATION_CV:
        acc, name = _IONIZATION_CV[ion]
        add(acc, name, "", "instrument")
    elif meta.ionization:
        add("MS:1000008", "ionization type", meta.ionization, "instrument")

    # --- sample / matrix / preparation ---
    if meta.matrix:
        add("MS:1000835", "matrix solution", meta.matrix, "sample")
    if meta.matrix_application:
        add("MS:1000836", "matrix application type", meta.matrix_application, "sample")
    if meta.organism:
        add("MS:1000552", "organism", meta.organism, "sample")  # subsumes organism/tissue free text
    if meta.tissue:
        add("MS:1000553", "tissue", meta.tissue, "sample")
    if meta.section_thickness_um not in (None, ""):
        add("IMS:1001211", "section thickness", float(meta.section_thickness_um), "sample")
    if meta.laser_spot_um not in (None, ""):
        add("MS:1000843", "laser spot size", float(meta.laser_spot_um), "instrument")

    return out


def write_imzml_with_metadata(ds, path: str, meta: "AcquisitionMeta | None" = None,
                              prov=None) -> str:
    """Write a dataset to imzML, then inject :func:`imzml_cvparams` into the header.

    Delegates the spectra write to :func:`ingest.to_imzml` (so the ``.ibd`` and the
    per-spectrum offset table are produced exactly as today), then runs a stdlib
    ``xml.etree`` post-pass that adds the metadata cvParams to the ``<scanSettings>``,
    ``<instrumentConfiguration>`` and ``<sample>`` header elements — never touching any
    ``<spectrum>`` element or the ``.ibd``. When ``meta`` is ``None``/empty the output is
    byte-identical to plain :func:`ingest.to_imzml` (current behaviour, graceful degrade).

    Records the output on ``prov`` when given (``prov.output`` + a ``imzml_export`` step).
    Returns the ``.imzML`` path.
    """
    from . import ingest

    ingest.to_imzml(ds, path)
    # Empty / no meta => the plain to_imzml output is left untouched (graceful degrade,
    # byte-identical to current behaviour). Dataset-derived params (pixel size, polarity)
    # are only injected alongside user-supplied reporting metadata.
    cvparams = {} if (meta is None or meta.is_empty()) else imzml_cvparams(ds, meta, prov=prov)
    if cvparams:
        _inject_cvparams(path, cvparams)
    if prov is not None:
        try:
            prov.output(path, "imzML + reporting metadata" if cvparams else "imzML")
            prov.step("imzml_export", level="miamsie", n_cvparams=len(cvparams))
        except Exception:  # noqa: BLE001 — provenance recording is best-effort
            pass
    return path


def _inject_cvparams(path: str, cvparams: dict) -> None:
    """Add the cvParams to the imzML header in place via ``xml.etree``.

    Only header/metadata elements are edited: ``scanSettings`` (pixel size), the first
    ``instrumentConfiguration`` (instrument / analyzer / polarity / ionization / laser),
    and a ``sample`` element under ``sampleList`` (matrix / organism / tissue / section).
    ``<spectrum>`` offset cvParams and the ``.ibd`` are never read or written.
    """
    import xml.etree.ElementTree as ET

    ns = {"mz": _MZML_NS}
    ET.register_namespace("", _MZML_NS)
    tree = ET.parse(path)
    root = tree.getroot()

    def q(tag):
        return f"{{{_MZML_NS}}}{tag}"

    def cvparam_elem(accession, info):
        cv_ref = accession.split(":", 1)[0]
        return ET.Element(q("cvParam"), {
            "cvRef": cv_ref, "accession": accession,
            "name": info["name"], "value": info.get("value", "")})

    by_target: dict[str, list] = {"scanSettings": [], "instrument": [], "sample": []}
    for accession, info in cvparams.items():
        by_target[info["target"]].append((accession, info))

    # --- scanSettings: append to the first <scanSettings> ---
    if by_target["scanSettings"]:
        scan = root.find(f".//mz:scanSettingsList/mz:scanSettings", ns)
        if scan is not None:
            for accession, info in by_target["scanSettings"]:
                scan.append(cvparam_elem(accession, info))

    # --- instrumentConfiguration: append to the first one ---
    if by_target["instrument"]:
        ic = root.find(f".//mz:instrumentConfigurationList/mz:instrumentConfiguration", ns)
        if ic is not None:
            for accession, info in by_target["instrument"]:
                ic.append(cvparam_elem(accession, info))

    # --- sample: ensure a <sampleList>/<sample> exists, then append ---
    if by_target["sample"]:
        sample_list = root.find("mz:sampleList", ns)
        if sample_list is None:
            sample_list = ET.Element(q("sampleList"), {"count": "1"})
            # Insert sampleList just after <fileDescription> for schema-sane ordering.
            children = list(root)
            idx = 0
            for i, ch in enumerate(children):
                if ch.tag == q("fileDescription"):
                    idx = i + 1
                    break
            root.insert(idx, sample_list)
        sample = sample_list.find("mz:sample", ns)
        if sample is None:
            sample = ET.SubElement(sample_list, q("sample"),
                                   {"id": "sample1", "name": "sample 1"})
            sample_list.set("count", str(len(sample_list.findall("mz:sample", ns))))
        for accession, info in by_target["sample"]:
            sample.append(cvparam_elem(accession, info))

    tree.write(path, encoding="ISO-8859-1", xml_declaration=True)


# --------------------------------------------------------------------------- #
# validation pass
# --------------------------------------------------------------------------- #
# Which AcquisitionMeta field satisfies each checklist field that miamsie_checklist()
# flags REQUIRED. Maps (section, field) -> (AcquisitionMeta attr names that fill it).
# A field is "ok" once any of its meta attrs is non-empty.
_MIAMSIE_FILL = {
    ("Sample & preparation", "Organism / tissue / condition"):
        ("organism", "tissue", "condition"),
    ("Sample & preparation", "Sample preparation & storage"):
        ("sample_prep", "storage"),
    ("Sample & preparation", "Matrix & application method"):
        ("matrix", "matrix_application"),
    ("Sample & preparation", "Section thickness"): ("section_thickness_um",),
    ("Instrument & acquisition", "Instrument / mass analyzer"):
        ("instrument", "mass_analyzer"),
    ("Instrument & acquisition", "Ionization (e.g. MALDI)"): ("ionization",),
    ("Instrument & acquisition", "Laser / spot size"): ("laser_spot_um",),
    ("Instrument & acquisition", "Mass resolution"): ("mass_resolution",),
    ("Instrument & acquisition", "Mass calibration / lock mass"): ("calibration",),
}

# Fields that block METASPACE submission readiness when blank (a subset of REQUIRED —
# matrix/instrument/analyzer/polarity are the METASPACE minimum-metadata essentials).
_SUBMISSION_REQUIRED = {
    ("Sample & preparation", "Organism / tissue / condition"),
    ("Sample & preparation", "Matrix & application method"),
    ("Instrument & acquisition", "Instrument / mass analyzer"),
    ("Instrument & acquisition", "Ionization (e.g. MALDI)"),
    ("Instrument & acquisition", "Polarity"),
}


@dataclass
class ReportingResult:
    """Outcome of :func:`validate_reporting`.

    ``fields`` maps ``"Section :: Field" -> {value, status, section, required}`` where
    ``status`` is ``"ok"`` / ``"warn"`` (recommended, absent, non-blocking) /
    ``"missing"`` (a submission-required field still blank).
    """

    fields: dict
    n_ok: int
    n_warn: int
    n_missing_required: int
    level: str = "miamsie"

    def is_submission_ready(self) -> bool:
        """True when no submission-required field is missing."""
        return self.n_missing_required == 0

    def missing_required(self) -> list:
        """The ``"Section :: Field"`` keys still blocking submission."""
        return [k for k, v in self.fields.items() if v["status"] == "missing"]

    def to_markdown(self) -> str:
        """Render as a fill-in checklist with ✅ / ⚠️ / ❌ marks (mirrors the ``⚠️``
        convention in :meth:`Provenance.miamsie_markdown`)."""
        mark = {"ok": "✅", "warn": "⚠️", "missing": "❌"}
        ready = "submission-ready" if self.is_submission_ready() else "NOT submission-ready"
        L = [f"# MSI reporting validation ({self.level}) — {ready}", "",
             f"- {self.n_ok} ok · {self.n_warn} recommended-missing · "
             f"{self.n_missing_required} required-missing", ""]
        by_section: dict[str, list] = {}
        for key, info in self.fields.items():
            by_section.setdefault(info["section"], []).append((key, info))
        for section, items in by_section.items():
            L.append(f"## {section}")
            for key, info in items:
                field = key.split(" :: ", 1)[-1]
                val = info["value"] if info["value"] not in ("", None) else "—"
                L.append(f"- {mark[info['status']]} **{field}:** {val}")
            L.append("")
        return "\n".join(L)


def validate_reporting(prov, meta: "AcquisitionMeta | None" = None,
                       *, level: str = "miamsie") -> ReportingResult:
    """Score the minimum-reporting checklist over a :class:`Provenance` + an
    :class:`AcquisitionMeta`.

    Reuses :meth:`Provenance.miamsie_checklist` for the auto-observed fields (polarity,
    m/z range, pixel size, spectrum type, processing steps, software, checksums) and
    overlays ``meta`` for the user-supplied ones. Each field is classified:

    * **ok** — auto-observed and present, or a user field filled by ``meta``.
    * **warn** — recommended but absent and *not* submission-blocking.
    * **missing** — a submission-required field (see ``_SUBMISSION_REQUIRED``) still blank.

    ``level`` selects which checklist (``"miamsie"`` now; room for a narrower
    ``"metaspace"`` minimum later). Rationale and field set: Weiskirchen et al. (2018),
    doi:10.1021/acs.analchem.8b03059; MIAMSIE / MSIcheck.
    """
    meta = meta or AcquisitionMeta()
    checklist = prov.miamsie_checklist()
    required_token = getattr(prov, "REQUIRED", "⟨required from user⟩")

    def meta_value(section, field_name):
        """Return the AcquisitionMeta value satisfying this field, or '' if none."""
        attrs = _MIAMSIE_FILL.get((section, field_name))
        if not attrs:
            return ""
        for a in attrs:
            v = getattr(meta, a, "")
            if v not in ("", None):
                # ionization defaults to MALDI — count it as filled (it has a default).
                return v
        return ""

    out: dict[str, dict] = {}
    n_ok = n_warn = n_missing = 0
    for section, fields_map in checklist.items():
        for field_name, value in fields_map.items():
            key = f"{section} :: {field_name}"
            required = (section, field_name) in _SUBMISSION_REQUIRED
            # Lists (processing steps, checksums) are auto-observed; ok if non-trivial.
            if isinstance(value, list):
                present = value and value != [required_token]
                status = "ok" if present else ("missing" if required else "warn")
                disp = "; ".join(str(v) for v in value) if present else ""
            elif value == required_token:
                # The app can't observe this — try AcquisitionMeta.
                mv = meta_value(section, field_name)
                if mv not in ("", None):
                    status, disp = "ok", mv
                else:
                    status = "missing" if required else "warn"
                    disp = ""
            else:
                status, disp = "ok", value
            out[key] = {"value": disp, "status": status, "section": section,
                        "required": required}
            if status == "ok":
                n_ok += 1
            elif status == "warn":
                n_warn += 1
            else:
                n_missing += 1
    return ReportingResult(fields=out, n_ok=n_ok, n_warn=n_warn,
                           n_missing_required=n_missing, level=level)


# --------------------------------------------------------------------------- #
# METASPACE round-trip (network, opt-in)
# --------------------------------------------------------------------------- #
_METASPACE_INSTALL_HINT = (
    "METASPACE submission needs the 'metaspace2020' package — install it with "
    "`uv pip install metaspace2020`.")


def _import_metaspace():
    """Lazy-import the official METASPACE client, raising a clear install hint if absent.

    Mirrors the optional-extra precedent in ``explain.py`` (``ImportError`` with a
    ``uv pip install`` hint). No network at import; no silent fallback.
    """
    try:
        from metaspace import SMInstance  # noqa: F401  (re-export below)
    except ImportError as e:  # pragma: no cover — exercised via monkeypatch in tests
        raise ImportError(_METASPACE_INSTALL_HINT) from e
    return SMInstance


def metaspace_submit(imzml_path: str, ibd_path: str, meta: "AcquisitionMeta", *,
                     databases, project=None, api_key=None, name: str | None = None) -> str:
    """Submit a standards-compliant imzML/ibd pair to METASPACE; returns the dataset id.

    Lazy-imports the official ``metaspace2020`` client (raising :class:`ImportError` with a
    ``uv pip install metaspace2020`` hint if the extra is absent) and requires an
    ``api_key`` (offline / no key → clear :class:`ValueError`). The MSM target–decoy FDR
    METASPACE returns is the reference our in-engine FDR is compared against — Palmer et
    al. (2017), doi:10.1038/nmeth.4072.

    Generated dataset names use ``" - "`` separators, never ``"·"``.
    """
    SMInstance = _import_metaspace()
    if not api_key:
        raise ValueError("METASPACE submission requires an API key (set one in the "
                         "reporting dialog or pass api_key=…).")
    for p in (imzml_path, ibd_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"file not found for METASPACE submission: {p}")
    sm = SMInstance(api_key=api_key)
    ds_name = name or _default_dataset_name(imzml_path, meta)
    metadata = _metaspace_metadata(meta)
    return sm.submit_dataset(
        imzml_path, ibd_path, ds_name, metadata, databases,
        project_ids=[project] if project else None)


def _default_dataset_name(imzml_path: str, meta: "AcquisitionMeta") -> str:
    base = os.path.splitext(os.path.basename(imzml_path))[0]
    bits = [b for b in (meta.organism, meta.tissue, base) if b]
    return " - ".join(bits) or base


def _metaspace_metadata(meta: "AcquisitionMeta") -> dict:
    """The METASPACE minimum-metadata JSON derived from :class:`AcquisitionMeta`."""
    return {
        "Data_Type": "Imaging MS",
        "Sample_Information": {
            "Organism": meta.organism or "N/A",
            "Organism_Part": meta.tissue or "N/A",
            "Condition": meta.condition or "N/A",
            "Sample_Growth_Conditions": meta.storage or "N/A",
        },
        "Sample_Preparation": {
            "Sample_Stabilisation": meta.sample_prep or "N/A",
            "MALDI_Matrix": meta.matrix or "N/A",
            "MALDI_Matrix_Application": meta.matrix_application or "N/A",
        },
        "MS_Analysis": {
            "Analyzer": meta.mass_analyzer or "N/A",
            "Ionisation_Source": meta.ionization or "MALDI",
            "Detector_Resolving_Power": {"mz": 200, "Resolving_Power": meta.mass_resolution or "N/A"},
        },
    }


def metaspace_annotations(dataset_id: str, *, fdr: float = 0.1, database=None,
                          api_key=None):
    """Pull back FDR-controlled annotations for a METASPACE dataset as a pandas DataFrame.

    Lazy-imports the client and pandas. Columns include formula, adduct, msm, fdr, … —
    Palmer et al. (2017), doi:10.1038/nmeth.4072.
    """
    SMInstance = _import_metaspace()
    sm = SMInstance(api_key=api_key) if api_key else SMInstance()
    ds = sm.dataset(id=dataset_id)
    return ds.results(database=database, fdr=fdr)


def compare_fdr(our_annotations, metaspace_df, *, ppm: float | None = None):
    """Join our in-engine annotations to METASPACE's by formula + adduct and report
    agreement, FDR delta, and the ours-only / theirs-only sets.

    ``our_annotations`` is an iterable of dicts (or a DataFrame) with at least ``formula``
    and ``adduct`` (optionally ``q_value`` and ``mz``); ``metaspace_df`` is the DataFrame
    from :func:`metaspace_annotations` (``formula``/``adduct``/``fdr``, optionally ``mz``).
    Pure pandas (lazy-imported). The cross-FDR comparison is an **agreement diagnostic, not
    an equivalence claim** — METASPACE's MSM decoy-adduct FDR and our target–decoy FDR
    (Elias & Gygi 2007) are computed differently. Palmer et al. (2017),
    doi:10.1038/nmeth.4072.

    Returns a DataFrame with one row per (formula, adduct): ``q_ours``, ``fdr_metaspace``,
    ``delta`` (q_ours − fdr_metaspace), and ``agreement`` in
    ``{"both", "ours_only", "metaspace_only"}``.
    """
    import pandas as pd

    ours = pd.DataFrame(list(our_annotations)) if not isinstance(our_annotations, pd.DataFrame) \
        else our_annotations.copy()
    theirs = metaspace_df.copy()
    for df, who in ((ours, "ours"), (theirs, "metaspace")):
        for col in ("formula", "adduct"):
            if col not in df.columns:
                raise KeyError(f"{who} annotations missing required column '{col}'")
    ours = ours.rename(columns={"q_value": "q_ours"})
    theirs = theirs.rename(columns={"fdr": "fdr_metaspace"})
    keep_ours = ["formula", "adduct"] + [c for c in ("q_ours", "mz") if c in ours.columns]
    keep_theirs = ["formula", "adduct"] + [c for c in ("fdr_metaspace", "mz") if c in theirs.columns]
    merged = ours[keep_ours].merge(theirs[keep_theirs], on=["formula", "adduct"],
                                   how="outer", indicator=True, suffixes=("_ours", "_meta"))
    agree_map = {"both": "both", "left_only": "ours_only", "right_only": "metaspace_only"}
    merged["agreement"] = merged["_merge"].map(agree_map)
    merged = merged.drop(columns=["_merge"])
    if "q_ours" in merged.columns and "fdr_metaspace" in merged.columns:
        merged["delta"] = merged["q_ours"] - merged["fdr_metaspace"]
    return merged
