"""Audit trail / data provenance — one place every analysis records *how a result was
made*: the full set of settings that produced it and the exact ROIs/regions that fed it.

The heavy lifting lives in :mod:`smile_msi.provenance` (the serializable record + the
methods paragraph + the CSV header renderer). This mixin is the GUI-side glue: a single
:meth:`AuditMixin.record_step` that every analysis handler calls, which

* appends a fully-parameterized step to the per-sample ``self.prov`` (persisted in the
  session, so the methods record survives close/reopen), with the ROIs resolved to compact,
  rename-stable descriptors (name + how-made + pixel count + a content hash of the mask), and
* returns a ``source_extra`` dict (``method`` / ``params`` / ``regions``) ready to hand to
  :meth:`ReportTabMixin._log_analysis_to_report`, so the same audit info shows in the Report
  tab's source block and rides into the exported data book.

Centralizing here means a new analysis is one ``record_step(...)`` call away from a complete
audit trail, instead of hand-maintaining ``prov.step`` calls scattered across a dozen tabs.
"""
from __future__ import annotations

from .. import provenance


class AuditMixin:
    # ------------------------------------------------------------------ #
    # region → descriptor resolution
    # ------------------------------------------------------------------ #
    def _region_origin(self, rg: dict) -> str:
        """How a region was made, for the audit record. A drawn/lasso/branch ROI carries a
        pixel ``mask``; a segmentation region carries the cluster ids it groups. An explicit
        ``via`` key (even ``""``) wins — callers that pass a synthetic region (e.g. a union
        of ROIs, or a group with only a pixel count) set it to avoid a misleading guess."""
        if "via" in rg:
            return rg.get("via") or ""
        if rg.get("mask") is not None:
            return "drawn/derived ROI"
        segs = rg.get("segments")
        if segs:
            try:
                ids = "+".join(str(s) for s in sorted(int(x) for x in segs))
            except (TypeError, ValueError):
                ids = ",".join(map(str, segs))
            return f"clusters {ids}"
        return ""

    def _mask_bbox(self, mask):
        """Display-grid bounding box ``[x0, y0, x1, y1]`` of a boolean pixel mask, or
        None — informational only, so any failure degrades to None."""
        if mask is None:
            return None
        try:
            import numpy as np
            img = np.asarray(self.ds.to_image(np.asarray(mask, dtype=float)))
            ys, xs = np.where(img > 0)
            if xs.size == 0:
                return None
            return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        except Exception:  # noqa: BLE001
            return None

    def _audit_region_descriptors(self, regions) -> list:
        """Resolve an analysis's region inputs to a list of
        :func:`provenance.region_descriptor` dicts. Each element of ``regions`` may be a
        region dict, a region name (looked up in ``self.regions``), or a ``(role, region)``
        pair where ``role`` tags the part it played (e.g. ``"A"`` / ``"B"``)."""
        out = []
        for item in (regions or []):
            role, ref = "", item
            if isinstance(item, (tuple, list)) and len(item) == 2:
                role, ref = item
            rg = ref
            if isinstance(ref, str):
                rg = next((r for r in getattr(self, "regions", [])
                           if r.get("name") == ref), None)
                if rg is None:                      # name with no live region — still record it
                    out.append(provenance.region_descriptor(ref, role=role or ""))
                    continue
            if not isinstance(rg, dict):
                continue
            try:
                mask = self._region_pixel_mask(rg)
            except Exception:  # noqa: BLE001
                mask = rg.get("mask")
            out.append(provenance.region_descriptor(
                rg.get("name", "?"), role=role or "",
                created_via=self._region_origin(rg), mask=mask,
                n_pixels=rg.get("n_pixels"),     # fallback when there's no resolvable mask
                bbox=self._mask_bbox(mask)))
        return out

    # ------------------------------------------------------------------ #
    # the one call every analysis makes
    # ------------------------------------------------------------------ #
    def record_step(self, kind: str, *, label: str = "", params: dict | None = None,
                    regions=None, now: str | None = None) -> dict:
        """Record one analysis into the session audit trail and return a ``source_extra``
        dict for the Report tab.

        ``kind`` is the canonical step name (drives the auto-drafted methods paragraph and
        citations — see :data:`smile_msi.provenance` step names). ``label`` is a
        human-readable title for the Report block. ``params`` is every setting that affected
        the result. ``regions`` are the ROIs that fed it (see
        :meth:`_audit_region_descriptors`)."""
        params = dict(params or {})
        descs = self._audit_region_descriptors(regions)
        if getattr(self, "prov", None) is not None:
            try:
                step_params = dict(params)
                if descs:
                    step_params["regions"] = descs
                self.prov.step(kind, now=now, **step_params)
            except Exception:  # noqa: BLE001 — the audit record never breaks the analysis
                import traceback
                traceback.print_exc()
        # the audit trail is session state → make sure it gets auto-saved even when the
        # analysis itself changed nothing else persistable (e.g. a colocalization ranking)
        if hasattr(self, "_mark_dirty"):
            try:
                self._mark_dirty()
            except Exception:  # noqa: BLE001
                pass
        extra = {}
        if label:
            extra["method"] = label
        settings = provenance.format_settings(params)
        if settings:
            extra["params"] = settings
        if descs:
            extra["regions"] = provenance.format_regions(descs)
        return extra

    def audit_csv_header(self, *, analysis: str = "", settings=None, regions=None) -> list:
        """The leading ``# …`` audit block for an exported CSV/TSV, built from this sample's
        provenance. ``regions`` may be a pre-rendered string or region inputs (resolved to
        descriptors). Returns ``[]`` when there's no provenance object yet."""
        prov = getattr(self, "prov", None)
        if prov is None:
            return []
        if regions is not None and not isinstance(regions, str):
            regions = provenance.format_regions(self._audit_region_descriptors(regions))
        try:
            return prov.csv_header_lines(analysis=analysis, settings=settings, regions=regions)
        except Exception:  # noqa: BLE001
            return []

    def audit_header_for_steps(self, step_names, *, analysis: str = "") -> list:
        """CSV audit header sourced from the most recent recorded step whose name is in
        ``step_names`` — so an exported feature/stats table carries the exact settings + ROIs
        that produced it. Falls back to a dataset+software-only header when nothing matched."""
        prov = getattr(self, "prov", None)
        if prov is None:
            return []
        wanted = set(step_names)
        step = next((s for s in reversed(prov.steps or []) if s.get("step") in wanted), None)
        settings = regions = None
        if step:
            params = dict(step.get("params") or {})
            descs = params.pop("regions", None)
            settings = provenance.format_settings(params)
            if descs:
                regions = provenance.format_regions(descs)
        try:
            return prov.csv_header_lines(analysis=analysis, settings=settings, regions=regions)
        except Exception:  # noqa: BLE001
            return []
