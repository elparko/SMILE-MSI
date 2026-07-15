"""The Analyze gallery — a card grid over the flow registry (plan 24, Phase 1).

Every analysis a flow step can be (:data:`registry.REGISTRY`) is surfaced as a card, grouped
by category. A card reads its readiness off a Qt-free ``state`` snapshot through
:func:`registry.unmet_needs`; a ready card, when clicked, merely reveals the existing tab that
already implements it (``reveal_view(sd.view)``). No analysis is re-implemented here — the
gallery is a launcher over the tabs that already exist, so it is zero-risk.

The gallery is the first consumer of the plan-24 signal bus (``datasetChanged`` /
``peaksChanged`` / ``regionsChanged`` / ``segChanged`` on :class:`MainWindow`): each fires
:meth:`GalleryMixin._refresh_gallery`, which re-evaluates every card's enabled state + reason.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from . import common
from .. import registry


def _mix(a, b, t):
    """``a`` blended ``t`` of the way toward ``b``."""
    return QtGui.QColor(round(a.red() + (b.red() - a.red()) * t),
                        round(a.green() + (b.green() - a.green()) * t),
                        round(a.blue() + (b.blue() - a.blue()) * t))


def _card_colors(pal):
    """Every colour a card paints, derived from the *live* palette — one source for both
    themes, a live Light/Dark switch, and the contrast test.

    A bare ``StyledPanel`` frame is near-invisible on the dark theme: it carries no fill and
    its hairline border sits ~6 levels off ``Window`` (29,32,36), so the choices stop reading
    as discrete objects. ``AlternateBase`` is barely better (1.08:1 against the window). The
    fill is therefore mixed *toward the ink* by a fixed amount, which lifts the card on dark
    and recesses it on light, and the border is mixed far enough to stay crisp on both."""
    win = pal.color(QtGui.QPalette.Window)
    ink = pal.color(QtGui.QPalette.Text)
    dark = win.lightness() < 128
    fill = _mix(win, ink, 0.14) if dark else pal.color(QtGui.QPalette.Base)
    # A near-white card on a light window separates by edge, not by fill, so the light border
    # is mixed further; on dark the fill does more of the work but the border still needs to
    # clear the window or the cards read as one grey slab ("grey on grey").
    return {
        "fill": fill.name(),
        "fill_hover": _mix(fill, ink, 0.07).name(),
        "border": _mix(win, ink, 0.48 if dark else 0.62).name(),
        "border_off": _mix(win, ink, 0.28 if dark else 0.34).name(),
        "off_fill": win.name(),
        # An *enabled* card reads at full ink/near-white contrast throughout — title and
        # blurb alike — rather than fading the blurb to a muted grey; only a *disabled* card
        # dims (below), since dimness is the signal that should mean "unavailable".
        "title": "#f4f7fa" if dark else ink.name(),
        "desc": "#f4f7fa" if dark else ink.name(),
        # A dimmed card still has to say what it needs, so the reason outranks the blurb.
        "off_title": _mix(win, ink, 0.66).name(),
        "off_desc": _mix(win, ink, 0.62).name(),
        "off_reason": _mix(win, ink, 0.70).name(),
    }


def _card_qss(c):
    """The card FRAME stylesheet — background + border only.

    The label colours are applied DIRECTLY to each QLabel in :meth:`_GalleryCard._restyle`,
    NOT with a ``QLabel#cardTitle {…}`` rule here. A ``color`` rule for a child QLabel written
    in an ancestor widget's stylesheet (a descendant selector) does not reach the label — Qt
    draws QLabel text from the palette and silently drops the inherited ``color``, so every
    card title/blurb rendered a muted grey regardless of the value set here (the "everything
    on the Analyze screen is greyed" bug). Setting the colour on the label itself works, and
    re-running on ``EnabledChange`` keeps the dim-when-unavailable cue."""
    return f"""
QFrame#analysisCard {{
    background: {c['fill']};
    border: 1px solid {c['border']};
    border-radius: 8px;
}}
QFrame#analysisCard:hover {{
    background: {c['fill_hover']};
    border: 1px solid {common.ACCENT};
}}
QFrame#analysisCard:disabled {{
    background: {c['off_fill']};
    border: 1px dashed {c['border_off']};
}}
"""


def _category_header(text):
    """A section heading one clear level above a card title: upper-case, letter-spaced and
    muted, so the two never read as the same rank."""
    lbl = QtWidgets.QLabel(text.upper())
    f = lbl.font()
    f.setBold(True)
    f.setPointSizeF(max(8.0, f.pointSizeF() - 1.0))
    f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 1.2)
    lbl.setFont(f)
    # A section landmark reads at near-primary contrast (its rank is carried by bold + CAPS +
    # letter-spacing, not by greyness) so the Analyze screen has a clear top-of-hierarchy and
    # doesn't read as "all grey". palette(text) tracks the Light/Dark toggle.
    lbl.setStyleSheet("color: palette(text);")
    lbl.setContentsMargins(2, 10, 0, 2)
    return lbl


# Steps that have been lifted from a dedicated tab into the generic AnalysisDialog (plan 24,
# Phase 2b onward). A card whose sd.id is in here opens AnalysisDialog(win, sd) instead of
# revealing a tab. Phase 4 converts more analyses by adding their ids here — nothing else changes.
#
# A step is only left OUT of this set when the view it names genuinely runs it: auto_segment's
# Segmentation tab is its editing surface, and the cohort steps keep their own screens. Anything
# else must be in here — a card that merely reveals "Ion image" and then does nothing (as
# find_peaks did) promises an analysis and delivers navigation.
# The Cohort screens a cohort-scope card can reveal (each cohort analysis keeps its own tab).
_COHORT_SCREENS = {"Cohort", "Cohort nested stats", "Cohort UMAP", "Cohort segmentation"}

CONVERTED = {"find_peaks", "find_spatial_features", "marker_panel", "filter_auc", "annotate",
             "shrunken_centroids", "dgmm",
             "colocalize", "coloc_modules", "region_correlation",
             "plsda", "classify_cv", "classify_map", "shap_biomarkers",
             "pca", "nmf", "embedding",
             "discriminating_features", "multigroup_features", "region_membership",
             "roi_localization", "roi_comparison", "region_comparison",
             "class_comparison", "class_composition"}


class _GalleryCard(QtWidgets.QFrame):
    """A single analysis card — a bold name over a same-ink, wrapped blurb, click-to-launch.

    A *disabled* card (an unmet-needs step) receives no mouse events — Qt routes them to the
    parent — so ``clicked`` fires only for a ready card.

    Every card is the same size. Each text row reserves a fixed number of lines rather than
    hugging its content, so a short blurb and a long one produce identically-sized cards and
    the FlowLayout wraps into a true grid instead of a ragged one. An over-long blurb clips;
    the full text is always on the tooltip."""

    clicked = QtCore.Signal()

    TITLE_LINES, DESC_LINES, REASON_LINES = 2, 3, 2
    WIDTH = 236

    def __init__(self, name, description, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisCard")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)   # QSS 'background' needs this
        self.setFixedWidth(self.WIDTH)
        self.setToolTip(description)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        self.title = QtWidgets.QLabel(name)
        self.title.setObjectName("cardTitle")
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)

        self.desc = QtWidgets.QLabel(description)
        self.desc.setObjectName("cardDesc")

        self.reason = QtWidgets.QLabel("")
        self.reason.setObjectName("cardReason")
        rf = self.reason.font()
        rf.setItalic(True)
        self.reason.setFont(rf)

        for lbl, lines in ((self.title, self.TITLE_LINES),
                           (self.desc, self.DESC_LINES),
                           (self.reason, self.REASON_LINES)):
            lbl.setWordWrap(True)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            lbl.setFixedHeight(QtGui.QFontMetrics(lbl.font()).lineSpacing() * lines)
            lay.addWidget(lbl)

        # Fix the height too, not just the width: FlowLayout hands each item its row's height,
        # which would stretch a card to the tallest thing in the row.
        lay.addStretch(0)
        self.setFixedHeight(self.sizeHint().height())

        self._restyling = False
        self._restyle()
        self._sync_cursor()

    def _restyle(self):
        """Re-apply the card QSS from the *application* palette.

        Two hazards, both hit in practice. Qt re-resolves a widget's palette when a stylesheet
        carrying ``color:`` is set, which posts a ``PaletteChange`` straight back into
        :meth:`changeEvent` — so this must not re-enter, and it must not read
        ``self.palette()``, whose value the stylesheet itself perturbs."""
        if self._restyling:
            return
        self._restyling = True
        try:
            c = _card_colors(QtWidgets.QApplication.palette())
            self.setStyleSheet(_card_qss(c))
            # Label colours are set DIRECTLY on each QLabel — a descendant `color` rule in the
            # frame stylesheet never reaches them (see _card_qss). `on` picks the enabled or
            # dimmed tone; changeEvent re-runs this on EnabledChange so a card that flips
            # ready/unavailable recolours its title + blurb + reason.
            on = self.isEnabled()
            self.title.setStyleSheet(f"color: {c['title'] if on else c['off_title']}; background: transparent;")
            self.desc.setStyleSheet(f"color: {c['desc'] if on else c['off_desc']}; background: transparent;")
            self.reason.setStyleSheet(f"color: {c['desc'] if on else c['off_reason']}; background: transparent;")
        finally:
            self._restyling = False

    def _sync_cursor(self):
        # A dimmed card is not actionable — the hand cursor would promise a click that no-ops.
        self.setCursor(QtCore.Qt.PointingHandCursor if self.isEnabled()
                       else QtCore.Qt.ArrowCursor)

    def changeEvent(self, ev):                # noqa: N802 (Qt API)
        if ev.type() == QtCore.QEvent.PaletteChange:
            self._restyle()                   # follow a live Light/Dark switch
        elif ev.type() == QtCore.QEvent.EnabledChange:
            self._restyle()                   # recolour title/blurb/reason for the new state
            self._sync_cursor()
        super().changeEvent(ev)

    def mouseReleaseEvent(self, ev):          # noqa: N802 (Qt API)
        if ev.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mouseReleaseEvent(ev)


class GalleryMixin:
    """The Analyze tab: a scrollable card gallery over :func:`registry.registry_by_category`."""

    def _tab_gallery(self):
        w, v = common.tab_page()
        v.addWidget(common.section_title("Analyze"))
        v.addWidget(common.note(
            "Every analysis in one place. A card lights up when its inputs are ready; click it "
            "to open the view that runs it. A dimmed card lists what it still needs."))

        # Scope toggle: filter cards by sd.targets. "This slide" is the single-slide surface;
        # "Cohort" the cross-sample one. The visibility test reuses flowdialog.py:1200's exact
        # predicate (see _gallery_card_out_of_scope).
        self._gallery_scope = "slide"
        bar = common.ControlBar()
        b_slide = common.button("This slide", lambda: self._set_gallery_scope("slide"),
                                tooltip="Analyses that run on the loaded slide")
        b_cohort = common.button("Cohort", lambda: self._set_gallery_scope("cohort"),
                                 tooltip="Analyses that run across every sample in the cohort")
        for b in (b_slide, b_cohort):
            b.setCheckable(True)
        b_slide.setChecked(True)
        grp = QtWidgets.QButtonGroup(w)
        grp.setExclusive(True)
        grp.addButton(b_slide)
        grp.addButton(b_cohort)
        self._gallery_scope_buttons = grp
        bar.add_group("Scope", b_slide, b_cohort)
        v.addWidget(bar)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        iv = QtWidgets.QVBoxLayout(inner)
        iv.setContentsMargins(2, 2, 2, 2)
        iv.setSpacing(8)

        # (sd, card, reason_label) for every card; (header, [cards]) per category — the two
        # registries _refresh_gallery walks. Built once; membership never changes.
        self._gallery_cards = []
        self._gallery_sections = []
        for category, defs in registry.registry_by_category().items():
            header = _category_header(category)
            iv.addWidget(header)
            row = QtWidgets.QWidget()
            common.FlowLayout(row)
            section_cards = []
            for sd in defs:
                card = _GalleryCard(sd.name, sd.description)
                card.clicked.connect(lambda s=sd: self._launch_card(s))
                row.layout().addWidget(card)
                self._gallery_cards.append((sd, card, card.reason))
                section_cards.append(card)
            iv.addWidget(row)
            self._gallery_sections.append((header, section_cards))
        iv.addStretch(1)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        self.tabs.addTab(w, "Analyze")
        self._refresh_gallery()

    def _launch_card(self, sd):
        """Open a ready card.

        In the **Cohort** scope the analysis is authoritative across subjects, so the card
        opens the Cohort screen (the "defer to Cohort" mechanism, plan 24 §3) rather than the
        single-slide dialog. In the **This slide** scope a *converted* step (``sd.id in
        CONVERTED``) opens the generic :class:`~smile_msi.gui.analysisdialog.AnalysisDialog`;
        every other card reveals the tab that implements it (``reveal_view(sd.view)``)."""
        if getattr(self, "_gallery_scope", "slide") == "cohort":
            # cohort analyses keep their own full screen — open the one this card names, or the
            # main Cohort screen for a slide step that merely also has a cohort counterpart.
            self.reveal_view(sd.view if sd.view in _COHORT_SCREENS else "Cohort")
            return
        if sd.id in CONVERTED:
            from . import analysisdialog          # lazy: keeps gallery import light + patchable
            self.open_analysis(analysisdialog.AnalysisDialog(self, sd))
        else:
            self.reveal_view(sd.view)

    def _set_gallery_scope(self, scope):
        self._gallery_scope = scope
        self._refresh_gallery()

    @staticmethod
    def _gallery_card_out_of_scope(sd, scope):
        """True when a card must be hidden in the current scope — the exact target/scope gate
        from ``gui/flowdialog.py`` (``win.registry.target == "slide" and "slide" not in sd.targets``),
        mirrored for the cohort scope."""
        return scope not in sd.targets

    def _gallery_state(self):
        """The Qt-free ``state`` dict :func:`registry.unmet_needs` / :func:`registry.resolve_inputs`
        consume, built from the live window (no Qt objects leak in). Masks are resolved through
        the same ``_region_pixel_mask`` the runner uses; ``mask`` is left ``None`` (whole slide)
        because the gallery asks "could this run?", not "run it in scope X". See the ``state``
        schema documented in ``registry.py`` above ``_resolve``."""
        ds = getattr(self, "ds", None)
        n_pixels = int(ds.n_pixels) if ds is not None else 0
        mzs = [float(p["mz"]) for p in (self.peaks or [])]

        groups = {}                       # {group_label: [pixel mask, …]}, first-seen order
        regions = {}                      # {region_name: pixel mask}
        for rg in (self.regions or []):
            mk = self._region_pixel_mask(rg)
            g = (rg.get("group") or "").strip()
            if g:
                groups.setdefault(g, []).append(mk)
            name = rg.get("name")
            if name is not None:
                regions[name] = mk

        seg = getattr(self, "seg", None)
        seg_labels = getattr(seg, "labels", None) if seg is not None else None
        # Name each segmentation cluster for the per-group tables: the region a cluster was
        # assigned to, else the lineage label the Segmentation tab shows ('Segment 0·2·1'). Lets
        # the seg fallback in _resolve_group_labels read real names instead of 'region N'.
        seg_names = None
        if seg is not None:
            lin = getattr(self, "_seg_lineage", {}) or {}
            seg_names = [(reg["name"] if (reg := self._seg_region_of(cl)) else f"Segment {lin.get(cl, cl)}")
                         for cl in range(seg.n_clusters)]

        # With no group tags AND no segmentation, let named ROIs stand in as the groups (their
        # names, not 'region N') so a slide of drawn regions is analysable — mirrors the Stats
        # tab's _labels_from_regions (nested children dropped so a parent+child don't both group).
        # A live segmentation is left to the seg fallback above, so this never overrides it.
        if not groups and seg is None:
            for rg in self._drop_nested_children(self.regions or []):
                nm = rg.get("name")
                mk = regions.get(nm)
                if nm and mk is not None and mk.any():
                    groups.setdefault(nm, []).append(mk)
            if len(groups) < 2:               # <2 named ROIs with pixels → nothing to compare
                groups = {}

        return {
            "n_pixels": n_pixels,
            "mask": None,                 # whole slide — the gallery evaluates readiness, not scope
            "mzs": mzs,
            "groups": groups,
            "seg_labels": seg_labels,
            "seg_names": seg_names,
            "regions": regions,
            "target_mz": getattr(self, "active_mz", None),
            "stats_df": getattr(self, "last_stats", None),
        }

    def _refresh_gallery(self):
        """Re-evaluate every card's scope visibility + enabled state + unmet-needs reason.

        No-ops safely before ``_tab_gallery`` has built its widgets (order-of-construction guard):
        the four signals and ``_refresh_action_states`` may fire during early setup."""
        cards = getattr(self, "_gallery_cards", None)
        if not cards:
            return
        scope = getattr(self, "_gallery_scope", "slide")
        state = self._gallery_state()
        for sd, card, reason in cards:
            hidden = self._gallery_card_out_of_scope(sd, scope)
            card.setVisible(not hidden)
            if hidden:
                continue
            reasons = registry.unmet_needs(state, sd)
            card.setEnabled(not reasons)               # disabled → dimmed + un-clickable
            # The reason row stays laid out even when empty: hiding it would shrink that one
            # card and ragged the grid. A ready card simply shows blank space.
            reason.setText(" · ".join(reasons))
        for header, section_cards in getattr(self, "_gallery_sections", []):
            header.setVisible(any(c.isVisible() for c in section_cards))
