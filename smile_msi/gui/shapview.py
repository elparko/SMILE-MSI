"""ShapMixin — retired (plan 24).

The SHAP biomarker bubble plot used to live in a dedicated tab built by ``_tab_shap``.
Plan 24 moved every downstream analysis into the generic AnalysisDialog, launched from the
Analyze gallery: SHAP now runs as a gallery card → AnalysisDialog → the GUI-free registry
``_run_shap`` (see :mod:`smile_msi.registry`), which drives :mod:`smile_msi.explain` and
renders via :mod:`smile_msi.export`. The whole tab body — assembly, run/draw, cohort loader,
export, and the FigureEditorDialog integration — was therefore dead (never registered, no
menu / reveal_view / Export-hub caller) and has been removed.

The class name is kept as an empty mixin so the existing ``from .shapview import ShapMixin``
import and the ``MainWindow`` base-class list stay valid without touching ``main.py``. The two
cross-tab calls from ``segment.py`` (``_shap_refresh_groupings`` / ``_shap_update_region_btn``)
are ``hasattr``-guarded and degrade to no-ops now that the methods are gone.
"""
from __future__ import annotations


class ShapMixin:
    pass
