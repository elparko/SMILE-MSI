"""AdvancedTabsMixin — retired.

The supervised Classify tab (PLS-DA / OPLS-DA + cross-validation) that lived here was
removed in plan 24: its engine calls now run through the generic
:class:`~smile_msi.gui.analysisdialog.AnalysisDialog` launched from the Analyze gallery,
registered as steps in :mod:`smile_msi.registry`. Markers (SSC) and Per-ion (DGMM) moved
the same way earlier.

The mixin is kept as an (empty) base so ``MainWindow``'s MRO is unchanged; it carries no
behavior. Drop it from :mod:`smile_msi.gui.main` to delete this module entirely.
"""
from __future__ import annotations


class AdvancedTabsMixin:
    pass
