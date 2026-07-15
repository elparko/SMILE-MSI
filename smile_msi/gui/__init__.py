"""SMILE MSI desktop application (PySide6 + pyqtgraph).

A desktop workspace for MALDI-MSI: ion images, mean spectrum + peak
picking, segmentation, co-localization, and exact ROI statistics — with lipid
annotation — backed by the out-of-core engine in :mod:`smile_msi.msi` /
:mod:`smile_msi.spatial`.

Launch with ``smile-msi-gui`` or ``python -m smile_msi.gui``.
"""

# NB: do NOT eagerly ``from .main import run`` here — that would pull in pyqtgraph (~250 ms)
# at console-script load, before the startup splash can paint. Both entry points are lazy
# shims so ``boot`` shows the splash *first*, then imports ``main``.


def boot(argv=None, open_path=None):
    """Real GUI entry point (``smile-msi-gui``): show the splash, then start the app."""
    from .splash import boot as _boot
    return _boot(argv, open_path)


def run(argv=None, open_path=None):
    """Build the window and run without a splash (kept for direct/programmatic callers)."""
    from .main import run as _run
    return _run(argv, open_path)


__all__ = ["boot", "run"]
