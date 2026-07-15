"""OS/app-identity helpers that must run **before** the QApplication (Cocoa) is
created and before the heavy GUI stack (pyqtgraph, numpy) is imported.

Split out of ``main.py`` so the startup splash (``splash.py`` / ``boot()``) can call
them without dragging in pyqtgraph — the whole point of the splash is to paint the
first pixel *before* that ~250 ms import runs. ``main.py`` re-exports these names, so
every existing ``main._name_macos_menubar`` / ``main._set_windows_app_id`` reference
(including the frozen-bundle selftest in ``scripts/launch.py``) keeps working.

Imports only stdlib + the light ``smile_msi`` package (~9 ms) — never Qt widgets or
pyqtgraph.
"""
from __future__ import annotations

import sys

from .. import APP_NAME


def _name_macos_menubar():
    """Make the macOS menu-bar app menu / Dock / ⌘-Tab read APP_NAME, not "Python".

    Unbundled (``python -m smile_msi.gui``) and the .app launcher (which ``exec``s an
    external interpreter) both leave Cocoa with no bundle name, so it falls back to the
    process name "Python". Overwriting ``CFBundleName`` in the main bundle's info dict —
    *before* the QApplication/Cocoa app is created — fixes the menu bar. No-op off macOS
    or when pyobjc isn't installed (the packaged .app also sets argv[0] via ``exec -a``)."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = APP_NAME
    except Exception:                # pyobjc absent or API shape changed — non-fatal
        pass


def _set_windows_app_id():
    """On Windows, give the process an explicit AppUserModelID so the taskbar shows the
    app's *own* icon and groups under "SMILE MSI" rather than the generic python.exe icon.
    Must run before any window is shown. No-op (and never fatal) off Windows."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("org.smilemsi.SMILEMSI")
    except Exception:                # noqa: BLE001 — cosmetic taskbar grouping, never fatal
        pass
