"""Frozen-app entry point (PyInstaller analyses this script)."""
import os

from smile_msi.gui import boot

if __name__ == "__main__":
    if os.environ.get("SMILE_MSI_SELFTEST"):
        # Frozen-bundle smoke test (used by CI right after PyInstaller): import the whole
        # GUI stack and build the main window, then exit 0. Catches a missing hidden import
        # in the bundle before it ships, without needing a display or human interaction.
        from PySide6 import QtWidgets

        from smile_msi.gui.main import MainWindow, apply_app_identity, _name_macos_menubar
        _name_macos_menubar()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        apply_app_identity(app)
        MainWindow()
        # Verify the optional UMAP stack actually runs in the bundle (numba/llvmlite are the
        # hardest things to freeze). Non-fatal — the app falls back to t-SNE — but the result
        # is printed so the CI log shows whether [embed] really made it into the build.
        try:
            import numpy as _np
            import umap as _umap
            _umap.UMAP(n_neighbors=5, n_components=2).fit_transform(_np.random.rand(20, 4))
            umap_status = "UMAP verified"
        except Exception as exc:  # noqa: BLE001
            umap_status = f"UMAP unavailable: {type(exc).__name__}"
        print(f"SMILE MSI selftest OK ({umap_status})", flush=True)
        # The import + window build (the thing we're verifying) succeeded. Exit hard so a
        # Qt/atexit teardown — which can abort with a non-zero code in a frozen bundle when
        # the QApplication was never exec()'d — can't turn a passing check into a failure.
        os._exit(0)
    else:
        boot()
