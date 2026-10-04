"""First-run **Setup wizard** (File ▸ Setup wizard…) — walks a new user through the few choices
that matter for good lipid annotation, so they don't have to discover them:

1. **Ionization mode** + **identification tolerance** (with a note that auto-recalibration
   handles a systematic mass offset, so a tight tolerance is fine).
2. **Lipid database** — keep the built-in in-silico DB, or add **LIPID MAPS** (drop in the
   downloaded ``structures.sdf`` and it's converted + merged in one step) / a custom CSV. The
   choice is **remembered** (``prefs``) and auto-loaded on every future launch, so setup sticks.

The wizard only *drives* the main window (sets its mode/tolerance combos, calls its shared
``_load_lipid_db_file`` / ``_set_lipid_db``); all the real logic lives on ``MainWindow`` so the
same paths back the manual *Data ▸ Import lipid database…* action. ``apply()`` is separated from
the Qt ``accept()`` so tests can drive the whole thing headless.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .common import NoScrollComboBox, NoScrollDoubleSpinBox

from .. import prefs
from . import filedialogs

LMSD_URL = "https://www.lipidmaps.org/databases/lmsd/download"
# a sensible MSI slice of the ~48k full LMSD: the lipid categories + the m/z span MSI covers,
# so the merged DB isn't 40k mostly-irrelevant species inflating isobaric ambiguity.
MSI_CATEGORIES = ["Glycerophospholipid", "Sphingolipid", "Sterol", "Fatty Acyl",
                  "Glycerolipid", "Prenol", "Saccharolipid"]
MSI_MIN_MASS, MSI_MAX_MASS = 250.0, 1100.0


class SetupWizard(QtWidgets.QWizard):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self._loaded_db = None            # (db, note) once a file is validated
        self._db_cfg = None               # the config to remember in prefs
        self.setWindowTitle("SMILE MSI — Setup")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)
        self.setOption(QtWidgets.QWizard.NoBackButtonOnStartPage, True)
        self.addPage(self._welcome_page())
        self.addPage(self._data_page())
        self.addPage(self._db_page())
        self.addPage(self._done_page())
        self.resize(620, 480)

    # ---- pages ------------------------------------------------------------- #
    def _welcome_page(self):
        p = QtWidgets.QWizardPage()
        p.setTitle("Welcome to SMILE MSI")
        v = QtWidgets.QVBoxLayout(p)
        v.addWidget(QtWidgets.QLabel(
            "This quick setup gets lipid annotation working well:\n\n"
            "  •  your ionization mode and mass tolerance\n"
            "  •  the lipid database to identify against (built-in, or LIPID MAPS)\n\n"
            "Everything here can also be changed later from the toolbar and the Data menu. "
            "Mass mis-calibration is corrected automatically, so you don't set that here."))
        v.addStretch(1)
        self.demo_check = QtWidgets.QCheckBox("Load the demo dataset when I finish (to explore)")
        v.addWidget(self.demo_check)
        return p

    def _data_page(self):
        p = QtWidgets.QWizardPage()
        p.setTitle("Data & matching")
        p.setSubTitle("How your ions were acquired and how tightly to match them.")
        form = QtWidgets.QFormLayout(p)
        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems(["negative", "positive"])
        self.mode_combo.setCurrentText(self.win.mode_combo.currentText())
        form.addRow("Ionization mode:", self.mode_combo)
        self.tol_spin = NoScrollDoubleSpinBox()
        self.tol_spin.setRange(1.0, 30.0); self.tol_spin.setSingleStep(1.0)
        self.tol_spin.setValue(float(self.win.id_ppm)); self.tol_spin.setSuffix(" ppm")
        form.addRow("Identification tolerance:", self.tol_spin)
        note = QtWidgets.QLabel(
            "5 ppm is a good default: the app measures your data's systematic mass offset and "
            "matches at the corrected m/z automatically, so you don't need to loosen this to "
            "catch mis-calibrated ions.")
        note.setWordWrap(True); note.setStyleSheet("color: gray;")
        form.addRow(note)
        return p

    def _db_page(self):
        p = QtWidgets.QWizardPage()
        p.setTitle("Lipid database")
        p.setSubTitle("What to identify ions against.")
        v = QtWidgets.QVBoxLayout(p)
        self.db_builtin = QtWidgets.QRadioButton("Built-in in-silico database (recommended default)")
        self.db_builtin.setChecked(True)
        self.db_external = QtWidgets.QRadioButton("Add LIPID MAPS / a custom database")
        v.addWidget(self.db_builtin); v.addWidget(self.db_external)

        self._ext_box = QtWidgets.QWidget()
        g = QtWidgets.QVBoxLayout(self._ext_box); g.setContentsMargins(24, 4, 0, 0)
        link = QtWidgets.QLabel(f'Download the LMSD structure file (SDF): '
                                f'<a href="{LMSD_URL}">{LMSD_URL}</a><br>'
                                'Pick the <b>“LMSD (ZIP)”</b> option, unzip, and choose the '
                                '<code>structures.sdf</code> below (a CSV/TSV with a formula '
                                'column also works).')
        link.setOpenExternalLinks(True); link.setWordWrap(True)
        g.addWidget(link)
        row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(); self.path_edit.setPlaceholderText("…/structures.sdf or a .csv")
        browse = QtWidgets.QPushButton("Browse…"); browse.clicked.connect(self._browse_db)
        row.addWidget(self.path_edit, 1); row.addWidget(browse)
        g.addLayout(row)
        self.msi_slice = QtWidgets.QCheckBox(
            f"Trim to MSI-relevant lipids ({int(MSI_MIN_MASS)}–{int(MSI_MAX_MASS)} m/z, lipid "
            "categories) — fewer false isobars (recommended for a big LMSD)")
        self.msi_slice.setChecked(True)
        g.addWidget(self.msi_slice)
        self.db_merge = QtWidgets.QRadioButton("Merge onto built-in (keep tissue priors + metabolites)")
        self.db_merge.setChecked(True)
        self.db_replace = QtWidgets.QRadioButton("Replace built-in with only this database")
        g.addWidget(self.db_merge); g.addWidget(self.db_replace)
        self.db_validate = QtWidgets.QPushButton("Load / validate")
        self.db_validate.clicked.connect(self._validate_db)
        g.addWidget(self.db_validate)
        self.db_status = QtWidgets.QLabel(""); self.db_status.setWordWrap(True)
        g.addWidget(self.db_status)
        v.addWidget(self._ext_box)
        v.addStretch(1)

        self._ext_box.setEnabled(False)
        self.db_external.toggled.connect(self._ext_box.setEnabled)
        # a fresh file selection invalidates a previous validation
        self.path_edit.textChanged.connect(lambda *_: setattr(self, "_loaded_db", None))

        # Re-running the wizard must not silently drop an already-remembered database — restore
        # it here so Finish (without touching this page) keeps it instead of reverting to
        # built-in. apply()'s "Finish without Load/validate" fallback re-loads path_edit's text,
        # so leaving _loaded_db unset (via the textChanged handler above) is fine.
        cfg = prefs.get(self.win._LIPID_DB_PREF)
        if isinstance(cfg, dict) and cfg.get("path"):
            self.db_external.setChecked(True)
            self.path_edit.setText(cfg["path"])
            self.db_replace.setChecked(cfg.get("mode") == "replace")
            self.msi_slice.setChecked(bool(cfg.get("categories")))
        return p

    def _done_page(self):
        p = QtWidgets.QWizardPage()
        p.setTitle("You're set up")
        v = QtWidgets.QVBoxLayout(p)
        v.addWidget(QtWidgets.QLabel(
            "That's it. Open an imzML (File ▸ Open imzML…) or a saved analysis, find peaks, and "
            "build a feature list — lipids are identified with auto-recalibration and your chosen "
            "database.\n\nYou can re-run this any time from File ▸ Setup wizard…, and swap the "
            "database from Data ▸ Import lipid database…."))
        v.addStretch(1)
        return p

    # ---- db validation ----------------------------------------------------- #
    def _browse_db(self):
        path, _ = filedialogs.get_open_file_name(
            self, "Choose lipid database", "",
            "Lipid databases (*.sdf *.csv *.tsv *.txt);;All (*)")
        if path:
            self.path_edit.setText(path)

    def _db_filters(self):
        if self.msi_slice.isChecked():
            return {"categories": MSI_CATEGORIES, "min_mass": MSI_MIN_MASS, "max_mass": MSI_MAX_MASS}
        return {"categories": None, "min_mass": None, "max_mass": None}

    def _validate_db(self):
        """'Load / validate' button. Runs the parse in the background (:meth:`MainWindow._run`
        on ``self.win``) — a large LMSD ``.sdf`` (~48k records) parsed inline here used to
        freeze the wizard for as long as the parse took, which reads as a crash."""
        path = self.path_edit.text().strip()
        if not path:
            self.db_status.setText("Choose a file first.")
            return
        mode = "merge" if self.db_merge.isChecked() else "replace"
        filt = self._db_filters()
        self.db_status.setText("Loading…")
        self.db_validate.setEnabled(False)

        def load():
            try:
                db, note = self.win._load_lipid_db_file(path, mode=mode, **filt)
            except Exception as e:  # noqa: BLE001 — surfaced inline below, not a popup
                return None, f"⚠ Could not load: {e}"
            if not db:
                return [], "⚠ No usable lipids (need name + formula)."
            return db, note

        def done(result):
            self.db_validate.setEnabled(True)
            db, note = result
            if not db:                            # None (exception) or [] (empty) — note has the reason
                self._loaded_db = None
                self.db_status.setText(note)
                return
            self._loaded_db = (db, note)
            self._db_cfg = {"path": path, "mode": mode, **filt}
            self.db_status.setText(f"✓ {note}. Total database: {len(db)} lipids.")

        self.win._run(load, on_done=done, busy="Loading lipid database…")

    # ---- apply (headless-testable) ----------------------------------------- #
    def apply(self):
        """Push the wizard's choices onto the main window + prefs. Called from ``accept()``;
        also directly callable in tests."""
        self.win.mode_combo.setCurrentText(self.mode_combo.currentText())
        self.win.id_ppm_spin.setValue(float(self.tol_spin.value()))
        if self.db_external.isChecked():
            if self._loaded_db is None:          # user hit Finish without pressing Load/validate
                # _validate_db() is now async (see above) so it can't be depended on to have
                # populated self._loaded_db by the time we get here — load inline instead. This
                # is the one path still capable of blocking on a huge file, same as before this
                # fix; the common, explicit "Load / validate" click path above no longer does.
                path = self.path_edit.text().strip()
                if path:
                    mode = "merge" if self.db_merge.isChecked() else "replace"
                    filt = self._db_filters()
                    try:
                        db, note = self.win._load_lipid_db_file(path, mode=mode, **filt)
                    except Exception:  # noqa: BLE001 — apply() must not raise into accept()
                        db, note = None, ""
                    if db:
                        self._loaded_db = (db, note)
                        self._db_cfg = {"path": path, "mode": mode, **filt}
            if self._loaded_db is not None:
                db, _note = self._loaded_db
                self.win._set_lipid_db(db)
                prefs.set(self.win._LIPID_DB_PREF, self._db_cfg)
        else:
            self.win._set_lipid_db(None)         # built-in
            prefs.set(self.win._LIPID_DB_PREF, None)
        prefs.set("setup_done", True)
        if self.demo_check.isChecked() and hasattr(self.win, "do_load_demo"):
            QtCore.QTimer.singleShot(0, self.win.do_load_demo)

    def accept(self):
        self.apply()
        super().accept()
