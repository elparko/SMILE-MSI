"""Analysis Profiles manager (Data ▸ Analysis profiles…).

A settings screen for the :mod:`smile_msi.profiles` store: pick the active profile,
edit a profile's standardized processing settings, save it as a new version, and
export/import to share. The editor is generated from :data:`profiles.SCHEMA`, so it
stays in sync with the engine automatically.

Soft by design — the active profile only pre-fills defaults; "Apply to current sample"
pushes them into the live controls, and the deviation banner shows where the open
sample has departed from the active profile (the departure is recorded, never blocked).
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .. import profiles
from . import filedialogs
from .common import (confirm, note, section_title, NoScrollComboBox, NoScrollDoubleSpinBox,
                     NoScrollSpinBox)


class ProfilesDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Analysis profiles")
        self.resize(760, 620)
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._working: dict = profiles.active()

        root = QtWidgets.QHBoxLayout(self)

        # ---- left: profile list + library actions --------------------------
        left = QtWidgets.QVBoxLayout()
        left.addWidget(section_title("Profiles"))
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(240)
        self.list.currentItemChanged.connect(self._on_select)
        left.addWidget(self.list, 1)
        for text, slot in (("New", self._new), ("Duplicate", self._duplicate),
                           ("Set active", self._set_active), ("Delete", self._delete),
                           ("Import…", self._import), ("Export…", self._export)):
            b = QtWidgets.QPushButton(text)
            if text == "Set active":
                b.setObjectName("primaryAction")          # the one accent action of the list
            elif text == "Delete":
                b.setObjectName("dangerAction")           # destructive (all versions) → red tier
            b.clicked.connect(slot)
            left.addWidget(b)
            self._editors.setdefault("_btn_" + text, b)
        root.addLayout(left)

        # ---- right: editor -------------------------------------------------
        right = QtWidgets.QVBoxLayout()
        self.title = section_title("")
        right.addWidget(self.title)
        self.hint = note("The active profile pre-fills processing defaults for every new "
                         "sample. Editing only changes this profile — apply it to the open "
                         "sample below; per-sample changes stay allowed and are recorded.")
        right.addWidget(self.hint)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Name"))
        self.name_edit = QtWidgets.QLineEdit()
        name_row.addWidget(self.name_edit, 1)
        right.addLayout(name_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        form_v = QtWidgets.QVBoxLayout(inner)
        for group in profiles.GROUPS:
            form_v.addWidget(section_title(group))
            form = QtWidgets.QFormLayout()
            for p in profiles.SCHEMA:
                if p.group != group:
                    continue
                w = self._make_widget(p)
                self._editors[p.key] = w
                w.setToolTip(p.help)
                form.addRow(p.label, w)
            form_v.addLayout(form)
        form_v.addStretch()
        scroll.setWidget(inner)
        right.addWidget(scroll, 1)

        self.deviation = note("")
        right.addWidget(self.deviation)

        btn_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save as new version")
        self.save_btn.setObjectName("primaryAction")      # principal commit action of the editor
        self.save_btn.clicked.connect(self._save)
        self.apply_btn = QtWidgets.QPushButton("Apply to current sample")
        self.apply_btn.clicked.connect(self._apply_to_sample)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.apply_btn)
        btn_row.addStretch()
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.hide)
        btn_row.addWidget(close)
        right.addLayout(btn_row)
        root.addLayout(right, 1)

        self.refresh()

    # ------------------------------------------------------------------ #
    # editor widgets
    # ------------------------------------------------------------------ #
    def _make_widget(self, p: profiles.Param) -> QtWidgets.QWidget:
        if p.kind == "choice":
            w = NoScrollComboBox()
            w.addItems(list(p.choices))
            return w
        if p.kind == "int":
            w = NoScrollSpinBox()
            w.setRange(int(p.lo if p.lo is not None else 0),
                       int(p.hi if p.hi is not None else 1_000_000))
            return w
        if p.kind == "float":
            w = NoScrollDoubleSpinBox()
            w.setRange(float(p.lo if p.lo is not None else 0.0),
                       float(p.hi if p.hi is not None else 1e9))
            w.setDecimals(p.decimals)
            return w
        return QtWidgets.QLineEdit()

    def _editor_set(self, params: dict) -> None:
        for p in profiles.SCHEMA:
            w = self._editors[p.key]
            v = params.get(p.key, p.default)
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentText(str(v))
            elif isinstance(w, QtWidgets.QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                w.setValue(float(v))
            else:
                w.setText(str(v))

    def _editor_get(self) -> dict:
        out = {}
        for p in profiles.SCHEMA:
            w = self._editors[p.key]
            if isinstance(w, QtWidgets.QComboBox):
                out[p.key] = w.currentText()
            elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                out[p.key] = w.value()
            else:
                out[p.key] = w.text()
        return profiles.merge_defaults(out)

    # ------------------------------------------------------------------ #
    # list / selection
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        active_name = profiles.active().get("name")
        self.list.blockSignals(True)
        self.list.clear()
        sel_row = 0
        for i, prof in enumerate(profiles.list_profiles()):
            mark = "● " if prof.get("name") == active_name else ""
            it = QtWidgets.QListWidgetItem(mark + profiles.label(prof))
            it.setData(QtCore.Qt.UserRole, prof.get("name"))
            self.list.addItem(it)
            if prof.get("name") == self._working.get("name"):
                sel_row = i
        self.list.setCurrentRow(sel_row)
        self.list.blockSignals(False)
        self._load_into_editor(profiles.load(
            self.list.currentItem().data(QtCore.Qt.UserRole) if self.list.currentItem() else None))

    def _on_select(self, cur, _prev) -> None:
        if cur is not None:
            self._load_into_editor(profiles.load(cur.data(QtCore.Qt.UserRole)))

    def _load_into_editor(self, prof: dict) -> None:
        self._working = prof
        builtin = bool(prof.get("builtin"))
        self.title.setText("Editing: " + profiles.label(prof)
                           + ("  (read-only baseline)" if builtin else ""))
        self.name_edit.setText("" if builtin else prof.get("name", ""))
        self.name_edit.setPlaceholderText("New profile name" if builtin else "")
        self._editor_set(prof.get("params", {}))
        # the built-in can be used as a starting point but not overwritten
        self.delete_enabled(not builtin)
        self._update_deviation()

    def delete_enabled(self, on: bool) -> None:
        self._editors["_btn_Delete"].setEnabled(on)

    # ------------------------------------------------------------------ #
    # deviation banner — current sample vs active profile
    # ------------------------------------------------------------------ #
    def _update_deviation(self) -> None:
        cur = None
        if hasattr(self.main, "_profile_widget_params"):
            try:
                cur = self.main._profile_widget_params()
            except Exception:                       # noqa: BLE001 — banner is best-effort
                cur = None
        act = profiles.active()
        if cur is None:
            self.deviation.setText("")
            return
        d = profiles.diff(cur, act)
        if not d:
            self.deviation.setText(f"✓ Open sample matches the active profile ({profiles.label(act)}).")
        else:
            parts = ", ".join(f"{k}: {v[1]} (vs {v[0]})" for k, v in list(d.items())[:6])
            more = "" if len(d) <= 6 else f" +{len(d) - 6} more"
            self.deviation.setText(
                f"⚠ Open sample deviates from {profiles.label(act)} — {parts}{more}. "
                "(Recorded in this sample's provenance.)")

    # ------------------------------------------------------------------ #
    # library actions
    # ------------------------------------------------------------------ #
    def _new(self) -> None:
        self._load_into_editor(profiles.make_profile("New profile", profiles.default_params()))
        self.title.setText("Editing: new profile")
        self.name_edit.setText("New profile")
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _duplicate(self) -> None:
        base = self._working
        self._working = profiles.make_profile(base.get("name", "Profile") + " copy",
                                              self._editor_get())
        self.title.setText("Editing: copy")
        self.name_edit.setText(self._working["name"])
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _save(self) -> None:
        name = self.name_edit.text().strip()
        if not name or name == profiles.BUILTIN_NAME:
            QtWidgets.QMessageBox.warning(
                self, "Name required",
                f"Enter a name other than '{profiles.BUILTIN_NAME}' (the read-only baseline).")
            return
        version = (max(profiles.versions_of(name)) + 1) if profiles.versions_of(name) else 1
        prof = profiles.make_profile(name, self._editor_get(), version=version)
        profiles.save(prof)
        self._working = prof
        self.main.statusBar().showMessage(f"Saved profile {profiles.label(prof)}.")
        self.refresh()

    def _set_active(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        profiles.set_active(it.data(QtCore.Qt.UserRole))
        self.main.statusBar().showMessage(f"Active profile: {profiles.label(profiles.active())}.")
        self.refresh()

    def _delete(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        name = it.data(QtCore.Qt.UserRole)
        if name == profiles.BUILTIN_NAME:
            return
        if confirm(self, "Delete profile",
                   f"Delete all versions of “{name}”? This can't be undone."):
            profiles.delete(name)
            self.refresh()

    def _import(self) -> None:
        path, _ = filedialogs.get_open_file_name(
            self, "Import analysis profile", "", "Profiles (*.json);;All (*)")
        if not path:
            return
        try:
            prof = profiles.import_from(path)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Import failed", f"Could not read profile:\n{e}")
            return
        if prof.get("name") == profiles.BUILTIN_NAME:
            prof["name"] = "Imported profile"
        profiles.save(prof)
        self.main.statusBar().showMessage(f"Imported profile {profiles.label(prof)}.")
        self._working = prof
        self.refresh()

    def _export(self) -> None:
        prof = profiles.load(self._working.get("name")) if not self._working.get("builtin") \
            else profiles.builtin()
        default = profiles._slug(prof.get("name", "profile")) + f"-v{prof.get('version', 1)}.json"
        path, _ = filedialogs.get_save_file_name(
            self, "Export analysis profile", default, "Profiles (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        profiles.export_to(prof, path)
        self.main.statusBar().showMessage(f"Exported {profiles.label(prof)}.")

    def _apply_to_sample(self) -> None:
        if not hasattr(self.main, "_apply_profile_params"):
            return
        self.main._apply_profile_params(self._editor_get())
        self.main.statusBar().showMessage(
            f"Applied {profiles.label(self._working)} to the current sample.")
        self._update_deviation()
