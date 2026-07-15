"""Analysis Script Console — write Python that drives the engine, run it, save it as a workflow.

The imperative companion to the Analysis Flow designer (``gui/flowdialog.py``): instead of
clicking declarative steps together, you write a short Python script against the
:class:`smile_msi.scripting.ScriptAPI` (bare-name functions like ``find_peaks`` /
``segment`` / ``compare``), run it on the loaded slide, and see tables + ion images + a log.
Scripts save as reusable *workflow* presets (``~/.smile-msi/workflows/``).

A built-in **AI guide** button emits :func:`smile_msi.scripting.capabilities_doc` — a
self-describing reference of every function, the analysis registry, the runtime context and
best practices — so an AI assistant can be handed the guide and write a correct workflow.

This module is purely the GUI; all analysis wiring + persistence lives in
:mod:`smile_msi.scripting`.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import imaging, scripting
from . import filedialogs
from .common import (ControlBar, RUN_GLYPH, button, fill_table, glossary_button,
                     menu_button, note, primary_button, section_title)

_MONO = "Menlo, Monaco, Consolas, 'DejaVu Sans Mono', monospace"
_MAX_TABLES = 12          # cap the surfaced tabs so a runaway loop can't spawn hundreds
_MAX_IMAGES = 12
_MAX_TABLE_ROWS = 2000    # render at most this many rows per table (note the rest)

_PLACEHOLDER = '''\
# Analysis workflow — Python against the loaded slide.
# Click "AI guide" for the full reference, or "Examples ▾" for starters.
peaks = find_peaks(snr=5)
log(f"{len(peaks)} peaks in the working set")
hits = annotate(mode="negative")
named = hits[hits["lipid"].astype(str) != ""]
table(named.head(20), "Lipid IDs")
'''


class ScriptConsoleMixin:
    """Lives on ``MainWindow``. Opens the lazily-built Script Console."""

    def _open_script_console(self):
        if self._script_console is None:
            self._script_console = ScriptConsoleDialog(self)
        else:
            self._script_console.refresh_context()
        self._show_dialog(self._script_console)


# --------------------------------------------------------------------------- #
# editor + syntax highlighting
# --------------------------------------------------------------------------- #
class _PyHighlighter(QtGui.QSyntaxHighlighter):
    """A small Python highlighter: keywords, the API bare names, strings, numbers, comments."""

    _KEYWORDS = ("and as assert break class continue def del elif else except finally for "
                 "from global if import in is lambda nonlocal not or pass raise return try "
                 "while with yield True False None").split()

    def __init__(self, doc):
        super().__init__(doc)
        def fmt(color, bold=False, italic=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Bold)
            f.setFontItalic(italic)
            return f
        # Theme-aware syntax palette (chosen from the active app theme at build time): the
        # dark scheme is the familiar VSCode-dark; the light scheme uses darker, saturated
        # inks so keywords/strings/numbers stay legible on the cool-slate code editor instead
        # of washing out. (Re-open the console after a live Light/Dark switch to repaint.)
        app = QtWidgets.QApplication.instance()
        base = app.palette().color(QtGui.QPalette.Base) if app else QtGui.QColor("#15171a")
        if base.lightnessF() < 0.5:                 # dark editor
            kw, api, num, stc, com = "#C586C0", "#4EC9B0", "#B5CEA8", "#CE9178", "#6A9955"
        else:                                       # light cool-slate editor
            kw, api, num, stc, com = "#8839BE", "#1F7A7A", "#0E7C3A", "#B5491B", "#5E7A52"
        self._kw = fmt(kw, bold=True)
        self._api = fmt(api)
        self._num = fmt(num)
        self._str = fmt(stc)
        self._com = fmt(com, italic=True)
        import re
        word = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
        self._word = word
        self._num_re = re.compile(r"\b[0-9][0-9_.eE+-]*\b")
        self._str_re = re.compile(r"(\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*')")
        self._api_names = set(scripting.ScriptAPI.NAMESPACE_FUNCS) | {
            "log", "table", "image", "record", "api", "ds", "np", "active_mz"}

    def highlightBlock(self, text):
        for m in self._num_re.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self._num)
        for m in self._word.finditer(text):
            w = m.group()
            if w in self._KEYWORDS:
                self.setFormat(m.start(), len(w), self._kw)
            elif w in self._api_names:
                self.setFormat(m.start(), len(w), self._api)
        for m in self._str_re.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self._str)
        h = text.find("#")
        if h >= 0:                                 # naive: '#' inside a string is rare here
            self.setFormat(h, len(text) - h, self._com)


class _CodeEditor(QtWidgets.QPlainTextEdit):
    """A monospace editor with soft tabs, indent-aware Enter, and ⌘/Ctrl+Return to run."""

    run_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QtGui.QFont(_MONO.split(",")[0].strip())
        f.setStyleHint(QtGui.QFont.Monospace)
        f.setPointSize(12)
        self.setFont(f)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setPlaceholderText("Write a workflow… (Examples ▾ for starters, AI guide for the reference)")

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and \
                (e.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)):
            self.run_requested.emit()
            return
        if e.key() == QtCore.Qt.Key_Tab:
            self.insertPlainText("    ")
            return
        if e.key() == QtCore.Qt.Key_Backtab:           # Shift+Tab — dedent the current line
            cur = self.textCursor()
            cur.movePosition(QtGui.QTextCursor.StartOfLine)
            cur.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor, 4)
            if cur.selectedText() == "    ":
                cur.removeSelectedText()
            return
        if e.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            line = self.textCursor().block().text()
            indent = line[:len(line) - len(line.lstrip())]
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(e)
            self.insertPlainText(indent)
            return
        super().keyPressEvent(e)


# --------------------------------------------------------------------------- #
# the console dialog
# --------------------------------------------------------------------------- #
class ScriptConsoleDialog(QtWidgets.QDialog):
    """Modeless console: edit + run a workflow on the loaded slide, save/load presets."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Analysis script")
        self.setModal(False)
        self.setMinimumSize(900, 680)
        self._running = False
        self._last_result = None
        self._wf_name = "Workflow"
        self._wf_desc = ""
        self._build()
        self.editor.setPlainText(_PLACEHOLDER)
        self.refresh_context()

    # ------------------------------------------------------------------ #
    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        bar = ControlBar()
        self.bar = bar
        self._wf_btn = menu_button("Workflows", [
            ("New (clear)", self._new),
            None,
            ("Save", self._save),
            ("Save as…", self._save_as),
            ("Load…", self._load),
            None,
            ("Import from file…", self._import),
            ("Export to file…", self._export),
            ("Delete…", self._delete),
        ], tooltip="Save / load this script as a reusable workflow preset.")
        self._ex_btn = menu_button("Examples", tooltip="Insert a starter script.")
        for name, code in scripting.EXAMPLES.items():
            self._ex_btn.menu().addAction(name, lambda _=False, c=code: self._insert_example(c))
        guide_btn = button("AI guide", self._show_guide,
                           tooltip="Show / copy the scripting reference to hand to an AI (or read yourself).",
                           icon="help")
        bar.add(self._wf_btn, self._ex_btn, guide_btn)
        bar.add(glossary_button([
            ["Workflow", "A Python script you write once, run on any slide, and save as a preset."],
            ["Feature set", "The working m/z list analyses run on — produced by find_peaks()."],
            ["Region / Group", "Named ROIs (and their A/B/… group tags) drawn in the app, "
                               "usable from a script via region('name') / group('A')."],
            ["AI guide", "A self-describing reference of every function + the runtime context, "
                         "to hand to an AI so it can write a correct script."],
        ], parent=self))
        outer.addWidget(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        # ---- editor ---------------------------------------------------- #
        ed_wrap = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(ed_wrap)
        ev.setContentsMargins(0, 0, 0, 0)
        ev.setSpacing(3)
        self._title_lbl = section_title("Workflow")
        ev.addWidget(self._title_lbl)
        self.editor = _CodeEditor()
        self.editor.run_requested.connect(self._run)
        _PyHighlighter(self.editor.document())
        ev.addWidget(self.editor, 1)
        self._ctx_note = note("")
        ev.addWidget(self._ctx_note)
        split.addWidget(ed_wrap)

        # ---- output tabs ---------------------------------------------- #
        self.out_tabs = QtWidgets.QTabWidget()
        self.out_tabs.setDocumentMode(True)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        lf = QtGui.QFont(_MONO.split(",")[0].strip())
        lf.setStyleHint(QtGui.QFont.Monospace)
        self.log_view.setFont(lf)
        self.out_tabs.addTab(self.log_view, "Log")
        split.addWidget(self.out_tabs)
        split.setSizes([420, 240])
        outer.addWidget(split, 1)

        # ---- footer ---------------------------------------------------- #
        foot = QtWidgets.QHBoxLayout()
        self.b_run = primary_button(f"{RUN_GLYPH} Run", self._run,
                                    tooltip="Run the script on the loaded slide (⌘/Ctrl+Return)")
        foot.addWidget(self.b_run)
        self.b_cancel = button("Cancel", self._cancel)
        self.b_cancel.setEnabled(False)
        foot.addWidget(self.b_cancel)
        self._apply_btn = menu_button("Apply to app", [
            ("Set as working features", self._apply_features),
            ("Set as segmentation", self._apply_segmentation),
        ], tooltip="Push the script's features / segmentation into the app's views.", name="menu")
        self._apply_btn.setEnabled(False)
        foot.addWidget(self._apply_btn)
        self._status = note("")
        foot.addWidget(self._status, 1)
        foot.addWidget(button("Close", self.close))
        outer.addLayout(foot)

    # ------------------------------------------------------------------ #
    # context banner
    # ------------------------------------------------------------------ #
    def refresh_context(self):
        """Update the one-line slide/regions banner under the editor (call on show)."""
        win = self.win
        if win.ds is None:
            self._ctx_note.setText("No slide loaded — open a sample first.")
            return
        regions = [rg.get("name", "?") for rg in (win.regions or [])]
        groups = sorted({(rg.get("group") or "").strip() for rg in (win.regions or [])} - {""})
        feats = len(win.peaks or [])
        self._ctx_note.setText(
            f"Slide: {os.path.basename(str(win.ds.source))} · {win.ds.n_pixels:,} px · "
            f"ppm {win.ppm:g} · norm {win.norm} · {feats} app features · "
            f"regions: {', '.join(regions) or '—'} · groups: {', '.join(groups) or '—'}")

    # ------------------------------------------------------------------ #
    # build the bound API from the live app state
    # ------------------------------------------------------------------ #
    def _build_api(self):
        win = self.win
        masks, groups = {}, {}
        for rg in (win.regions or []):
            m = win._region_pixel_mask(rg)
            if m is None:
                continue
            m = np.asarray(m, dtype=bool)
            masks[rg.get("name", f"region {len(masks)}")] = m
            g = (rg.get("group") or "").strip()
            if g:
                groups[g] = (groups[g] | m) if g in groups else m.copy()
        return scripting.ScriptAPI(win.ds, ppm=win.ppm, norm=win.norm, reduce=win.reduce,
                                   masks=masks, groups=groups,
                                   features=[float(p["mz"]) for p in (win.peaks or [])],
                                   active_mz=win.active_mz)

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    def _run(self):
        win = self.win
        if win.ds is None:
            win.statusBar().showMessage("Load a dataset first.")
            return
        if self._running:
            return
        code = self.editor.toPlainText()
        if not code.strip():
            self._status.setText("Nothing to run — write a script (or pick an example).")
            return
        api = self._build_api()
        self._running = True
        self._set_running(True)
        self._status.setText("Running…")
        win._run(lambda: scripting.run_script(code, api),
                 on_done=self._on_done, busy="Running workflow…")

    def _on_done(self, result):
        self._running = False
        self._set_running(False)
        self._last_result = result
        self._render(result)
        self._apply_btn.setEnabled(bool(result.peaks) or result.segmentation is not None)
        self.win.statusBar().showMessage(
            ("Workflow failed — see the Log tab." if not result.ok
             else f"Workflow ran — {result.summary()}."))
        if result.ok and not result.error and not result.tables and not result.images:
            self._status.setText("Completed — no output. Use log() / table() / image() "
                                 "to surface results.")
        else:
            self._status.setText(result.summary())

    def _cancel(self):
        try:
            self.win._cancel = True
        except Exception:  # noqa: BLE001 — best-effort cancel signal; window may be tearing down
            pass
        self._status.setText("Cancelling…")

    def _set_running(self, on):
        self.b_run.setEnabled(not on)
        self.b_cancel.setEnabled(on)
        self.editor.setReadOnly(on)

    def keyPressEvent(self, e):
        # Escape closes the modeless console (routing through closeEvent so a mid-run close
        # still signals cancel) instead of QDialog's silent reject().
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        # Closing while a workflow is running must signal the worker to stop and restore the
        # Run/Cancel/editor state — otherwise the dialog reopens with the buttons stuck and a
        # background run still going.
        if self._running:
            self._cancel()
            self._set_running(False)
            self._running = False
        super().closeEvent(e)

    # ------------------------------------------------------------------ #
    # render results
    # ------------------------------------------------------------------ #
    def _render(self, result):
        # rebuild tabs: Log first (kept), then a tab per table + per image
        while self.out_tabs.count() > 1:
            w = self.out_tabs.widget(1)
            self.out_tabs.removeTab(1)
            w.deleteLater()
        lines = []
        if result.logs:
            lines += list(result.logs)
        if result.stdout.strip():
            lines += ["", "── stdout ──", result.stdout.rstrip()]
        if result.values:
            lines += ["", "── recorded values ──"]
            lines += [f"{k} = {v!r}" for k, v in result.values.items()]
        if result.error:
            lines += ["", "── ERROR ──", result.error.rstrip()]
        if not lines:
            lines = ["(ran with no output — use log()/table()/image() to surface results)"]
        self.log_view.setPlainText("\n".join(lines))
        if result.error:
            self.out_tabs.setCurrentIndex(0)

        for ti, (title, df) in enumerate(result.tables[:_MAX_TABLES]):
            try:
                widget = self._table_widget(df)
            except Exception as exc:                       # a malformed frame must not kill the render
                widget = note(f"(could not render this table: {exc})")
            self.out_tabs.addTab(widget, title or f"Table {ti + 1}")
        if len(result.tables) > _MAX_TABLES:
            self.log_view.appendPlainText(f"\n(+{len(result.tables) - _MAX_TABLES} more table(s) not shown)")
        for ii, (title, arr, _meta) in enumerate(result.images[:_MAX_IMAGES]):
            try:
                widget = self._image_widget(arr)
            except Exception as exc:                       # a bad array / OOM must not kill the render
                widget = note(f"(could not render this image: {exc})")
            self.out_tabs.addTab(widget, title or f"Image {ii + 1}")
        if len(result.images) > _MAX_IMAGES:
            self.log_view.appendPlainText(f"\n(+{len(result.images) - _MAX_IMAGES} more image(s) not shown)")
        if (result.tables or result.images) and not result.error:
            self.out_tabs.setCurrentIndex(1)

    def _table_widget(self, df):
        t = QtWidgets.QTableWidget()
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSortingEnabled(True)
        headers = [str(c) for c in df.columns]
        shown = df.head(_MAX_TABLE_ROWS)
        rows = []
        for _, r in shown.iterrows():
            rows.append([(f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r.tolist()])
        fill_table(t, headers, rows)
        from .common import install_table_export
        install_table_export(t, self, stem="workflow_table", title="Export table")
        return t

    def _image_widget(self, arr):
        wrap = QtWidgets.QScrollArea()
        wrap.setWidgetResizable(True)
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        cmap = getattr(self.win, "cmap", None) or (
            self.win.cmap_combo.currentText() if getattr(self.win, "cmap_combo", None) else "viridis")
        rgba = imaging.apply_colormap(np.nan_to_num(np.asarray(arr, dtype=float)), cmap=cmap,
                                      low=0.0, high=99.5)
        h, w = rgba.shape[:2]
        qimg = QtGui.QImage(rgba.tobytes(), w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        pm = QtGui.QPixmap.fromImage(qimg).scaled(420, 420, QtCore.Qt.KeepAspectRatio,
                                                  QtCore.Qt.FastTransformation)
        lbl.setPixmap(pm)
        wrap.setWidget(lbl)
        return wrap

    # ------------------------------------------------------------------ #
    # apply results to the app
    # ------------------------------------------------------------------ #
    def _apply_features(self):
        r = self._last_result
        if r is None or not r.peaks:
            self.win.statusBar().showMessage("No features from the last run.")
            return
        self.win._pending_pick_region = None
        self.win._pending_pick_save = False
        self.win._on_peaks(list(r.peaks))
        self.win.statusBar().showMessage(f"Applied {len(r.peaks)} features to the app.")

    def _apply_segmentation(self):
        r = self._last_result
        if r is None or r.segmentation is None:
            self.win.statusBar().showMessage("No segmentation from the last run.")
            return
        self.win._on_seg(r.segmentation)
        self.win.reveal_view("Segmentation")
        self.win.statusBar().showMessage(
            f"Applied segmentation ({int(r.segmentation.n_clusters)} clusters) to the app.")

    # ------------------------------------------------------------------ #
    # AI guide
    # ------------------------------------------------------------------ #
    def _show_guide(self):
        try:
            api = self._build_api() if self.win.ds is not None else None
            text = scripting.capabilities_doc(api)
        except Exception as exc:                           # the guide button must always yield text
            text = (f"Could not build the scripting guide: {exc}\n\n"
                    "This is unexpected — please report it. You can still write a script "
                    "against the bare-name API (find_peaks, segment, compare, …).")
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Scripting guide — for an AI (or you)")
        dlg.setMinimumSize(720, 640)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(note("Hand this to an AI assistant and ask it to write a workflow, or "
                         "read it yourself. It reflects the current slide."))
        view = QtWidgets.QPlainTextEdit()
        view.setReadOnly(True)
        f = QtGui.QFont(_MONO.split(",")[0].strip())
        f.setStyleHint(QtGui.QFont.Monospace)
        view.setFont(f)
        view.setPlainText(text)
        v.addWidget(view, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(button("Copy to clipboard",
                             lambda: (QtWidgets.QApplication.clipboard().setText(text),
                                      self.win.statusBar().showMessage("Scripting guide copied.")),
                             icon="copy"))
        row.addWidget(button("Save as Markdown…", lambda: self._save_guide(text), icon="export"))
        row.addStretch(1)
        row.addWidget(button("Close", dlg.accept))
        v.addLayout(row)
        dlg.exec()

    def _save_guide(self, text):
        path, _ = filedialogs.get_save_file_name(self, "Save scripting guide",
                                                 "SCRIPTING_GUIDE.md", "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.win.statusBar().showMessage(f"Guide saved: {os.path.basename(path)}")

    # ------------------------------------------------------------------ #
    # examples + workflow presets
    # ------------------------------------------------------------------ #
    def _insert_example(self, code):
        if self.editor.toPlainText().strip() and self.editor.toPlainText().strip() != _PLACEHOLDER.strip():
            from .common import confirm
            if not confirm(self, "Replace script?",
                           "Replace the current script with this example?", danger=False,
                           ok_text="Replace"):
                return
        self.editor.setPlainText(code)

    def _new(self):
        from .common import confirm
        if self.editor.toPlainText().strip() and not confirm(
                self, "New workflow?", "Clear the editor and start a new workflow?",
                danger=False, ok_text="Clear"):
            return
        self._wf_name, self._wf_desc = "Workflow", ""
        self.editor.setPlainText("")
        self._title_lbl.setText("Workflow")

    def _current_workflow(self):
        return scripting.Workflow(name=self._wf_name, code=self.editor.toPlainText(),
                                  description=self._wf_desc)

    def _save(self):
        if self._wf_name and self._wf_name != "Workflow":
            path = self._current_workflow().save()
            self.win.statusBar().showMessage(f"Workflow saved: {os.path.basename(path)}")
        else:
            self._save_as()

    def _save_as(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save workflow", "Workflow name:",
                                                  text=self._wf_name)
        if not ok or not name.strip():
            return
        desc, _ = QtWidgets.QInputDialog.getText(self, "Save workflow",
                                                 "One-line description (optional):", text=self._wf_desc)
        self._wf_name = name.strip()
        self._wf_desc = desc.strip()
        path = self._current_workflow().save()
        self._title_lbl.setText(self._wf_name)
        self.win.statusBar().showMessage(f"Workflow saved: {os.path.basename(path)}")

    def _load(self):
        wfs = scripting.list_workflows()
        if not wfs:
            self.win.statusBar().showMessage("No saved workflows yet — Save as… first.")
            return
        labels = [(f"{w['name']} — {w['description']}" if w["description"] else w["name"]) for w in wfs]
        label, ok = QtWidgets.QInputDialog.getItem(self, "Load workflow", "Workflow:", labels, 0, False)
        if not ok:
            return
        w = wfs[labels.index(label)]
        self._apply_loaded(scripting.Workflow.load(w["path"]))

    def _import(self):
        path, _ = filedialogs.get_open_file_name(self, "Import workflow", "", "Workflow (*.json)")
        if path:
            try:
                self._apply_loaded(scripting.Workflow.load(path))
            except Exception as exc:
                self.win.statusBar().showMessage(f"Could not read workflow: {exc}")

    def _export(self):
        path, _ = filedialogs.get_save_file_name(self, "Export workflow",
                                                 f"{self._wf_name}.json", "Workflow (*.json)")
        if path:
            self._current_workflow().save(path)
            self.win.statusBar().showMessage(f"Workflow exported: {os.path.basename(path)}")

    def _delete(self):
        wfs = scripting.list_workflows()
        if not wfs:
            return
        names = [w["name"] for w in wfs]
        name, ok = QtWidgets.QInputDialog.getItem(self, "Delete workflow", "Workflow:", names, 0, False)
        if not ok:
            return
        from .common import confirm
        if not confirm(self, "Delete workflow?", f"Delete '{name}'? This cannot be undone."):
            return
        try:
            os.remove(wfs[names.index(name)]["path"])
            self.win.statusBar().showMessage(f"Deleted workflow '{name}'.")
        except OSError as exc:
            self.win.statusBar().showMessage(f"Could not delete: {exc}")

    def _apply_loaded(self, wf):
        self._wf_name = wf.name
        self._wf_desc = wf.description
        self.editor.setPlainText(wf.code)
        self._title_lbl.setText(wf.name)
        self.win.statusBar().showMessage(f"Loaded workflow '{wf.name}'.")
