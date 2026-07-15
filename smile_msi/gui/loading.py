"""Modal loading window shown while a sample's data streams in and the fast ion
cache (m/z cube) builds.

Selecting a sample reads the ``.ibd`` into RAM, primes it, and builds the sparse
m/z cube so every later interaction is instant — a heavy pass that runs on a
worker thread. This window blocks interaction with the main window for the
duration (so a half-loaded sample can't be poked at) while staying responsive:
the worker drives the bar via :meth:`set_progress`, and the event loop keeps
running, so the optional Cancel button works.

Driven from ``MainWindow._run(..., modal=True)``; see ``_close_load_dialog``.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .common import ACCENT, MUTED_QSS

# Margin reserved around the card so the drop shadow has room to render (the
# window itself is translucent; only the card paints).
_SHADOW_MARGIN = 18


class LoadingWindow(QtWidgets.QDialog):
    """App-modal, frameless progress window.

    The bar starts indeterminate (busy marquee) and becomes determinate the
    moment the first ``set_progress(done, total)`` arrives — so it animates
    immediately even before the worker reports its first chunk.
    """

    def __init__(self, parent=None, message="Loading sample…", cancelable=True,
                 title="Loading sample…", stages=None):
        super().__init__(parent, QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        # Translucent window so the card can float with a soft drop shadow instead
        # of reading as a flat rectangle pasted over the workspace.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setFixedWidth(460 + 2 * _SHADOW_MARGIN)
        self._cancel_cb = None

        card = QtWidgets.QFrame(self)
        card.setObjectName("loadCard")
        card.setStyleSheet(
            "#loadCard{background:palette(window);"
            "border:1px solid palette(mid);border-radius:12px;}"
        )
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(34)
        shadow.setOffset(0, 6)
        shadow.setColor(QtGui.QColor(0, 0, 0, 110))
        card.setGraphicsEffect(shadow)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(_SHADOW_MARGIN, _SHADOW_MARGIN,
                                 _SHADOW_MARGIN, _SHADOW_MARGIN)
        outer.addWidget(card)

        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(8)

        title_lbl = QtWidgets.QLabel(title)
        f = title_lbl.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        title_lbl.setFont(f)
        lay.addWidget(title_lbl)

        self._detail = QtWidgets.QLabel()
        self._detail.setStyleSheet(MUTED_QSS)
        self._detail.setWordWrap(True)
        lay.addWidget(self._detail)

        # Optional stage checklist: one row per named sub-step ("Reading spectra…" →
        # "Priming…"), so the card narrates *which* pass is running, not just a bar
        # crawling with no context. ``set_stage(name)`` marks earlier rows done (✓),
        # the current row active (▸), and the rest pending (○).
        self._stage_order = list(stages) if stages else []
        self._stage_rows = {}
        if self._stage_order:
            stage_box = QtWidgets.QVBoxLayout()
            stage_box.setSpacing(3)
            stage_box.setContentsMargins(0, 4, 0, 2)
            for sname in self._stage_order:
                row = QtWidgets.QHBoxLayout()
                row.setSpacing(8)
                icon = QtWidgets.QLabel("○")
                icon.setFixedWidth(14)
                icon.setStyleSheet(MUTED_QSS)
                text = QtWidgets.QLabel(sname)
                text.setStyleSheet(MUTED_QSS)
                row.addWidget(icon)
                row.addWidget(text)
                row.addStretch(1)
                stage_box.addLayout(row)
                self._stage_rows[sname] = (icon, text)
            lay.addLayout(stage_box)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)              # indeterminate until the first chunk lands
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(
            "QProgressBar{background:palette(midlight);border:none;border-radius:3px;}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}"
        )
        lay.addWidget(self._bar)

        if cancelable:
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            self._cancel_btn = QtWidgets.QPushButton("Cancel")
            self._cancel_btn.clicked.connect(self._on_cancel_clicked)
            row.addWidget(self._cancel_btn)
            lay.addLayout(row)

        self.set_message(message)
        self.adjustSize()

    # -- placement --------------------------------------------------------- #
    def showEvent(self, e):                  # noqa: N802 (Qt override)
        """Centre the card over the parent window (or screen) the first time it
        shows, so it lands where the user is looking instead of an arbitrary
        corner the window manager picks for a frameless dialog."""
        super().showEvent(e)
        par = self.parentWidget()
        if par:
            ref = par.frameGeometry()
        else:
            scr = self.screen() or QtWidgets.QApplication.primaryScreen()
            ref = scr.availableGeometry()
        self.move(ref.center() - self.rect().center())

    # -- caption ----------------------------------------------------------- #
    def set_message(self, message: str):
        """Set the detail line, eliding overly long paths to fit the card."""
        fm = self._detail.fontMetrics()
        avail = self.width() - 2 * _SHADOW_MARGIN - 60
        self._detail.setText(fm.elidedText(message, QtCore.Qt.ElideMiddle, max(avail, 80)))

    # -- stage narration --------------------------------------------------- #
    def set_stage(self, name: str):
        """Advance the stage checklist to ``name``: rows before it read done (✓),
        it reads active (▸), rows after it stay pending (○). A stage name not in the
        list (or when no list was given) falls back to the detail caption, so a stray
        or ad-hoc stage still shows rather than vanishing."""
        if name not in self._stage_rows:
            self.set_message(name)
            return
        idx = self._stage_order.index(name)
        for i, sname in enumerate(self._stage_order):
            icon, text = self._stage_rows[sname]
            if i < idx:
                icon.setText("✓")
                icon.setStyleSheet(f"color:{ACCENT};")
                text.setStyleSheet(MUTED_QSS)
            elif i == idx:
                icon.setText("▸")
                icon.setStyleSheet(f"color:{ACCENT};font-weight:bold;")
                text.setStyleSheet(f"color:{ACCENT};font-weight:bold;")
            else:
                icon.setText("○")
                icon.setStyleSheet(MUTED_QSS)
                text.setStyleSheet(MUTED_QSS)

    # -- progress ---------------------------------------------------------- #
    def set_progress(self, done: int, total: int):
        if total and total > 0:
            if self._bar.maximum() == 0:      # leave the indeterminate marquee
                self._bar.setRange(0, 100)
            self._bar.setValue(max(0, min(100, int(100 * done / total))))

    # -- cancel ------------------------------------------------------------ #
    def on_cancel(self, callback):
        """Register a callback fired when the user clicks Cancel."""
        self._cancel_cb = callback

    def _on_cancel_clicked(self):
        if hasattr(self, "_cancel_btn"):
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling…")
        if self._cancel_cb is not None:
            self._cancel_cb()

    # Frameless modal: intercept Esc so a stray keypress can't reject the dialog
    # (QDialog's default) out from under the running worker. Esc routes to Cancel
    # when cancelable; otherwise it's swallowed. The window is torn down only by
    # MainWindow when the worker hits a terminal signal — never call super() here,
    # so the programmatic close()/deleteLater path stays the only way out.
    def keyPressEvent(self, e):              # noqa: N802 (Qt override)
        if e.key() == QtCore.Qt.Key_Escape:
            if self._cancel_cb is not None:
                self._on_cancel_clicked()
            e.ignore()
            return
        super().keyPressEvent(e)
