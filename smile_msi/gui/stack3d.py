"""3D serial-section reconstruction (Data ▸ 3D reconstruction…).

Stack the samples of the current cohort into an ordered, spacing-aware
:class:`smile_msi.volume3d.SectionStack`, register each section into a common
``(x, y)`` frame, and resample one chosen ion into a ``(Z, H, W)``
:class:`smile_msi.volume3d.IonVolume`. The volume is shown as triplanar
orthoslices (axial / coronal / sagittal) with a z slider and a NaN-safe
maximum-intensity-projection toggle. An optional interactive GL volume render is
a button that lazy-imports ``pyqtgraph[opengl]`` (the ``viz3d`` extra) and shows a
clear error when it is absent — the numpy MIP path is the offline, CPU-only default.

Drives :mod:`smile_msi.volume3d`. The build streams sections one cube at a time via
a :class:`smile_msi.cohort.SectionLoader` (RAM bounded to ~one resident cube), and a
successful build records a ``volume3d`` audit step on the host.

The dialog reads only what it needs off the ``main`` host (see ``load_from_main``):
``main.cohort`` (the roster), ``main.peaks`` / ``main.active_mz`` (ion choices),
``main.statusBar()`` and the optional ``main.record_step`` audit hook. The menu /
opener wiring is the integrator's job.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .common import (dark_image_view, icon_button, note, primary_button, section_title,
                     NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSlider)

_REFERENCES = ("previous", "first", "centroid")
_MODES = ("rigid", "affine", "deformable")


class Stack3DDialog(QtWidgets.QDialog):
    """Build and view a 3D molecular volume from the cohort's serial sections."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("3D reconstruction — serial sections")
        self.resize(940, 720)

        self._refs: list = []                 # SampleRef per ordering-list row (parallel)
        self._stack = None                    # registered SectionStack (after Register)
        self._volume = None                   # built IonVolume (after Build)
        self._mip = False                     # MIP toggle state
        self._views: dict[str, object] = {}   # pane key -> pyqtgraph ImageItem
        self._vbs: dict[str, object] = {}     # pane key -> ViewBox

        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(self._build_controls(), 0)
        root.addLayout(self._build_viewer(), 1)

    # ---- left controls --------------------------------------------------- #
    def _build_controls(self) -> QtWidgets.QVBoxLayout:
        col = QtWidgets.QVBoxLayout()
        col.addWidget(section_title("Serial-section stack"))
        col.addWidget(note(
            "Order the cohort's samples by cutting depth (first cut at top), set the "
            "section spacing, register them into a common frame, then reconstruct one "
            "ion into a 3D volume. Build streams one slide at a time."))

        # ordering list (reorderable) -------------------------------------- #
        self.order_list = QtWidgets.QListWidget()
        self.order_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.order_list.setToolTip("Sections from first cut (top) to last (bottom). "
                                   "Use the Up / Down buttons to reorder.")
        col.addWidget(self.order_list, 1)

        move_row = QtWidgets.QHBoxLayout()
        move_row.addWidget(icon_button("up", "Up", self._move_up, tooltip="Move section earlier"))
        move_row.addWidget(icon_button("down", "Down", self._move_down,
                                       tooltip="Move section later"))
        move_row.addStretch(1)
        col.addLayout(move_row)

        # stack parameters ------------------------------------------------- #
        form = QtWidgets.QFormLayout()
        self.spacing = NoScrollDoubleSpinBox()
        self.spacing.setRange(0.0, 1e6)
        self.spacing.setDecimals(2)
        self.spacing.setValue(10.0)
        self.spacing.setSuffix(" µm")
        self.spacing.setToolTip("Nominal inter-section spacing (0 = irregular cuts).")
        form.addRow("Section spacing", self.spacing)

        self.reference = NoScrollComboBox()
        self.reference.addItems(_REFERENCES)
        self.reference.setToolTip("Which section is the fixed registration frame.")
        form.addRow("Reference", self.reference)

        self.mode = NoScrollComboBox()
        self.mode.addItems(_MODES)
        self.mode.setToolTip("rigid = phase-correlation (default, CPU). affine / deformable "
                             "need the registration backend.")
        form.addRow("Registration", self.mode)

        self.channel = NoScrollComboBox()
        self.channel.setEditable(True)
        self.channel.setToolTip("The ion m/z to reconstruct into a volume "
                                "(pick a peak or type an m/z).")
        form.addRow("Ion m/z", self.channel)

        self.anchor = NoScrollComboBox()
        self.anchor.setEditable(True)
        self.anchor.setToolTip("Anatomical-anchor ion to register on. "
                               "(TIC) registers on total ion current.")
        form.addRow("Register on", self.anchor)
        col.addLayout(form)

        # actions ---------------------------------------------------------- #
        act = QtWidgets.QHBoxLayout()
        self.b_register = icon_button("run", "Register", self._register,
                                      tooltip="Align every section to the reference frame")
        self.b_build = primary_button("Build volume", self._build,
                                      tooltip="Resample the chosen ion into a (Z, H, W) volume")
        act.addWidget(self.b_register)
        act.addStretch(1)
        act.addWidget(self.b_build)
        col.addLayout(act)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        col.addWidget(self.status)
        col.setStretch(2, 1)
        return col

    # ---- right viewer ---------------------------------------------------- #
    def _build_viewer(self) -> QtWidgets.QVBoxLayout:
        import pyqtgraph as pg

        col = QtWidgets.QVBoxLayout()
        col.addWidget(section_title("Orthoslices"))

        self.view = pg.GraphicsLayoutWidget()
        dark_image_view(self.view)            # ion canvas reads against black in both themes
        for key, (r, c, label) in {
            "axial": (0, 0, "Axial (z)"),
            "coronal": (0, 1, "Coronal (y)"),
            "sagittal": (1, 0, "Sagittal (x)"),
        }.items():
            self.view.addLabel(label, row=2 * r, col=c, size="8pt")
            vb = self.view.addViewBox(row=2 * r + 1, col=c)
            vb.invertY(True)
            vb.setAspectLocked(True)
            it = pg.ImageItem()
            vb.addItem(it)
            self._views[key] = it
            self._vbs[key] = vb
        col.addWidget(self.view, 1)

        # z slider + toggles ----------------------------------------------- #
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("z"))
        self.z_slider = NoScrollSlider(QtCore.Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.setEnabled(False)
        self.z_slider.valueChanged.connect(self._refresh_slices)
        ctl.addWidget(self.z_slider, 1)
        self.z_label = QtWidgets.QLabel("—")
        ctl.addWidget(self.z_label)

        self.mip_toggle = QtWidgets.QCheckBox("MIP")
        self.mip_toggle.setToolTip("Show the maximum-intensity projection instead of single slices.")
        self.mip_toggle.toggled.connect(self._on_mip)
        ctl.addWidget(self.mip_toggle)

        self.b_gl = icon_button("run", "Volume render…", self._gl_render,
                                tooltip="Interactive 3D render (needs the viz3d extra)")
        self.b_gl.setEnabled(False)
        ctl.addWidget(self.b_gl)
        col.addLayout(ctl)
        return col

    # ---- populate from the cohort ---------------------------------------- #
    def load_from_main(self):
        """Fill the ordering list from ``main.cohort`` and the ion combos from the peaks."""
        cohort = getattr(self.main, "cohort", None)
        samples = list(getattr(cohort, "samples", []) or [])
        self._refs = list(samples)
        self.order_list.clear()
        for r in samples:
            label = getattr(r, "name", "") or getattr(r, "key", lambda: "?")()
            grp = getattr(r, "group", "") or ""
            self.order_list.addItem(f"{label}" + (f"   [{grp}]" if grp else ""))
        if samples:
            self.order_list.setCurrentRow(0)

        mzs = [f"{float(p['mz']):.4f}" for p in (getattr(self.main, "peaks", []) or [])
               if p.get("mz") is not None]
        self.channel.clear()
        self.channel.addItems(mzs)
        self.anchor.clear()
        self.anchor.addItem("(TIC)")
        self.anchor.addItems(mzs)
        active = getattr(self.main, "active_mz", None)
        if active is not None:
            self.channel.setEditText(f"{float(active):.4f}")

        self._stack = None
        self._volume = None
        self._set_status(f"{len(samples)} sample(s) in the cohort. "
                         "Order them, then Register and Build."
                         if samples else "The cohort is empty — add samples first.")

    # ---- ordering helpers ------------------------------------------------ #
    def _move_up(self):
        self._swap(self.order_list.currentRow(), -1)

    def _move_down(self):
        self._swap(self.order_list.currentRow(), +1)

    def _swap(self, row: int, delta: int):
        j = row + delta
        if row < 0 or j < 0 or j >= self.order_list.count():
            return
        self._refs[row], self._refs[j] = self._refs[j], self._refs[row]
        it = self.order_list.takeItem(row)
        self.order_list.insertItem(j, it)
        self.order_list.setCurrentRow(j)
        self._stack = None                    # ordering changed → re-register

    # ---- engine helpers -------------------------------------------------- #
    def _ordered_refs(self) -> list:
        """Refs in the current list order (parallel to the list rows)."""
        return list(self._refs)

    def _make_loader(self, refs):
        """A primed :class:`SectionLoader` with a ``refs_by_key`` map so the engine can
        resolve a :class:`StackSection.ref_key` back to its :class:`SampleRef`."""
        from .. import cohort as cohort_engine

        loader = cohort_engine.SectionLoader(dense=False)
        loader.prime(refs)
        loader.refs_by_key = {r.key(): r for r in refs}
        return loader

    def _build_stack_model(self, refs):
        from .. import volume3d

        keys = [r.key() for r in refs]
        return volume3d.make_stack(
            self._stack_name(), keys,
            spacing_um=float(self.spacing.value()),
            reference=self.reference.currentText(),
            mode=self.mode.currentText())

    def _stack_name(self) -> str:
        cohort = getattr(self.main, "cohort", None)
        base = getattr(cohort, "name", "") or "Cohort"
        return f"{base} - 3D stack"          # " - " not "·" (naming rule)

    def _mz_value(self, combo, *, allow_tic=False):
        txt = combo.currentText().strip()
        if allow_tic and (txt == "" or txt.lower().startswith("(tic")):
            return None
        try:
            return float(txt)
        except (TypeError, ValueError):
            return None

    # ---- actions --------------------------------------------------------- #
    def _register(self):
        from .. import volume3d

        refs = self._ordered_refs()
        if len(refs) < 2:
            self._set_status("Need at least two sections to register a stack.")
            return
        stack = self._build_stack_model(refs)
        loader = self._make_loader(refs)
        anchor = self._mz_value(self.anchor, allow_tic=True)
        try:
            self._stack = volume3d.register_stack(
                stack, loader, channel_mz=anchor, random_state=0)
        except Exception as exc:  # noqa: BLE001 — surface the engine error, never crash the UI
            self._set_status(f"Registration failed: {exc}")
            return
        finally:
            loader.close()
        n = sum(1 for s in self._stack.sections if s.transform is not None)
        self._set_status(f"Registered {n} of {len(refs)} section(s) "
                         f"({self.mode.currentText()}, reference = {self.reference.currentText()}).")

    def _build(self):
        from .. import volume3d

        refs = self._ordered_refs()
        if not refs:
            self._set_status("The cohort is empty — add samples first.")
            return
        mz = self._mz_value(self.channel)
        if mz is None:
            self._set_status("Pick an ion m/z to reconstruct.")
            return
        stack = self._stack
        if stack is None:                     # build is allowed unregistered (identity warp)
            stack = self._build_stack_model(refs)
        loader = self._make_loader(refs)
        try:
            self._volume = volume3d.build_channel_volume(
                stack, mz, loader, random_state=0)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Build failed: {exc}")
            return
        finally:
            loader.close()

        Z = self._volume.shape[0]
        self.z_slider.setMaximum(max(0, Z - 1))
        self.z_slider.setValue(Z // 2)
        self.z_slider.setEnabled(Z > 1)
        self.b_gl.setEnabled(True)
        self._refresh_slices()
        self._set_status(f"Built a {self._volume.shape} volume at m/z {mz:.4f} "
                         f"({'registered' if self._stack is not None else 'unregistered'}).")
        self._record_audit(stack, refs, mz)

    def _record_audit(self, stack, refs, mz):
        if not hasattr(self.main, "record_step"):
            return
        secs = stack.ordered()
        anchor = self._mz_value(self.anchor, allow_tic=True)
        params = {
            "section_keys": [s.ref_key for s in secs],
            "z_index": [s.z_index for s in secs],
            "spacing_um": float(self.spacing.value()),
            "reference": self.reference.currentText(),
            "mode": self.mode.currentText(),
            "channel_mz": float(mz),
            "anchor_mz": anchor,
            "tol_ppm": 50.0,
            "reduce": "sum",
            "norm": "tic",
            "z_interp": "nearest",
            "random_state": 0,
        }
        try:
            self.main.record_step("volume3d", label="3D serial-section reconstruction",
                                  params=params, regions=None)
        except Exception:  # noqa: BLE001 — the audit record never breaks the build
            pass

    # ---- viewer ---------------------------------------------------------- #
    def _on_mip(self, checked: bool):
        self._mip = bool(checked)
        self.z_slider.setEnabled(not self._mip and self._volume is not None
                                 and self._volume.shape[0] > 1)
        self._refresh_slices()

    def _refresh_slices(self):
        vol = self._volume
        if vol is None:
            return
        import numpy as np

        if self._mip:
            axial = vol.mip(axis=0)
            coronal = vol.mip(axis=1)
            sagittal = vol.mip(axis=2)
            self.z_label.setText("MIP")
        else:
            z = int(self.z_slider.value())
            axial, coronal, sagittal = vol.orthoslices(z=z)
            depth = float(vol.z_um[z]) if z < len(vol.z_um) else 0.0
            self.z_label.setText(f"{z}  ({depth:.0f} µm)")

        finite = vol.data[np.isfinite(vol.data)]
        levels = ((float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0))
        for key, img in (("axial", axial), ("coronal", coronal), ("sagittal", sagittal)):
            disp = np.where(np.isfinite(img), img, levels[0]).T  # ImageItem is column-major
            self._views[key].setImage(disp, levels=levels, autoLevels=False)
            self._vbs[key].autoRange()

    def _gl_render(self):
        from .. import volume3d

        if self._volume is None:
            self._set_status("Build a volume first.")
            return
        try:
            item = volume3d.gl_volume_item(self._volume)
        except ImportError as exc:
            self._set_status(str(exc).splitlines()[0])
            return
        try:
            import pyqtgraph.opengl as gl

            w = gl.GLViewWidget()
            w.setWindowTitle(f"3D volume — m/z {self._volume.mz:.4f}")
            w.addItem(item)
            Z, H, W = self._volume.shape
            w.opts["distance"] = 2 * max(Z, H, W)
            w.resize(640, 640)
            w.show()
            self._gl_window = w               # keep a reference so it isn't GC'd
            self._set_status("Opened the interactive 3D volume render.")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"3D render failed: {exc}")

    # ---- status ---------------------------------------------------------- #
    def _set_status(self, msg: str):
        self.status.setText(msg)
        sb = getattr(self.main, "statusBar", None)
        if callable(sb):
            try:
                sb().showMessage(msg)
            except Exception:  # noqa: BLE001
                pass
