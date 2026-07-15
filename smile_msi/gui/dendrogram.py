"""DendrogramView — an interactive HCA (hierarchical cluster analysis) tree.

The segmentation engine already builds a cuttable :class:`spatial.Hierarchy` (a Ward /
bisecting linkage over micro-clusters) that the Detail slider slices live. This widget
*draws* that tree so the cut is no longer blind: the user sees the branching structure
and a horizontal line through it sets how many segments to keep — the classic
interactive HCA (hierarchical cluster analysis) interaction. The line and the Detail slider are two handles on the same
cut, kept in sync by the segmentation tab.

The tree is built over ~120-200 micro-clusters, but the merges *inside* a real segment
happen at tiny heights and only clutter the picture (Ward's root merges are orders of
magnitude taller, so they'd crush everything else into a sliver). So we **truncate**:
collapse each block of micro-clusters that the tree would keep together into a single
leaf, and draw only the top merges that actually separate segments. That is exactly the
band the Detail slider cuts in, so the cut line always lands on a visible branch.

Two click modes (a toggle at the top switches between them):

* **Move cut line** — click anywhere (or near the axis) to drop the cut there; the dashed
  line is also draggable. Re-segments exactly like the Detail slider.
* **Highlight branches** — click a branch to light up its whole subtree, sticky like a
  shift-click; click it again to drop it. Lit branches accumulate, and "New region from
  highlighted branches" turns their pixels into a region — so you can run down one branch
  of the tree, then another, and combine them into one region (splitting a segment finer
  than the live cut would).

Pure view: it owns no region/segmentation state beyond the live ``Hierarchy`` handed to it
(itself ephemeral, rebuilt on every segmentation run) plus the transient branch selection.
It reports the cut via :attr:`cutChanged`, the lit branches via :attr:`branchSelectionChanged`
(a frozenset of micro-cluster ids), and a region request via :attr:`makeRegionRequested`;
the host owns regions and pixels. Colours come from the shared ``PALETTE`` and the same
size-ordered relabelling ``spatial.cut`` uses, so a branch's colour matches its blob on the
segmentation map.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import spatial
from .common import (PALETTE, ACCENT, GUIDE_LINE, HILITE, MUTED_QSS, section_title,
                     plot_caption, glossary_button, icon)

# Show this many collapsed leaves at most — enough headroom below the slider's finest cut
# that every cut the user can pick lands on a drawn branch, while staying readable.
_LEAF_MARGIN = 8
_MIN_LEAVES = 32

# How close (in screen pixels) a click must land to a drawn branch to select it.
_HIT_PX = 14.0


class DendrogramView(QtWidgets.QWidget):
    """A plot of a :class:`spatial.Hierarchy` with a draggable cut line and clickable
    branches.

    Emits :attr:`cutChanged` with the new segment count ``k`` when the cut moves;
    :attr:`branchSelectionChanged` with the set of micro-cluster ids under the currently
    lit branches; and :attr:`makeRegionRequested` (same payload) when the user asks to
    turn the lit branches into a region. The host wires the cut to the Detail slider and
    the branch payloads to its region builder.
    """

    cutChanged = QtCore.Signal(int)
    branchSelectionChanged = QtCore.Signal(object)     # frozenset[int] of micro-cluster ids
    makeRegionRequested = QtCore.Signal(object)         # frozenset[int] of micro-cluster ids
    modeChanged = QtCore.Signal(str)                    # "cut" | "pick"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hier = None
        self._maxk = 24
        self._k = 0
        self._mode = "cut"              # "cut" → click moves the line; "pick" → click lights branches
        self._suppress = False          # guard: programmatic line moves must not re-emit
        self._curves = {}               # colour hex → reusable PlotCurveItem
        self._height = None             # full linkage merge heights (raw)
        self._dh = None                 # √-compressed display heights (drive the cut math)
        self._n_micro = 0
        self._sel_nodes = set()         # linkage node ids whose subtrees are lit
        self._sel_micros = frozenset()  # cached union of micro ids under the lit nodes
        self._reset_geometry()

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(section_title("Cluster tree (HCA)"))
        head.addStretch(1)
        head.addWidget(glossary_button([
            ["HCA", "Hierarchical cluster analysis: pixels are merged bottom-up into a tree "
                    "by how similar their spectra are. Cutting the tree at a height yields "
                    "that many segments."],
            ["merge distance", "Height of a branch = how different the two groups it joins "
                               "are. Tall joins = very distinct; short joins = nearly alike. "
                               "A big vertical gap is a natural place to cut."],
            ["cut line", "Sets how many segments to keep. In 'Move cut line' mode, click "
                         "anywhere (or near the axis) to drop it; it also drags. Same control "
                         "as the Detail slider above."],
            ["highlight branches", "In 'Highlight branches' mode, click a branch to light up "
                                   "its whole subtree (sticky — click again to drop it). Run "
                                   "down one branch, then another, then make a region from them."],
            ["collapsed leaf", "Each leaf bundles the micro-clusters the tree keeps together "
                               "at this depth, so the picture stays readable."],
        ], parent=self))
        v.addLayout(head)

        # --- mode toggle: what a click does (move the cut vs light up a branch) --------
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(4)
        click_lbl = QtWidgets.QLabel("Click to:")
        click_lbl.setStyleSheet(MUTED_QSS)
        mode_row.addWidget(click_lbl)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self.b_mode_cut = QtWidgets.QToolButton()
        self.b_mode_cut.setText("Move cut line")
        self.b_mode_cut.setIcon(icon("navigate"))
        self.b_mode_cut.setCheckable(True)
        self.b_mode_cut.setChecked(True)
        self.b_mode_cut.setToolTip("Click anywhere on the tree (or near the axis) to set the "
                                   "cut; the dashed line also drags.")
        self.b_mode_pick = QtWidgets.QToolButton()
        self.b_mode_pick.setText("Highlight branches")
        self.b_mode_pick.setIcon(icon("find"))
        self.b_mode_pick.setCheckable(True)
        self.b_mode_pick.setToolTip("Click a branch to light up its subtree (click again to "
                                    "drop it); then build a region from the lit branches.")
        for b in (self.b_mode_cut, self.b_mode_pick):
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            self._mode_group.addButton(b)
            mode_row.addWidget(b)
        self.b_mode_cut.toggled.connect(lambda on: on and self._set_mode("cut"))
        self.b_mode_pick.toggled.connect(lambda on: on and self._set_mode("pick"))
        mode_row.addStretch(1)
        v.addLayout(mode_row)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumWidth(220)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.hideAxis("bottom")                 # leaf order carries no meaning
        # √ scale: Ward's root merges dwarf the rest, so a linear axis crushes the fine
        # structure into a sliver. √ keeps the ordering and the big-gap cues but spreads
        # the small merges out so the tree is legible. It's monotonic, so every cut
        # computation below is identical to working in raw distance.
        self.plot.setLabel("left", "merge distance (√ scale)")
        self._vb = self.plot.getViewBox()
        self._vb.setMouseEnabled(x=False, y=False)   # fixed frame; the line is the only handle
        self._vb.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0)
        self.plot.scene().sigMouseClicked.connect(self._scene_clicked)
        v.addWidget(self.plot, 1)

        # placeholder shown until a tree exists
        self._placeholder = pg.TextItem("Run segmentation to build the cluster tree.",
                                         anchor=(0.5, 0.5), color=GUIDE_LINE)
        self.plot.addItem(self._placeholder)
        self._placeholder.setPos(0.5, 0.5)

        # highlight overlay (lit branches), painted above the coloured tree but below the line
        self._hl_curve = pg.PlotCurveItem(pen=pg.mkPen(HILITE, width=3.4), antialias=True,
                                          connect="finite")
        self._hl_curve.setZValue(12)
        self.plot.addItem(self._hl_curve)

        self._cut_line = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen(ACCENT, width=2, style=QtCore.Qt.DashLine))
        self._cut_line.setZValue(20)
        self._cut_line.hide()
        self._cut_line.sigPositionChangeFinished.connect(self._line_dragged)
        self.plot.addItem(self._cut_line)

        self._readout = QtWidgets.QLabel("")
        self._readout.setStyleSheet(MUTED_QSS)
        v.addWidget(self._readout)

        self._sel_label = QtWidgets.QLabel("")        # lit-branch summary (separate from the cut)
        self._sel_label.setStyleSheet(MUTED_QSS)
        self._sel_label.hide()
        v.addWidget(self._sel_label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        self.b_make_region = QtWidgets.QPushButton("New region from highlighted branches")
        self.b_make_region.setObjectName("primaryAction")  # the widget's single create action
        self.b_make_region.setIcon(icon("add"))
        self.b_make_region.setToolTip("Combine every lit branch's pixels into a new region.")
        self.b_make_region.setEnabled(False)
        self.b_make_region.hide()
        self.b_make_region.clicked.connect(
            lambda: self.makeRegionRequested.emit(self._sel_micros))
        self.b_clear_branches = QtWidgets.QPushButton("Clear")
        self.b_clear_branches.setIcon(icon("remove"))
        self.b_clear_branches.setToolTip("Drop every highlighted branch and start over.")
        self.b_clear_branches.setEnabled(False)
        self.b_clear_branches.hide()
        self.b_clear_branches.clicked.connect(self.clear_branches)
        btn_row.addWidget(self.b_make_region, 1)
        btn_row.addWidget(self.b_clear_branches)
        v.addLayout(btn_row)

        self._caption = plot_caption("")
        v.addWidget(self._caption)
        self._set_mode("cut")                         # sets caption + line drag affordance

    def _reset_geometry(self):
        """Forget the drawn tree (kept in one place so __init__ and _clear_tree agree)."""
        self._link_x = self._link_y = None          # (M,4) U-link coords per shown merge
        self._link_node = None                       # node id of each shown merge (ascending)
        self._child_ref = None                       # (M,2) left/right child refs
        self._child_is_group = None                  # (M,2) is that child a collapsed leaf?
        self._group_rep = {}                         # collapsed-group id → a member micro id
        self._parent = None                          # (2N-1,) linkage parent of each node (-1 root)
        self._node_leaves = None                     # node id → np.array of micro ids beneath it

    # ------------------------------------------------------------------ #
    # building the static tree
    # ------------------------------------------------------------------ #
    def set_hierarchy(self, hier, maxk: int = 24):
        """Lay out ``hier``'s linkage once — truncated to readable cluster blocks — and
        draw it. ``maxk`` clamps how fine the cut line may go (matches the slider's max)."""
        from scipy.cluster.hierarchy import fcluster

        self._hier = hier
        self._maxk = int(max(2, maxk))
        self._clear_selection()                      # a fresh tree invalidates lit branches
        Z = np.asarray(getattr(hier, "linkage", np.empty((0, 4))), dtype=float)
        N = int(getattr(hier, "n_micro", 0))
        if Z.shape[0] < 1 or N < 2:
            self._clear_tree()
            return
        self._placeholder.hide()

        left = Z[:, 0].astype(int)
        right = Z[:, 1].astype(int)
        height = Z[:, 2].astype(float)
        # display heights: √-compressed (see the axis-label note). All drawing + cut math
        # below uses these; colouring uses the raw linkage, so nothing else changes.
        dh = np.sqrt(np.clip(height, 0.0, None))
        self._height, self._dh, self._n_micro = height, dh, N
        self._micro_sizes = np.bincount(np.asarray(hier.micro_labels), minlength=N)

        # parent + per-node leaf (micro-cluster) membership over the *full* linkage — drives
        # branch hit-testing and the highlight overlay (N is small, so this is cheap).
        self._parent = np.full(2 * N - 1, -1, dtype=int)
        leaves = [None] * (2 * N - 1)
        for m in range(N):
            leaves[m] = [m]
        for i in range(N - 1):
            a, b = int(left[i]), int(right[i])
            self._parent[a] = self._parent[b] = N + i
            leaves[N + i] = leaves[a] + leaves[b]
        self._node_leaves = [np.asarray(g, dtype=int) for g in leaves]

        # Truncate: keep at most ``P`` blocks. Each block is one flat cluster at k=P, i.e.
        # a bundle of micro-clusters the tree holds together — drawn as a single leaf.
        P = int(min(N, max(self._maxk + _LEAF_MARGIN, _MIN_LEAVES)))
        gP = np.asarray(fcluster(Z, t=P, criterion="maxclust"))      # per-micro block id, ≥1
        self._group_rep = {}
        for m in range(N):
            self._group_rep.setdefault(int(gP[m]), m)

        # A node's block is its block id if its whole subtree sits in one block, else -1
        # (-1 = a "shown" merge that actually separates two blocks).
        node_block = np.zeros(2 * N - 1, dtype=int)
        node_block[:N] = gP
        for i in range(N - 1):
            a, b = node_block[left[i]], node_block[right[i]]
            node_block[N + i] = a if (a == b and a > 0) else -1

        # leaf order (full DFS) → left-to-right slot for each block by first appearance.
        order = []
        stack = [2 * N - 2]
        while stack:
            node = stack.pop()
            if node < N:
                order.append(node)
            else:
                i = node - N
                stack.append(right[i])
                stack.append(left[i])
        slot = {}
        for leaf in order:
            slot.setdefault(int(gP[leaf]), len(slot))
        block_x = {g: 10.0 * s + 5.0 for g, s in slot.items()}

        # x of every node: a collapsed block → its leaf x; a shown merge → midpoint of kids.
        nodex = np.zeros(2 * N - 1)
        for nid in range(2 * N - 1):
            g = node_block[nid]
            if g > 0:
                nodex[nid] = block_x[g]
        for i in range(N - 1):
            if node_block[N + i] < 0:
                nodex[N + i] = 0.5 * (nodex[left[i]] + nodex[right[i]])

        # Build only the shown merges (top P-1), in ascending node order so a shown child
        # is always laid out before its parent.
        link_x, link_y, link_node = [], [], []
        child_ref, child_is_group = [], []
        for i in range(N - 1):
            nid = N + i
            if node_block[nid] > 0:                  # collapsed away → not drawn
                continue
            a, b, h = left[i], right[i], dh[i]
            ya = 0.0 if node_block[a] > 0 else dh[a - N]
            yb = 0.0 if node_block[b] > 0 else dh[b - N]
            link_x.append([nodex[a], nodex[a], nodex[b], nodex[b]])
            link_y.append([ya, h, h, yb])
            link_node.append(nid)
            child_ref.append([int(node_block[a]) if node_block[a] > 0 else a,
                              int(node_block[b]) if node_block[b] > 0 else b])
            child_is_group.append([node_block[a] > 0, node_block[b] > 0])

        self._link_x = np.asarray(link_x, dtype=float)
        self._link_y = np.asarray(link_y, dtype=float)
        self._link_node = np.asarray(link_node, dtype=int)
        self._child_ref = np.asarray(child_ref, dtype=int)
        self._child_is_group = np.asarray(child_is_group, dtype=bool)

        ymax = float(dh.max()) if dh.size else 1.0
        self._ymax = ymax * 1.05
        self._vb.setRange(xRange=(0.0, 10.0 * max(1, len(slot))),
                          yRange=(0.0, self._ymax), padding=0)
        self._cut_line.show()
        self._cut_line.setBounds((0.0, self._ymax))

    def _clear_tree(self):
        """No usable tree: blank the curves and show the placeholder."""
        for item in self._curves.values():
            item.setData([], [])
        self._cut_line.hide()
        self._placeholder.show()
        self._hier = None
        self._height = self._dh = None
        self._clear_selection()
        self._reset_geometry()
        self._readout.setText("")

    # ------------------------------------------------------------------ #
    # cutting / colouring
    # ------------------------------------------------------------------ #
    def set_cut(self, k: int):
        """Move the line to the height that yields ``k`` segments and recolour every
        branch by the segment it belongs to (size-ordered, matching the map). Does not
        emit :attr:`cutChanged` — it is the *response* to a cut, not a new one."""
        if self._hier is None or self._link_node is None:
            return
        k = int(max(2, min(int(k), self._maxk)))
        self._k = k
        self._recolor(k)
        self._position_line(k)
        self._readout.setText(self._cut_readout(k))

    def _cut_readout(self, k: int) -> str:
        # In pick mode the line is locked and clicks light branches, so don't invite a cut.
        return f"Cut locked at {k} segments" if self._mode == "pick" else f"Cut here → {k} segments"

    def _recolor(self, k: int):
        from scipy.cluster.hierarchy import fcluster

        macro = fcluster(self._hier.linkage, t=k, criterion="maxclust")   # per-micro, 1..≤k
        relab = spatial._relabel_by_size(macro, self._micro_sizes)        # 0..k-1, size-ordered
        # each collapsed block sits wholly in one segment (k ≤ P), so a representative
        # micro fixes the block's segment id.
        block_seg = {g: int(relab[m]) for g, m in self._group_rep.items()}

        node_seg = {}                 # shown node id → its segment, or -1 if it spans two
        segs = {}                     # colour hex → [xs, ys] with NaN gaps between U-links
        for idx in range(self._link_node.shape[0]):
            kids = []
            for side in (0, 1):
                ref = int(self._child_ref[idx, side])
                kids.append(block_seg[ref] if self._child_is_group[idx, side]
                            else node_seg.get(ref, -1))
            ca, cb = kids
            seg = ca if (ca == cb and ca >= 0) else -1
            node_seg[int(self._link_node[idx])] = seg
            color = GUIDE_LINE if seg < 0 else PALETTE[seg % len(PALETTE)]
            xs, ys = segs.setdefault(color, ([], []))
            lx, ly = self._link_x[idx], self._link_y[idx]
            xs += [lx[0], lx[1], lx[2], lx[3], np.nan]
            ys += [ly[0], ly[1], ly[2], ly[3], np.nan]

        for color, (xs, ys) in segs.items():
            self._curve(color).setData(np.asarray(xs), np.asarray(ys))
        for color, item in self._curves.items():       # blank colours not used this cut
            if color not in segs:
                item.setData([], [])

    def _curve(self, color: str):
        """Reuse one PlotCurveItem per colour so dragging the cut never churns items."""
        item = self._curves.get(color)
        if item is None:
            width = 2.4 if color == GUIDE_LINE else 1.7
            # connect='finite' breaks the path at the NaN gaps between U-links, so the
            # tree's branches never get joined by a stray diagonal.
            item = pg.PlotCurveItem(pen=pg.mkPen(color, width=width), antialias=True,
                                    connect="finite")
            item.setZValue(5 if color == GUIDE_LINE else 6)
            self.plot.addItem(item)
            self._curves[color] = item
        return item

    def _position_line(self, k: int):
        """Place the cut line at a (display) height that uniquely yields ``k`` segments:
        midway between the sorted merge heights ``hs[N-k-1]`` and ``hs[N-k]`` — i.e. just
        above the ``N-k`` merges that must fall below the cut to leave ``k`` clusters."""
        hs = np.sort(self._dh)
        N = self._n_micro
        lo_idx, hi_idx = N - k - 1, N - k
        if hi_idx >= hs.size:                 # k == 1 (never reached via maxk≥2, kept safe)
            y = float(hs[-1]) * 1.05
        elif lo_idx < 0:                      # k == N
            y = float(hs[0]) * 0.5
        else:
            y = 0.5 * (float(hs[lo_idx]) + float(hs[hi_idx]))
        self._suppress = True
        self._cut_line.setValue(y)
        self._suppress = False

    def _k_for_y(self, y: float) -> int:
        """Segments left by cutting at (display) height ``y`` == leaves minus merges
        at-or-below ``y``, clamped to the cut line's usable range."""
        k = int(self._n_micro - int((self._dh <= y).sum()))
        return int(max(2, min(k, self._maxk)))

    # ------------------------------------------------------------------ #
    # mode toggle
    # ------------------------------------------------------------------ #
    def is_picking(self) -> bool:
        return self._mode == "pick"

    def _set_mode(self, mode: str):
        self._mode = "pick" if mode == "pick" else "cut"
        picking = self._mode == "pick"
        # In pick mode the line is fixed so clicking branches near it never nudges the cut.
        self._cut_line.setMovable(not picking)
        if self.b_mode_cut.isChecked() == picking:
            self.b_mode_cut.setChecked(not picking)
        if self.b_mode_pick.isChecked() != picking:
            self.b_mode_pick.setChecked(picking)
        self.b_make_region.setVisible(picking)
        self.b_clear_branches.setVisible(picking)
        self._sel_label.setVisible(picking and bool(self._sel_nodes))
        # Hide the amber tree overlay in cut mode (it would dangle with no explaining control),
        # but keep _sel_nodes/_sel_micros so switching back restores it with no recompute.
        self._hl_curve.setVisible(picking)
        if self._k >= 2 and self._hier is not None:
            self._readout.setText(self._cut_readout(self._k))
        self._caption.setText(
            "Click a branch to light up its subtree (click again to drop it). Run down one "
            "branch, then another, then 'New region from highlighted branches'."
            if picking else
            "Click anywhere (or near the axis) to set the cut · drag the dashed line too · "
            "down = more, finer segments. Branch colours match the segmentation map.")
        self.modeChanged.emit(self._mode)

    # ------------------------------------------------------------------ #
    # branch highlighting
    # ------------------------------------------------------------------ #
    def _clear_selection(self):
        """Drop every lit branch (no signal — callers that need one emit it)."""
        self._sel_nodes = set()
        self._sel_micros = frozenset()
        if getattr(self, "_hl_curve", None) is not None:
            self._hl_curve.setData([], [])
        if getattr(self, "_sel_label", None) is not None:
            self._sel_label.hide()
        for b in ("b_make_region", "b_clear_branches"):
            if getattr(self, b, None) is not None:
                getattr(self, b).setEnabled(False)

    def clear_branches(self):
        """Public: drop the lit branches and notify the host (e.g. after a region is made)."""
        if not self._sel_nodes:
            return
        self._clear_selection()
        self.branchSelectionChanged.emit(self._sel_micros)

    def _node_at(self, scene_pos) -> int | None:
        """Linkage node id of the drawn branch nearest ``scene_pos``, or ``None`` if no
        branch is within :data:`_HIT_PX` screen pixels. Clicking the horizontal join of a
        merge picks the whole merged cluster; clicking a vertical riser picks the child
        subtree hanging from it (a collapsed leaf or a deeper merge)."""
        if self._link_x is None or not self._link_x.size:
            return None
        click = self._vb.mapSceneToView(scene_pos)
        cx, cy = float(click.x()), float(click.y())
        # view→pixel scale per axis so the threshold is isotropic on screen despite the
        # very different x (leaf slots) and y (√-height) units.
        px = self._vb.viewPixelSize()
        if not (px[0] > 0 and px[1] > 0):                 # no on-screen transform yet → no hit
            return None
        sx, sy = 1.0 / px[0], 1.0 / px[1]

        def _seg_px(x0, y0, x1, y1):
            ax, ay = (cx - x0) * sx, (cy - y0) * sy
            bx, by = (x1 - x0) * sx, (y1 - y0) * sy
            denom = bx * bx + by * by
            t = 0.0 if denom == 0 else max(0.0, min(1.0, (ax * bx + ay * by) / denom))
            dx, dy = ax - t * bx, ay - t * by
            return (dx * dx + dy * dy) ** 0.5

        best_d, best = float("inf"), None
        for row in range(self._link_x.shape[0]):
            lx, ly = self._link_x[row], self._link_y[row]
            for i0, i1, which in ((0, 1, "L"), (1, 2, "T"), (2, 3, "R")):
                d = _seg_px(lx[i0], ly[i0], lx[i1], ly[i1])
                if d < best_d:
                    best_d, best = d, (row, which)
        if best is None or best_d > _HIT_PX:
            return None
        row, which = best
        nid = int(self._link_node[row])
        if which == "T":
            return nid
        i = nid - self._n_micro
        Z = self._hier.linkage
        return int(Z[i, 0]) if which == "L" else int(Z[i, 1])

    def _within(self, node: int, roots) -> bool:
        """Is ``node`` the same as, or a descendant of, any node in ``roots``? (Linkage
        parent ids strictly increase, so walking up from ``node`` only needs to climb
        while the id stays ≤ the root being tested.)"""
        for r in roots:
            cur = node
            while 0 <= cur <= r:
                if cur == r:
                    return True
                cur = int(self._parent[cur])
        return False

    def _toggle_node(self, node: int):
        """Toggle a branch, keeping the lit set a disjoint cover so the amber overlay is
        always faithful to what is selected: clicking exactly a lit node drops it; clicking
        *inside* an already-lit subtree turns that covering subtree off (so a click on the
        amber always un-lights it); a fresh node subsumes any lit descendants it covers."""
        if node in self._sel_nodes:
            self._sel_nodes.discard(node)
        elif self._within(node, self._sel_nodes):                # inside a lit subtree → un-light it
            self._sel_nodes = {r for r in self._sel_nodes if not self._within(node, {r})}
        else:                                                    # new subtree → absorb lit descendants
            self._sel_nodes = {r for r in self._sel_nodes if not self._within(r, {node})}
            self._sel_nodes.add(node)
        self._draw_highlight()
        self._emit_selection()

    def _draw_highlight(self):
        """Paint every drawn branch segment that sits under a lit node in the amber
        highlight pen (NaN-broken so risers/bars never join across the gaps)."""
        if not self._sel_nodes or self._link_x is None:
            self._hl_curve.setData([], [])
            return
        roots = self._sel_nodes
        xs, ys = [], []
        for row in range(self._link_x.shape[0]):
            nid = int(self._link_node[row])
            i = nid - self._n_micro
            a, b = int(self._hier.linkage[i, 0]), int(self._hier.linkage[i, 1])
            lx, ly = self._link_x[row], self._link_y[row]
            for node, i0, i1 in ((a, 0, 1), (nid, 1, 2), (b, 2, 3)):
                if self._within(node, roots):
                    xs += [lx[i0], lx[i1], np.nan]
                    ys += [ly[i0], ly[i1], np.nan]
        self._hl_curve.setData(np.asarray(xs), np.asarray(ys))

    def _emit_selection(self):
        micros = set()
        for nid in self._sel_nodes:
            micros.update(int(m) for m in self._node_leaves[nid])
        self._sel_micros = frozenset(micros)
        n_branch = len(self._sel_nodes)
        # Gate the readout + buttons on resolvable *pixels*, not micro count, so the
        # make-region button is never enabled for a selection that maps to no pixels.
        npx = int(np.isin(self._hier.micro_labels, list(micros)).sum()) \
            if (micros and self._hier is not None) else 0
        if npx > 0:
            self._sel_label.setText(f"{n_branch} branch{'es' if n_branch != 1 else ''} lit "
                                    f"· {npx:,} px — make a region below")
            self._sel_label.show()
        else:
            self._sel_label.hide()
        self.b_make_region.setEnabled(npx > 0)
        self.b_clear_branches.setEnabled(bool(self._sel_nodes))
        self.branchSelectionChanged.emit(self._sel_micros)

    # ------------------------------------------------------------------ #
    # user input
    # ------------------------------------------------------------------ #
    def _scene_clicked(self, ev):
        """A click on the plot: in cut mode drop the line at the click height; in pick
        mode toggle the nearest branch. Dragging the dashed line (cut mode only) goes
        through :meth:`_line_dragged` instead — this fires only on genuine clicks."""
        if self._hier is None or self._link_node is None:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if self._mode == "pick":
            node = self._node_at(ev.scenePos())
            if node is not None:
                self._toggle_node(node)
                ev.accept()
            return
        # cut mode: a bare left-click anywhere (including directly on the movable line —
        # pyqtgraph still fires sigMouseClicked for an un-dragged click) drops the cut at
        # the click height. Genuine drags route through _line_dragged instead.
        y = float(self._vb.mapSceneToView(ev.scenePos()).y())
        y = max(0.0, min(y, getattr(self, "_ymax", y)))
        k = self._k_for_y(y)
        ev.accept()
        if k == self._k:                      # clicking the current cut height is a no-op
            return
        self.set_cut(k)                       # snap the line to k's canonical height + recolour
        self.cutChanged.emit(k)

    def _line_dragged(self):
        if self._suppress or self._hier is None or self._dh is None:
            return
        k = self._k_for_y(float(self._cut_line.value()))
        self.set_cut(k)                       # snap the line to k's canonical height + recolour
        self.cutChanged.emit(k)
