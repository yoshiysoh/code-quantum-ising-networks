"""Fig. 3 and Extended Data Fig. 3 (and the movies): dynamical classes on
one network.

One random-network realization and one ring, both from
scripts/generate_examples.py and scripts/generate_ring_example.py. Each
column is one initial Fock state: network with the field and the state
(top), F(t), m(t) and <Z_i(t)>, and the time-averaged occupation of the
sector's Fock states on a Hamming-distance layout (bottom). The random
network shows the five classes in one sector, in the order of the
classification flowchart; the ring shows the Néel scar and a thermalizing
state of the same sector.
"""

from __future__ import annotations

import io

import colorcet as cc
import h5py
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec

from .. import DATA_DIR as RESULTS
from .. import scar_metrics as sm
from ..model import GAMMA, all_sectors, sector_hamiltonian  # noqa: F401
from ..scar_metrics import DISPLAY_TIME_MAX, evolve_for_display
from .common import CLASS_COLORS, CLASS_LABELS
from .figure import (
    LABEL_BOLD,
    LABEL_FONT,
    MAX_FIG_WIDTH,
    journal_labels,
)

OCC_FLOOR = 1e-8  # occupations below this count as zero
N_ER = 5  # random-network columns
CLASS_ORDER = [0, 1, 4, 2, 3]  # flowchart order: localized, recurrence,
# thermalized, scar, plateau scar
MODELS = {
    "ferro": {
        "net_file": "examples_N12_ferro_phi1.00_M0.h5",
        "ring_file": "ring_N12_ferro_h2.0.h5",
        "fig_h_factor": 1.1016,
        "block_titles": (
            r"Erdős–Rényi network, $\phi=1$",
            r"1D ring, uniform $h = 2J$",
        ),
        "first_letter": 1,
    },
    "pmJ": {
        "net_file": "examples_N12_pmJ_phi1.00_M0.h5",
        "ring_file": "ring_N12_pmJ_h2.0_seed0.h5",
        "fig_h_factor": 1.1254,
        "block_titles": (
            r"Erdős–Rényi network ($\pm J$), $\phi=1$",
            r"Frustration-free $\pm J$ ring",
        ),
        "first_letter": 0,
    },
}

norm_z = mcolors.Normalize(vmin=-1, vmax=1)
# Saturate outside [1e-4, 0.1]; the pointed colorbar (extend='both') shows it.
norm_occ = mcolors.LogNorm(vmin=1e-4, vmax=0.1, clip=False)
cmap_z = cc.cm.CET_D1A
cmap_occ = cc.cm.CET_R2.copy()
cmap_occ.set_under(cmap_occ(0.0))
cmap_occ.set_over(cmap_occ(1.0))


def _extents(artists, renderer):
    """Window extents of the artists that report one, in order.

    Layout probes only: an artist that cannot report an extent (an empty
    collection, an unrendered text) is skipped, whatever it raises.
    """
    boxes = []
    for artist in artists:
        try:
            boxes.append(artist.get_window_extent(renderer))
        except Exception:  # noqa: BLE001, S112
            continue
    return boxes


def load_examples(model: str) -> dict:
    """The network example (one shared sector) and the ring example (a
    sector per class) of one model."""
    cfg = MODELS[model]
    net_path = RESULTS / cfg["net_file"]
    d = {"net_path": net_path}
    with h5py.File(net_path) as f:
        d["net_P"] = dict(f["params"].attrs)
        for key, name in [
            ("net_im", "im"),
            ("net_h", "h"),
            ("net_sec_idx", "sec_idx"),
            ("net_A", "adjacency"),
            ("net_alphas", "alphas"),
            ("net_time", "time"),
            ("net_z", "z_of_t"),
            ("net_fid", "fidelity_of_t"),
            ("net_occ", "occupation"),
            ("net_fid_rev", "fid_rev"),
            ("net_n_eff", "n_eff"),
            ("net_comp_size", "comp_size"),
        ]:
            d[key] = f[name][:]
    N = d["N"] = int(d["net_P"]["num_spins"])
    d["net_states"] = sm.states_from_indices(d["net_sec_idx"], N)

    ring_path = d["ring_path"] = RESULTS / cfg["ring_file"]
    ring_cols = d["ring_cols"] = []
    with h5py.File(ring_path) as f:
        d["ring_P"] = dict(f["params"].attrs)
        d["ring_im"] = f["im"][:]
        d["ring_h"] = f["h"][:]
        for c in range(5):
            g = f[f"class_{c}"]
            col = {"valid": bool(g.attrs["valid"]), "cls": c}
            if col["valid"]:
                sec_idx = g["sec_idx"][:]
                col.update(
                    sec_idx=sec_idx,
                    A=g["adjacency"][:],
                    alpha=int(g.attrs["alpha"]),
                    time=g["time"][:],
                    z=g["z_of_t"][:],
                    fid=g["fidelity_of_t"][:],
                    occ=g["occupation"][:],
                    fid_rev=float(g.attrs["fid_rev"]),
                    n_eff=float(g.attrs["n_eff"]),
                    comp_size=int(g.attrs["comp_size"]),
                    states=sm.states_from_indices(sec_idx, N),
                )
            ring_cols.append(col)
    return d


def network_layout(im, h, use_rcm=True):
    G = nx.from_numpy_array(im)
    if use_rcm:
        order = list(nx.utils.reverse_cuthill_mckee_ordering(G))
    else:
        order = list(range(len(h)))
    perm = {old: new for new, old in enumerate(order)}
    edges = sorted(
        (min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in G.edges()
    )
    return order, edges, h[order], im[np.ix_(order, order)]


def network_ylim(layout, y_spin=0.0, h_len=0.55, spacing=0.9, arrow_gap=0.45):
    """Data ylim that fits arcs + field arrows for one layout."""
    _order, edges, h_ord, _im = layout
    n = len(h_ord)
    x = np.arange(n, dtype=float) * spacing
    radii = [abs(x[j] - x[i]) / 2 for i, j in edges]
    y_lo = y_spin - (arrow_gap + h_len + 0.25)
    y_hi = max(radii, default=0.5) + 0.35
    return y_lo, y_hi


def draw_network(ax, layout, spins, y_spin=0.0, ylim=None, arrow_scale=1.0):
    """Network + field + initial Fock state; arrow_scale widens the field
    arrows (Movie 3 draws one network much larger than a figure column).

    Shared `ylim` keeps the spin row on one horizontal line across ER
    and ring panels (ER arcs are taller; without a shared ylim,
    equal-aspect centering shifts the spins).
    """
    order, edges, h_ord, im_ord = layout
    n = len(h_ord)
    spacing = 0.9
    node_s = 30  # marker area (points^2)
    # Edge width per node radius as in Fig. 1a (radius = sqrt(node_s) / 4).
    EDGE_RADIUS_RATIO = 0.31058538064907454
    lw = EDGE_RADIUS_RATIO * np.sqrt(node_s) / 4
    h_len = 0.90  # longer longitudinal-field arrows
    arrow_gap = 0.45  # keep clearance from spin nodes
    ferro = cc.cm.CET_D3_r(1.0)
    antiferro = cc.cm.CET_D3_r(0.0)
    x = np.arange(n, dtype=float) * spacing
    radii = [abs(x[j] - x[i]) / 2 for i, j in edges]

    ax.set_aspect("equal")
    ax.set_anchor("S")
    ax.axis("off")
    ax.set_xlim(-0.6, x[-1] + 0.6)
    if ylim is None:
        ylim = network_ylim(
            layout,
            y_spin=y_spin,
            h_len=h_len,
            spacing=spacing,
            arrow_gap=arrow_gap,
        )
    ax.set_ylim(*ylim)

    # Points per data unit (aspect='equal', so x shares it).
    fig = ax.figure
    fig.canvas.draw()
    _p0 = ax.transData.transform((0, 0))
    _p1 = ax.transData.transform((0, 1))
    pt_per_data_y = abs(_p1[1] - _p0[1]) / fig.dpi * 72

    # Node-to-arrow gap as in Fig. 1a.
    GAP_PT = 0.7124178858922372
    true_node_radius = (np.sqrt(node_s) / 4) / pt_per_data_y
    tight_arrow_gap = true_node_radius + GAP_PT / pt_per_data_y

    for (i, j), r in zip(edges, radii):
        pos = im_ord[i, j] > 0
        ax.add_patch(
            mpatches.Arc(
                ((x[i] + x[j]) / 2, y_spin),
                width=2 * r,
                height=2 * r,
                theta1=0,
                theta2=180,
                lw=lw,
                color=ferro if pos else antiferro,
                linestyle="solid" if pos else "dashed",
                zorder=1,
            )
        )

    # Arrow shaft width: the h = J arrow is as wide as an edge.
    arrow_width_base = arrow_scale * lw / pt_per_data_y
    for xi, h in zip(x, h_ord):
        w = arrow_width_base * abs(h)
        if h >= 0:
            ax.arrow(
                xi,
                y_spin - tight_arrow_gap - h_len,
                0,
                h_len,
                width=w,
                head_width=3 * w,
                head_length=0.50,
                fc="k",
                lw=0,
                length_includes_head=True,
                zorder=3,
                clip_on=False,
            )
        else:  # negative field on the gauged ±J ring: arrow points down
            ax.arrow(
                xi,
                y_spin + tight_arrow_gap + h_len,
                0,
                -h_len,
                width=w,
                head_width=3 * w,
                head_length=0.50,
                fc="k",
                lw=0,
                length_includes_head=True,
                zorder=3,
                clip_on=False,
            )

    s_ord = spins[order]
    ax.scatter(
        x,
        np.full(n, y_spin),
        s=node_s,
        zorder=4,
        linewidth=0.5,
        facecolor=np.where(s_ord > 0, "k", "w"),
        edgecolor="k",
    )


def shelf_pack(boxes, wrap):
    pos, x_off, y_off, row_h, w_max = {}, 0.0, 0.0, 0.0, 0.0
    for nodes, xy, (w, h) in boxes:
        if x_off > 0 and x_off + w > wrap:
            x_off, y_off, row_h = 0.0, y_off - row_h, 0.0
        for k, v in zip(nodes, xy):
            pos[k] = (v[0] + x_off, v[1] + y_off)
        x_off += w
        row_h = max(row_h, h)
        w_max = max(w_max, x_off)
    height = -y_off + row_h
    return pos, w_max, height


def pack_near_giant(boxes, wrap, gap=0.08):
    """Place every smaller component beside the largest.

    Packed bottom-right to bottom-left: the first (largest) small
    component sits at the bottom-right, and subsequent ones extend
    leftward along the same row. Once a row would exceed `wrap`, a new
    row starts slightly higher (still right-anchored) rather than
    continuing further left -- this keeps small fragments grounded
    near the giant's own baseline and spread across the panel's width
    instead of stacked in one arbitrarily tall column.
    """
    nodes0, xy0, (w0, h0) = boxes[0]
    pos = {k: (float(v[0]), float(v[1])) for k, v in zip(nodes0, xy0)}
    if len(boxes) == 1:
        return pos, w0, h0

    rows, row_w = [[]], 0.0
    for box in boxes[1:]:
        w = box[2][0]
        if row_w > 0 and row_w + gap + w > wrap:
            rows.append([])
            row_w = 0.0
        rows[-1].append(box)
        row_w += (gap if row_w > 0 else 0.0) + w

    x_off = w0 + gap
    y_cursor = 0.0  # first row sits on the giant's own baseline
    max_row_w = 0.0
    for row in rows:
        row_width = sum(b[2][0] for b in row) + gap * (len(row) - 1)
        row_height = max(b[2][1] for b in row)
        max_row_w = max(max_row_w, row_width)
        x_cursor = row_width  # right edge of this row, local coords
        for nodes, xy, (w, h) in row:
            x_cursor -= w
            for k, v in zip(nodes, xy):
                pos[k] = (
                    float(v[0]) + x_off + x_cursor,
                    float(v[1]) + y_cursor,
                )
            x_cursor -= gap
        y_cursor += row_height + gap

    total_h = max(h0, y_cursor - gap)
    return pos, x_off + max_row_w, total_h


def hamming_projection_layout(sec_idx, A, ref_idx):
    """x = Hamming distance from the reference (initial) state, y = a
    centered 'beeswarm' spread within each Hamming-distance shell.

    Gives a light-cone view of how a wavepacket spreads through
    Hilbert space with the initial state pinned at x=0 (on the left)
    and every other state placed by how many spin flips separate it
    from the initial state -- rather than an undirected spring layout,
    which conveys connectivity but not distance from a chosen origin.
    Edges (single-spin-flip hops) still only ever connect x=d to
    x=d+1, so they read as a fan spreading rightward. See Horiike &
    Fujishiro (2025) PRE.
    """
    ref = int(sec_idx[ref_idx])
    dist = np.array([(int(s) ^ ref).bit_count() for s in sec_idx], dtype=float)
    XY = np.zeros((len(sec_idx), 2))
    XY[:, 0] = dist
    order_all = np.argsort(sec_idx)
    for d in np.unique(dist):
        members = order_all[dist[order_all] == d]
        m = len(members)
        y = np.arange(m) - (m - 1) / 2.0
        XY[members, 1] = y
    G_sec = nx.from_numpy_array(A)
    EDGES = (
        np.array(list(G_sec.edges()))
        if G_sec.number_of_edges()
        else np.empty((0, 2), dtype=int)
    )
    return XY, EDGES


def hilbert_layout(sec_idx, A, panel_aspect, ref_idx, near_giant=False):
    G_sec = nx.from_numpy_array(A)
    comps = sorted(nx.connected_components(G_sec), key=len, reverse=True)
    boxes = []
    for comp in comps:
        sub = G_sec.subgraph(comp)
        scale = max(0.5, np.sqrt(len(comp)) / 3.5)
        q = nx.spring_layout(sub, seed=0, scale=scale, iterations=250)
        xy = np.array([q[k] for k in sub.nodes()])
        xy -= xy.min(axis=0)
        boxes.append((list(sub.nodes()), xy, xy.max(axis=0) + 0.30))
    if near_giant and len(boxes) > 1:
        # Choose the row width whose bounding box best matches panel_aspect.
        _w0, h0 = boxes[0][2]
        best = None
        for fac in np.linspace(0.3, 4.0, 40):
            wrap = max(0.1, fac * h0)
            pos_try, w, h = pack_near_giant(boxes, wrap)
            if h <= 0:
                continue
            cost = abs(np.log((w / h) / panel_aspect))
            if best is None or cost < best[0]:
                best = (cost, pos_try, w, h)
        _cost, pos, _W, _H = best
    else:
        area = sum(b[2][0] * b[2][1] for b in boxes)
        best = None
        for fac in np.linspace(0.6, 4.0, 40):
            wrap = fac * np.sqrt(area)
            pos_try, w, h = shelf_pack(boxes, wrap)
            if h <= 0:
                continue
            cost = abs(np.log((w / h) / panel_aspect))
            if best is None or cost < best[0]:
                best = (cost, pos_try, w, h)
        _cost, pos, _W, _H = best
    XY = np.array([pos[i] for i in range(len(sec_idx))])
    # Rotate the layout rigidly so the initial state sits on the left.
    centroid = XY.mean(axis=0)
    v = XY[ref_idx] - centroid
    if np.linalg.norm(v) > 1e-9:
        theta = np.pi - np.arctan2(v[1], v[0])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        XY = (XY - centroid) @ R.T + centroid
    EDGES = (
        np.array(list(G_sec.edges()))
        if G_sec.number_of_edges()
        else np.empty((0, 2), dtype=int)
    )
    return XY, EDGES


def load_sector(path, group=None):
    """(im, h, sector Fock states) of an example file; group selects one
    class of a ring file, e.g. "class_2" for the Néel scar."""
    with h5py.File(path) as f:
        im = f["im"][:]
        h = f["h"][:]
        sec_idx = (f[group]["sec_idx"] if group else f["sec_idx"])[:]
    return im, h, sec_idx


def example_sectors():
    """The three sectors of Extended Data Figs. 5 and 6: the ferromagnetic
    and ±J
    network examples and the ring's Néel-scar sector."""
    return [
        (
            "Ferromagnetic random network",
            load_sector(RESULTS / MODELS["ferro"]["net_file"]),
        ),
        (
            r"$\pm J$ random network",
            load_sector(RESULTS / MODELS["pmJ"]["net_file"]),
        ),
        (
            "Ring",
            load_sector(
                RESULTS / MODELS["ferro"]["ring_file"], group="class_2"
            ),
        ),
    ]


TIE_DECIMALS = 9
# No 6-up/6-down Fock state thermalizes on the ferromagnetic ring; 5 up /
# 7 down is the closest to the network columns' 6/6.
RING_THERMAL_NUM_UP = 5


def ring_typical_thermal_column(
    ring_im,
    ring_h,
    scar_col,
    time_max=DISPLAY_TIME_MAX,
    num_up=None,
    alpha=None,
):
    """A bulk-like thermalized Fock state in the ring scar's sector.

    The generator's median-fid_rev pick still shows strong finite-size
    revivals on the ring; prefer the thermalized state with the weakest
    late-time fidelity peak in the display window (ties broken by lower
    fid_rev, then lower Fock index). `num_up` restricts the pool to states
    with that many up spins. Symmetry-related ring states score equal up to
    rounding, so scores are compared at TIE_DECIMALS; otherwise the pick
    among them would depend on floating-point noise. With `alpha` (a
    position in the sector) the search is skipped and that state is used.
    """
    from ..examples import CLASSIFY_TIME_MAX

    sec_idx = scar_col["sec_idx"]
    num_spins = int(scar_col["states"].shape[1])
    for _rank, _e, si, H in all_sectors(ring_im, ring_h, GAMMA):
        if not np.array_equal(si, sec_idx):
            continue
        A = sm.adjacency_from_H(H, GAMMA)
        states = sm.states_from_indices(si, num_spins)
        m = sm.measure_sector(
            si,
            H,
            GAMMA,
            num_spins,
            time_max=CLASSIFY_TIME_MAX,
            max_component=len(si),
            sample_size=len(si),
            sample_rng=np.random.default_rng(0),
        )
        cls = sm.classify(
            m.fid_min, m.fid_rev, m.n_eff, num_spins, comp_size=m.comp_size
        )
        pool = np.flatnonzero(cls == sm.THERMALIZED)
        if num_up is not None:
            ups = (states[m.state_index[pool]] > 0).sum(axis=1)
            pool = pool[ups == num_up]
        if alpha is not None:
            pos = int(np.flatnonzero(m.state_index == alpha)[0])
            if cls[pos] != sm.THERMALIZED:
                raise RuntimeError(f"ring state {alpha} is not thermalized")
        elif pool.size == 0:
            raise RuntimeError("no thermalized states in ring scar sector")
        else:
            # Score a shortlist of low-fid_rev candidates by late-time max F.
            fid_rev = np.round(m.fid_rev, TIE_DECIMALS)
            short = pool[np.lexsort((m.state_index[pool], fid_rev[pool]))][:40]
            best = None
            for pos in short:
                a = int(m.state_index[pos])
                t, _z, fid, _occ = evolve_for_display(
                    H, si, num_spins, [a], time_max
                )
                late_max = round(float(fid[0][t > 30].max()), TIE_DECIMALS)
                key = (late_max, float(fid_rev[pos]), a)
                if best is None or key < best[0]:
                    best = (key, pos, a)
            _key, pos, alpha = best
        time_list, z_t, fid_t, occ = evolve_for_display(
            H, si, num_spins, [alpha], time_max
        )
        return {
            "valid": True,
            "cls": int(sm.THERMALIZED),
            "alpha": alpha,
            "sec_idx": si,
            "A": A,
            "states": states,
            "time": time_list,
            "z": z_t[0],
            "fid": fid_t[0],
            "occ": occ[0],
            "fid_rev": float(m.fid_rev[pos]),
            "n_eff": float(m.n_eff[pos]),
            "comp_size": int(m.comp_size[pos]),
        }
    raise RuntimeError("ring scar sector not found for thermalized re-pick")


def ring_thermal_column(model: str, d: dict | None = None):
    """The ring's thermalized column: the re-pick among ferromagnetic-ring
    states with RING_THERMAL_NUM_UP up spins, and on the ±J ring its gauge
    image, so that both rings show the same two states (the Néel scar is
    already a gauge pair) and S2, Q and S_half agree between them."""
    from ..examples import gauge_index

    def scar_of(dd):
        return next(
            col
            for col in dd["ring_cols"]
            if col["valid"] and col["cls"] == sm.SCAR
        )

    ferro = d if model == "ferro" and d is not None else load_examples("ferro")
    col = ring_typical_thermal_column(
        ferro["ring_im"],
        ferro["ring_h"],
        scar_of(ferro),
        num_up=RING_THERMAL_NUM_UP,
    )
    if model == "ferro":
        return col
    d = d if d is not None else load_examples(model)
    scar = scar_of(d)
    tau = np.sign(d["ring_h"])  # gauge_ring: h'_i = tau_i * h_uniform
    fock = gauge_index(int(col["sec_idx"][col["alpha"]]), tau)
    alpha = int(np.flatnonzero(scar["sec_idx"] == fock)[0])
    return ring_typical_thermal_column(
        d["ring_im"], d["ring_h"], scar, alpha=alpha
    )


def probabilities_at(evals, evecs, alpha, times):
    """|<n|psi(t)>|^2 of every Fock state n of the sector at `times`, for
    psi(0) = Fock state alpha; shape (dim, len(times))."""
    phase = np.exp(-1j * np.outer(times, evals))
    amps = evecs @ (phase * evecs[alpha, :][None, :]).T
    return np.abs(amps) ** 2


def example_columns(model: str):
    """(data, network columns, ring columns): the five network classes in
    CLASS_ORDER (a class without a candidate is a column with valid=False),
    then the ring's Néel scar and a typical thermalized state."""
    d = load_examples(model)
    print("network:", d["net_path"].name, "N=", d["N"])
    print("ring:", d["ring_path"].name)
    for c in range(5):
        print(
            " ",
            CLASS_LABELS[c],
            "valid" if d["ring_cols"][c]["valid"] else "N/A",
        )
    net_cols = []
    for c in CLASS_ORDER:
        if int(d["net_alphas"][c]) >= 0:
            net_cols.append(
                {
                    "valid": True,
                    "cls": c,
                    "alpha": int(d["net_alphas"][c]),
                    "sec_idx": d["net_sec_idx"],
                    "A": d["net_A"],
                    "states": d["net_states"],
                    "time": d["net_time"],
                    "z": d["net_z"][c],
                    "fid": d["net_fid"][c],
                    "occ": d["net_occ"][c],
                    "fid_rev": float(d["net_fid_rev"][c]),
                    "n_eff": float(d["net_n_eff"][c]),
                    "comp_size": int(d["net_comp_size"][c]),
                }
            )
        else:
            net_cols.append({"valid": False, "cls": c})
    ring_scar = next(
        col for col in d["ring_cols"] if col["valid"] and col["cls"] == sm.SCAR
    )
    ring_thermal = ring_thermal_column(model, d)
    ring_present = [ring_scar, ring_thermal]
    thermal_col = next(nc for nc in net_cols if nc["cls"] == sm.THERMALIZED)
    print(
        "columns:",
        N_ER + len(ring_present),
        f"(ER {N_ER} + ring {len(ring_present)})",
    )
    print("ER thermalized fid_rev=", round(thermal_col["fid_rev"], 3))
    print(
        "ring thermalized alpha=",
        ring_thermal["alpha"],
        "ups=",
        int((ring_thermal["states"][ring_thermal["alpha"]] > 0).sum()),
        "fid_rev=",
        round(ring_thermal["fid_rev"], 3),
        "comp=",
        ring_thermal["comp_size"],
    )
    return d, net_cols, ring_present


def dynamics_figure(model: str):
    """Fig. 3's panels b onward (model="ferro") or Extended Data Fig. 3
    (model="pmJ"),
    with panel letters placed for save_unlabeled."""
    cfg = MODELS[model]
    d, net_cols, ring_present = example_columns(model)
    N = d["N"]
    net_im, net_h = d["net_im"], d["net_h"]
    ring_im, ring_h = d["ring_im"], d["ring_h"]
    n_ring = len(ring_present)
    total_cols = N_ER + n_ring

    HEIGHT_RATIOS = [0.80, 0.40, 0.40, 0.70, 0.90]
    CBAR_WIDTH = 0.001
    UNIT = 5.0 / sum([0.60, 0.40, 1.00, 1.05])
    FIG_H = (
        UNIT * sum(HEIGHT_RATIOS) * cfg["fig_h_factor"]
    )  # spanning titles sit on letters
    fig_w = MAX_FIG_WIDTH
    panel_w = (
        fig_w * total_cols / (total_cols + CBAR_WIDTH)
    ) / total_cols - 0.12
    panel_h = FIG_H * HEIGHT_RATIOS[4] / sum(HEIGHT_RATIOS) - 0.10
    panel_aspect = panel_w / panel_h

    fig = plt.figure(figsize=(fig_w, FIG_H), constrained_layout=True)
    fig.set_constrained_layout_pads(
        h_pad=0.02, w_pad=0.01, hspace=0.04, wspace=0.03
    )
    gs = GridSpec(
        5,
        total_cols + 1,
        figure=fig,
        height_ratios=HEIGHT_RATIOS,
        width_ratios=[1.0] * total_cols + [CBAR_WIDTH],
    )

    ax_net = np.empty(total_cols, dtype=object)
    ax_fid = np.empty(total_cols, dtype=object)
    ax_mag = np.empty(total_cols, dtype=object)
    ax_spin = np.empty(total_cols, dtype=object)
    ax_hil = np.empty(total_cols, dtype=object)
    for c in range(total_cols):
        ax_net[c] = fig.add_subplot(gs[0, c])
        ax_fid[c] = fig.add_subplot(gs[1, c])
        ax_mag[c] = fig.add_subplot(
            gs[2, c], sharex=ax_fid[c], sharey=ax_mag[0] if c else None
        )
        ax_spin[c] = fig.add_subplot(gs[3, c], sharex=ax_fid[c])
        ax_hil[c] = fig.add_subplot(gs[4, c])

    ax_cbar_z = fig.add_subplot(gs[3, -1])  # match heatmap row only
    ax_cbar_o = fig.add_subplot(gs[4, -1])

    BLOCK_TITLES = cfg["block_titles"]
    BLOCK_TITLE_FONTSIZE = 7.0  # pt

    lay_net = network_layout(net_im, net_h, use_rcm=True)
    lay_ring = network_layout(ring_im, ring_h, use_rcm=False)
    y_lo = min(
        network_ylim(lay_net, h_len=0.90, arrow_gap=0.45)[0],
        network_ylim(lay_ring, h_len=0.90, arrow_gap=0.45)[0],
    )
    y_hi = max(
        network_ylim(lay_net, h_len=0.90, arrow_gap=0.45)[1],
        network_ylim(lay_ring, h_len=0.90, arrow_gap=0.45)[1],
    )
    SHARED_YLIM = (y_lo, y_hi)

    def render_columns(
        cols,
        layout,
        ax_net_s,
        ax_fid_s,
        ax_mag_s,
        ax_spin_s,
        ax_hil_s,
        label_col0=True,
        near_giant_cols=(),
        shared_hilbert=None,
        use_hamming=False,
        hil_top_k=None,
    ):
        for i, col in enumerate(cols):
            cls = col.get("cls", i)
            colour = CLASS_COLORS[cls]
            ax_n, ax_f, ax_m, ax_s, ax_h = (
                ax_net_s[i],
                ax_fid_s[i],
                ax_mag_s[i],
                ax_spin_s[i],
                ax_hil_s[i],
            )
            if not col["valid"]:
                for ax in (ax_n, ax_f, ax_m, ax_s, ax_h):
                    ax.axis("off")
                ax_n.set_title(CLASS_LABELS[cls], color=colour, pad=2)
                ax_h.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax_h.transAxes,
                    color="0.5",
                )
                continue

            alpha = col["alpha"]
            time = col["time"]
            z_of_t, fid_of_t, occ = col["z"], col["fid"], col["occ"]

            draw_network(ax_n, layout, col["states"][alpha], ylim=SHARED_YLIM)
            ax_n.set_title(CLASS_LABELS[cls], color=colour, pad=2)

            ax_f.plot(time, fid_of_t, lw=0.7, color=colour)
            ax_f.set_xlim(time[0], time[-1])
            ax_f.set_ylim(-0.05, 1.10)
            ax_f.set_yticks([0, 1])
            ax_f.tick_params(labelbottom=False)
            if i == 0 and label_col0:
                ax_f.set_ylabel(r"Fidelity $F(t)$")
            else:
                ax_f.set_yticklabels([])

            ax_m.plot(time, z_of_t.mean(axis=0), lw=0.7, color=colour)
            ax_m.set_xlim(time[0], time[-1])  # y-limits: shared, from data
            ax_m.tick_params(labelbottom=False)
            if i == 0 and label_col0:
                ax_m.set_ylabel("Magnetization\n" r"$m(t)$")
            else:
                ax_m.tick_params(
                    labelleft=False
                )  # sharey: not set_yticklabels

            n_spin = z_of_t.shape[0]
            # Upsample along time only, so rows (spins) are never blended.
            time_fine = np.linspace(time[0], time[-1], (len(time) - 1) * 8 + 1)
            z_fine = np.empty((n_spin, time_fine.size))
            for row in range(n_spin):
                z_fine[row] = np.interp(time_fine, time, z_of_t[row])
            # Rasterized: vector quads show seams between cells.
            ax_s.pcolormesh(
                time_fine,
                np.arange(1, n_spin + 1),
                z_fine,
                cmap=cmap_z,
                norm=norm_z,
                shading="nearest",
                rasterized=True,
            )
            ax_s.set_xlim(time[0], time[-1])
            ax_s.set_xlabel(r"Time $Jt$")
            if i == 0 and label_col0:
                ax_s.set_yticks(np.arange(1, n_spin + 1))
                ax_s.set_yticklabels(
                    [
                        str(k) if k in (1, n_spin) else ""
                        for k in range(1, n_spin + 1)
                    ]
                )
                ax_s.set_ylabel(r"Spin $i$")
            else:
                ax_s.set_yticks([1, n_spin])
                ax_s.set_yticklabels([])

            if use_hamming:
                # Hamming-distance projection: x is the distance from the initial state.
                sec_h, A_h = col["sec_idx"], col["A"]
                if hil_top_k is not None and len(sec_h) > hil_top_k:
                    # The hil_top_k most-occupied states (initial state always kept) and the
                    # single-flip edges among them; ties go to states nearer the initial one.
                    ref = int(sec_h[alpha])
                    dist = np.array(
                        [(int(x) ^ ref).bit_count() for x in sec_h]
                    )
                    occ_key = np.where(occ >= OCC_FLOOR, occ, 0.0)
                    keep = np.lexsort((dist, -occ_key))[:hil_top_k]
                    if alpha not in keep:
                        keep[-1] = alpha
                    keep = np.sort(keep)
                    sec_h, A_h = sec_h[keep], A_h[np.ix_(keep, keep)]
                    occ = occ[keep]
                    alpha = int(np.flatnonzero(keep == alpha)[0])
                XY, EDGES = hamming_projection_layout(sec_h, A_h, alpha)
            elif shared_hilbert is not None:
                XY, EDGES = shared_hilbert
            else:
                XY, EDGES = hilbert_layout(
                    col["sec_idx"],
                    col["A"],
                    panel_aspect,
                    alpha,
                    near_giant=(i in near_giant_cols),
                )
            ax_h.axis("off")
            if len(EDGES):
                ax_h.add_collection(
                    plt.matplotlib.collections.LineCollection(
                        XY[EDGES], colors="0.88", linewidths=0.25, zorder=1
                    )
                )
            vis = occ >= OCC_FLOOR
            ax_h.scatter(
                XY[~vis, 0], XY[~vis, 1], s=1.4, c="0.9", linewidth=0, zorder=2
            )
            ax_h.scatter(
                XY[vis, 0],
                XY[vis, 1],
                s=4.0,
                c=occ[vis],
                cmap=cmap_occ,
                norm=norm_occ,
                linewidth=0,
                zorder=3,
            )
            ax_h.scatter(
                XY[alpha, 0],
                XY[alpha, 1],
                s=64,
                facecolor="none",
                edgecolor=colour,
                linewidth=1.2,
                zorder=5,
            )
            if use_hamming:
                # No equal aspect: x and y are different quantities.
                x_span = XY[:, 0].max() - XY[:, 0].min()
                y_span = max(XY[:, 1].max() - XY[:, 1].min(), 1e-9)
                ax_h.set_xlim(
                    XY[:, 0].min() - 0.06 * max(x_span, 1e-9),
                    XY[:, 0].max() + 0.06 * max(x_span, 1e-9),
                )
                ax_h.set_ylim(
                    XY[:, 1].min() - 0.02 * y_span,
                    XY[:, 1].max() + 0.02 * y_span,
                )
                ax_h.diagram_ymax = XY[:, 1].max()
            else:
                # Pad for the initial-state ring marker.
                pad = 0.10
                ax_h.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
                ax_h.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
                ax_h.set_aspect("equal", adjustable="box")
            ax_h.margins(0)

    HIL_TOP_K = len(ring_present[0]["sec_idx"])
    render_columns(
        net_cols,
        lay_net,
        ax_net[:N_ER],
        ax_fid[:N_ER],
        ax_mag[:N_ER],
        ax_spin[:N_ER],
        ax_hil[:N_ER],
        use_hamming=True,
        hil_top_k=HIL_TOP_K,
    )
    render_columns(
        ring_present,
        lay_ring,
        ax_net[N_ER:],
        ax_fid[N_ER:],
        ax_mag[N_ER:],
        ax_spin[N_ER:],
        ax_hil[N_ER:],
        label_col0=False,
        use_hamming=True,
    )
    fig.align_ylabels([ax_fid[0], ax_mag[0], ax_spin[0]])

    FigureCanvasAgg(fig)
    fig.savefig(io.BytesIO(), format="png")
    fig.set_constrained_layout(False)
    fig.savefig(io.BytesIO(), format="png")
    renderer = fig.canvas.get_renderer()

    for ax in ax_net:
        ax.set_xlim(-0.6, (N - 1) * 0.9 + 0.6)
        ax.set_ylim(*SHARED_YLIM)
        ax.set_aspect("equal", adjustable="box", anchor="S")
        ax.set_ylim(
            *SHARED_YLIM
        )  # re-assert after aspect so arrows are not clipped

    fig.canvas.draw()

    net_pos = [ax.get_position() for ax in ax_net]
    y0 = min(p.y0 for p in net_pos)
    height = min(p.height for p in net_pos)
    width = min(p.width for p in net_pos)
    for ax, p in zip(ax_net, net_pos):
        x0 = p.x0 + 0.5 * (p.width - width)
        ax.set_position([x0, y0, width, height])

    # Equal heights for the F(t), m(t) and <Z_i> rows.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_h = (
        ax_mag[0].yaxis.label.get_window_extent(renderer).height
        / fig.bbox.height
    )
    ROW_GAP_PT = 8.5  # gap between the F/m/<Z> rows
    row_gap = ROW_GAP_PT * fig.dpi / 72 / fig.bbox.height
    fid_top = max(ax.get_position().y1 for ax in ax_fid)
    for k, row in enumerate((ax_fid, ax_mag, ax_spin)):
        y0_row = fid_top - (k + 1) * row_h - k * row_gap
        for ax in row:
            p = ax.get_position()
            ax.set_position([p.x0, y0_row, p.width, row_h])

    for ax in (ax_cbar_z, ax_cbar_o):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ps = ax_spin[0].get_position()
    ph = ax_hil[0].get_position()
    last_x1 = max(ax.get_position().x1 for ax in ax_spin)
    cb_gap = 0.006
    cb_host_w = 0.055
    cb_x0 = min(last_x1 + cb_gap, 0.995 - cb_host_w)
    ax_cbar_z.set_position([cb_x0, ps.y0, cb_host_w, ps.height])
    ax_cbar_o.set_position([cb_x0, ph.y0, cb_host_w, ph.height])

    GAP_PX = 2

    def _measure_letter_height(renderer):
        t = fig.text(
            0,
            0,
            "Wy",
            fontfamily="Helvetica",
            fontweight="bold" if LABEL_BOLD else "normal",
            fontsize=LABEL_FONT,
            va="bottom",
            ha="left",
        )
        fig.canvas.draw()
        h = t.get_window_extent(renderer).height
        t.remove()
        return h

    def _shift_axes(axes, dy_frac):
        for ax in axes:
            p = ax.get_position()
            ax.set_position([p.x0, p.y0 + dy_frac, p.width, p.height])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letter_h = _measure_letter_height(renderer)
    H = fig.bbox.height
    gap_frac = GAP_PX / H
    lh_frac = letter_h / H
    gutter_frac = gap_frac + lh_frac  # room for one letter + small gap

    title_top_px = max(
        ax.title.get_window_extent(renderer).y1 for ax in ax_net
    )
    block_h = BLOCK_TITLE_FONTSIZE * fig.dpi / 72.0
    need_top_px = (
        title_top_px + GAP_PX + letter_h + GAP_PX + 1.6 * block_h + GAP_PX
    )
    if need_top_px > H - GAP_PX:
        dy = (need_top_px - (H - GAP_PX)) / H
        _shift_axes(
            list(ax_net)
            + list(ax_fid)
            + list(ax_mag)
            + list(ax_spin)
            + list(ax_hil)
            + [ax_cbar_z, ax_cbar_o],
            -dy,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    net_bottom = min(ax.get_window_extent(renderer).y0 for ax in ax_net)
    fid_top = max(ax.get_window_extent(renderer).y1 for ax in ax_fid)
    have = (net_bottom - fid_top) / H
    if have < gutter_frac:
        need = gutter_frac - have
        _shift_axes(ax_net, need / 2)
        _shift_axes(
            list(ax_fid)
            + list(ax_mag)
            + list(ax_spin)
            + list(ax_hil)
            + [ax_cbar_z, ax_cbar_o],
            -need / 2,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    spin_xlabel_y0 = []
    for ax in ax_spin:
        bb = ax.xaxis.get_tightbbox(renderer) or ax.get_window_extent(renderer)
        spin_xlabel_y0.append(bb.y0)
    xlabel_bottom = min(spin_xlabel_y0)
    hil_pos = [ax.get_position() for ax in ax_hil]
    slot_top = (xlabel_bottom - GAP_PX - letter_h - GAP_PX) / H
    HIL_H_IN = 1.0  # Hilbert-row height (inches); the rest is trimmed
    slot_y0 = max(0.01, slot_top - HIL_H_IN * fig.dpi / H)
    slot_h = max(0.08, slot_top - slot_y0)
    hil_slots = [(p.x0, p.width) for p in hil_pos]
    for ax, p in zip(ax_hil, hil_pos):
        ax.set_position([p.x0, slot_y0, p.width, slot_h])
    fig.canvas.draw()
    hil_pos = [ax.get_position() for ax in ax_hil]
    for ax, p, (sx0, sw) in zip(ax_hil, hil_pos, hil_slots):
        x0 = sx0 + 0.5 * (sw - p.width)
        y0 = max(0.01, slot_top - p.height)
        ax.set_position([x0, y0, p.width, p.height])
    hil_pos = [ax.get_position() for ax in ax_hil]
    hil_y0 = min(p.y0 for p in hil_pos)
    hil_h = max(p.y0 + p.height for p in hil_pos) - hil_y0
    ps = ax_spin[0].get_position()
    last_x1 = max(ax.get_position().x1 for ax in ax_spin)
    cb_x0 = min(last_x1 + 0.006, 0.995 - cb_host_w)
    ax_cbar_z.set_position([cb_x0, ps.y0, cb_host_w, ps.height])
    ax_cbar_o.set_position([cb_x0, hil_y0, cb_host_w, hil_h])

    CBAR_FRAC = 0.38
    CBAR_LEFT = 0.02

    def _cbar_in(ax_host, mappable, label, extend=None):
        bb = ax_host.get_position()
        w = bb.width * CBAR_FRAC
        x0 = bb.x0 + bb.width * CBAR_LEFT
        cax = fig.add_axes([x0, bb.y0, w, bb.height])
        cb = fig.colorbar(mappable, cax=cax, extend=extend)
        cb.set_label(label)
        return cb

    cb_z = _cbar_in(
        ax_cbar_z,
        plt.cm.ScalarMappable(norm=norm_z, cmap=cmap_z),
        r"$\langle \hat{Z}_i(t) \rangle$",
    )  # short: heatmap row is short
    cb_z.set_ticks([-1, 0, 1])
    cb_o = _cbar_in(
        ax_cbar_o,
        plt.cm.ScalarMappable(norm=norm_occ, cmap=cmap_occ),
        "Time-averaged occupation",
        extend="both",
    )
    cb_o.set_ticks([1e-4, 1e-3, 1e-2, 1e-1])

    # Fig. 3's letters start at 'b': panel a is the flowchart (fig-03a).
    labels = journal_labels(
        "abcdefghijklmnopqrstuvwxyz"[cfg["first_letter"] :]
    )

    def _put(letter, x_px, y_px, va="bottom"):
        return fig.text(
            x_px / fig.bbox.width,
            y_px / fig.bbox.height,
            letter,
            fontfamily="Helvetica",
            fontweight="bold" if LABEL_BOLD else "normal",
            fontsize=LABEL_FONT,
            va=va,
            ha="left",
            clip_on=False,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letter_h = _measure_letter_height(renderer)
    H = fig.bbox.height

    col_x = []
    for c in range(total_cols):
        xs = []
        for ax in (ax_fid[c], ax_mag[c], ax_spin[c]):
            bb = ax.yaxis.get_tightbbox(renderer) or ax.get_window_extent(
                renderer
            )
            xs.append(bb.x0)
        col_x.append(min(xs))

    def _place_above(content_top_px, above_bottom_px, x_px, letter):
        """Place letter above content_top, fully below above_bottom.

        Pin the letter top just under the panel above (x-label / axes bottom)
        so that clearance rule is guaranteed; the gutter-reservation step
        above ensures this still sits above the content.
        """
        y_top = above_bottom_px - GAP_PX
        if y_top - letter_h < content_top_px + GAP_PX:
            y_top = content_top_px + GAP_PX + letter_h
        _put(letter, x_px, y_top, va="top")
        return y_top

    letter_tops_net = []
    net_letter_artists = []
    li = 0
    for c, ax in enumerate(ax_net):
        title_bb = ax.title.get_window_extent(renderer)
        y_bottom = title_bb.y1 + GAP_PX
        art = _put(labels[li], col_x[c], y_bottom, va="bottom")
        net_letter_artists.append(art)
        letter_tops_net.append(y_bottom + letter_h)
        li += 1

    block_title_artists = []
    for ax_slice, text, tops in (
        (ax_net[:N_ER], BLOCK_TITLES[0], letter_tops_net[:N_ER]),
        (ax_net[N_ER:], BLOCK_TITLES[1], letter_tops_net[N_ER:]),
    ):
        xs = [ax.get_window_extent(renderer).x0 for ax in ax_slice]
        xe = [ax.get_window_extent(renderer).x1 for ax in ax_slice]
        y_frac = (max(tops) + GAP_PX) / fig.bbox.height
        bt = fig.text(
            (min(xs) + max(xe)) / 2 / fig.bbox.width,
            y_frac,
            text,
            ha="center",
            va="bottom",
            fontdict={
                "family": "Helvetica",
                "weight": "regular",
                "style": "normal",
                "size": BLOCK_TITLE_FONTSIZE,
            },
        )
        block_title_artists.append(bt)

    for c, ax in enumerate(ax_fid):
        content_top = ax.get_window_extent(renderer).y1
        above_bottom = ax_net[c].get_window_extent(renderer).y0
        _place_above(content_top, above_bottom, col_x[c], labels[li])
        li += 1

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for c, ax in enumerate(ax_hil):
        tops = [
            bb.y1
            for bb in _extents(ax.collections, renderer)
            if np.isfinite(bb.y1) and bb.height > 1
        ]
        diagram_top = max(tops) if tops else ax.get_window_extent(renderer).y1
        if hasattr(ax, "diagram_ymax"):
            diagram_top = (
                ax.transData.transform((0, ax.diagram_ymax))[1]
                + np.sqrt(4.0) / 2 * fig.dpi / 72
            )
        spin_bb = ax_spin[c].xaxis.get_tightbbox(renderer) or ax_spin[
            c
        ].get_window_extent(renderer)
        y_letter = diagram_top
        if y_letter + letter_h > spin_bb.y0 - GAP_PX:
            y_letter = spin_bb.y0 - GAP_PX - letter_h
        _put(labels[li], col_x[c], y_letter, va="bottom")
        li += 1

    def _shift_all(dx_frac, dy_frac):
        for ax in fig.axes:
            p = ax.get_position()
            ax.set_position(
                [p.x0 + dx_frac, p.y0 + dy_frac, p.width, p.height]
            )
        for t in fig.texts:
            x, y = t.get_position()
            t.set_position((x + dx_frac, y + dy_frac))

    def _data_axes():
        return (
            list(ax_net)
            + list(ax_fid)
            + list(ax_mag)
            + list(ax_spin)
            + list(ax_hil)
        )

    def _remap_data_x(x0_min, x1_max, new_x0, new_x1):
        """Affine-map every data axis (and panel letters) into [new_x0, new_x1]."""
        span = x1_max - x0_min
        new_span = new_x1 - new_x0
        if span <= 0 or new_span <= 0:
            return

        def remap(x):
            return new_x0 + (x - x0_min) / span * new_span

        for ax in _data_axes():
            p = ax.get_position()
            a0, a1 = remap(p.x0), remap(p.x0 + p.width)
            ax.set_position([a0, p.y0, a1 - a0, p.height])

        block_set = set(block_title_artists)
        for t in fig.texts:
            if t in block_set:
                continue
            x, y = t.get_position()
            if x0_min - 0.08 <= x <= x1_max + 0.02:
                t.set_position((remap(x), y))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        H = fig.bbox.height
        for bt, ax_slice, letters in (
            (block_title_artists[0], ax_net[:N_ER], net_letter_artists[:N_ER]),
            (block_title_artists[1], ax_net[N_ER:], net_letter_artists[N_ER:]),
        ):
            xs = [ax.get_position().x0 for ax in ax_slice]
            xe = [
                ax.get_position().x0 + ax.get_position().width
                for ax in ax_slice
            ]
            letter_top = max(t.get_window_extent(renderer).y1 for t in letters)
            bt.set_position(
                ((min(xs) + max(xe)) / 2, (letter_top + GAP_PX) / H)
            )

    def _pack_horizontal():
        """Flush y-labels to the left edge; keep colorbar labels inside the canvas."""
        gap = 0.005
        bar_w = 0.0055
        left_pad_px = 1
        right_pad_px = 12

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        W = fig.bbox.width

        def _left_overhang():
            # Left margin from the y-axis labels only, not the panel letters.
            boxes = []
            for ax in (ax_fid[0], ax_mag[0], ax_spin[0], ax_hil[0]):
                boxes += _extents(
                    [ax.yaxis.label, *ax.get_yticklabels()], renderer
                )
            ax0 = ax_fid[0].get_window_extent(renderer)
            left_content = min(b.x0 for b in boxes)
            return max(0.0, ax0.x0 - left_content)

        def _cbar_right_extent():
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            max_x1 = 0.0
            for cb in (cb_z, cb_o):
                max_x1 = max(max_x1, cb.ax.yaxis.label.get_window_extent(r).x1)
                for tick in cb.ax.get_yticklabels():
                    max_x1 = max(max_x1, tick.get_window_extent(r).x1)
            return max_x1

        def _place_cbars(label_w):
            fig.canvas.draw()
            ps = ax_spin[0].get_position()
            last_x1 = max(ax.get_position().x1 for ax in ax_spin)
            x0 = last_x1 + gap
            total_w = bar_w + label_w

            hil_pos = [ax.get_position() for ax in ax_hil]
            hy0 = min(p.y0 for p in hil_pos)
            hh = max(p.y0 + p.height for p in hil_pos) - hy0
            ax_cbar_z.set_position([x0, ps.y0, total_w, ps.height])
            ax_cbar_o.set_position([x0, hy0, total_w, hh])
            for host in (ax_cbar_z, ax_cbar_o):
                host.set_visible(False)
            fs = ax_spin[0].yaxis.get_ticklabels()[0].get_fontsize()
            for host, cb in ((ax_cbar_z, cb_z), (ax_cbar_o, cb_o)):
                hb = host.get_position()
                cb.ax.set_aspect("auto")
                cb.ax.set_position([hb.x0, hb.y0, bar_w, hb.height])
                cb.ax.yaxis.set_label_position("right")
                cb.ax.yaxis.set_ticks_position("right")
                cb.ax.yaxis.label.set_size(5)  # 5 pt colorbar labels
                cb.set_label(cb.ax.yaxis.label.get_text(), labelpad=0.5)
                cb.ax.tick_params(axis="y", pad=0.5, labelsize=fs)
                cb.outline.set_visible(False)
                cb.outline.set_linewidth(0)
                cb.ax.set_frame_on(False)
                for spine in cb.ax.spines.values():
                    spine.set_visible(False)

        overhang = _left_overhang()
        left_frac = (overhang + left_pad_px) / W
        label_w = 0.055
        data_right = 1.0 - (gap + bar_w + label_w + right_pad_px / W)

        x0_min = min(ax.get_position().x0 for ax in _data_axes())
        x1_max = max(ax.get_position().x1 for ax in _data_axes())
        _remap_data_x(x0_min, x1_max, left_frac, data_right)
        _place_cbars(label_w)

        max_x1 = _cbar_right_extent()
        target = W - right_pad_px
        for _ in range(4):
            if max_x1 <= target:
                break
            delta = (max_x1 - target) / W + 0.003
            x0_min = min(ax.get_position().x0 for ax in _data_axes())
            x1_max = max(ax.get_position().x1 for ax in _data_axes())
            _remap_data_x(x0_min, x1_max, x0_min, x1_max - delta)
            label_w = max(0.035, label_w + delta)
            _place_cbars(label_w)
            max_x1 = _cbar_right_extent()

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        y_boxes = []
        for ax in (ax_fid[0], ax_mag[0], ax_spin[0], ax_hil[0]):
            y_boxes += _extents(
                [ax.yaxis.label, *ax.get_yticklabels()], renderer
            )
        left_ink = min(b.x0 for b in y_boxes)
        extra = left_ink - left_pad_px
        if extra > 1.0:
            shift = extra / W
            for ax in fig.axes:
                p = ax.get_position()
                ax.set_position([p.x0 - shift, p.y0, p.width, p.height])
            for t in fig.texts:
                x, y = t.get_position()
                t.set_position((x - shift, y))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for t in fig.texts:
            if len(t.get_text()) != 1:
                continue
            bb = t.get_window_extent(renderer)
            if bb.x0 < 0.5:
                x, y = t.get_position()
                t.set_position((x + (0.5 - bb.x0) / W, y))

    def _snap_block_titles(lift_to_top=False):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        H = fig.bbox.height
        for bt, ax_slice, letters in (
            (block_title_artists[0], ax_net[:N_ER], net_letter_artists[:N_ER]),
            (block_title_artists[1], ax_net[N_ER:], net_letter_artists[N_ER:]),
        ):
            xs = [ax.get_position().x0 for ax in ax_slice]
            xe = [
                ax.get_position().x0 + ax.get_position().width
                for ax in ax_slice
            ]
            letter_top = max(t.get_window_extent(renderer).y1 for t in letters)
            bt.set_va("bottom")
            bt.set_position(
                ((min(xs) + max(xe)) / 2, (letter_top + GAP_PX) / H)
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_top = max(
            bt.get_window_extent(renderer).y1 for bt in block_title_artists
        )
        top_pad_px = 4
        if lift_to_top:
            lift = (H - top_pad_px) - title_top
            if lift > 1:
                dy = lift / H
                for ax in fig.axes:
                    p = ax.get_position()
                    ax.set_position([p.x0, p.y0 + dy, p.width, p.height])
                for t in fig.texts:
                    x, y = t.get_position()
                    t.set_position((x, y + dy))
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                title_top = max(
                    bt.get_window_extent(renderer).y1
                    for bt in block_title_artists
                )
        overflow = title_top + top_pad_px - H
        if overflow > 0:
            dy = overflow / H
            for ax in fig.axes:
                p = ax.get_position()
                ax.set_position([p.x0, p.y0 - dy, p.width, p.height])
            for t in fig.texts:
                x, y = t.get_position()
                t.set_position((x, y - dy))
            fig.canvas.draw()

    _snap_block_titles(lift_to_top=True)

    _pack_horizontal()
    _snap_block_titles(lift_to_top=True)
    for ax in list(ax_hil) + [ax_cbar_o]:
        p = ax.get_position()
        if p.y0 < 0.01:
            ax.set_position([p.x0, 0.01, p.width, p.height])

    # Saved at MAX_FIG_WIDTH without bbox_inches='tight': stretch the
    # columns until the colorbar labels reach the right edge.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W = fig.bbox.width
    _edge_boxes = []
    for cb in (cb_z, cb_o):
        _edge_boxes.append(cb.ax.yaxis.label.get_window_extent(renderer))
        for tick in cb.ax.get_yticklabels():
            _edge_boxes.append(tick.get_window_extent(renderer))
    right_ink_px = max(b.x1 for b in _edge_boxes)
    target_px = W - 6  # matches the near-zero left margin (left_pad_px = 1)
    if right_ink_px < target_px:
        delta = (target_px - right_ink_px) / W
        x0_min = min(ax.get_position().x0 for ax in _data_axes())
        x1_max = max(ax.get_position().x1 for ax in _data_axes())
        _remap_data_x(x0_min, x1_max, x0_min, x1_max + delta)
        for cb in (cb_z, cb_o):
            p = cb.ax.get_position()
            cb.ax.set_position([p.x0 + delta, p.y0, p.width, p.height])
        fig.canvas.draw()

    def _trim_empty_bottom(top_pad_px=4, bottom_pad_px=8):
        """Drop unused canvas below the bottom row (no vertical stretch)."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        H = fig.bbox.height
        boxes = _extents(
            [ax for ax in fig.axes if ax.get_visible()], renderer
        ) + _extents(fig.texts, renderer)
        boxes = [b for b in boxes if np.isfinite(b.y0) and np.isfinite(b.y1)]
        if not boxes:
            return
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        if y0 <= bottom_pad_px + 1:
            return
        new_H_px = (y1 - y0) + top_pad_px + bottom_pad_px
        if new_H_px >= H - 1:
            return

        def remap(y_frac):
            y_px = y_frac * H
            return ((y_px - y0) + bottom_pad_px) / new_H_px

        for ax in fig.axes:
            p = ax.get_position()
            ny0 = remap(p.y0)
            ny1 = remap(p.y0 + p.height)
            ax.set_position([p.x0, ny0, p.width, max(ny1 - ny0, 1e-4)])
        for t in fig.texts:
            x, y = t.get_position()
            t.set_position((x, remap(y)))
        w_in, _ = fig.get_size_inches()
        fig.set_size_inches(w_in, new_H_px / fig.dpi, forward=True)
        fig.canvas.draw()

    _trim_empty_bottom()
    for cb in (cb_z, cb_o):
        cb.outline.set_visible(False)
        cb.outline.set_linewidth(0)
        cb.ax.set_frame_on(False)
        for spine in cb.ax.spines.values():
            spine.set_visible(False)
    # Hide colorbar frames, including the extend triangles' edges.
    for cb in (cb_z, cb_o):
        cb.outline.set_visible(False)
        cb.outline.set_linewidth(0)
        cb.ax.set_frame_on(False)
        cb.ax.patch.set_visible(False)
        for spine in cb.ax.spines.values():
            spine.set_visible(False)
    fig.canvas.draw()
    return fig
