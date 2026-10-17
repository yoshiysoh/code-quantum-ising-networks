"""Render frames [start, end) of Supplementary Movie 1 (--fig fig-03) or 2
(--fig fig-s03): the columns of Fig. 3 or Extended Data Fig. 3 evolving in
time.

Each frame depends only on its index, so frame ranges can be rendered in
parallel and joined with `ffmpeg -f concat`.

Run: python3 render_movie.py --fig fig-03 --start 0 --end 1800 \
         --out figures/movie-01.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import colorcet as cc
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec

from qising_networks import scar_metrics as sm
from qising_networks.plots import dynamics
from qising_networks.plots.common import CLASS_COLORS, CLASS_LABELS
from qising_networks.plots.figure import MAX_FIG_WIDTH
from qising_networks.scar_metrics import DISPLAY_TIME_MAX

OCC_FLOOR = dynamics.OCC_FLOOR
FPS = 30
DURATION_S = 60
N_FRAMES = FPS * DURATION_S

MODEL_OF = {"fig-03": "ferro", "fig-s03": "pmJ"}  # Movies 1 and 2


def build_columns(model):
    """(data, network columns, ring columns), as in the figure but without
    network classes that have no candidate."""
    data = dynamics.load_examples(model)
    net_cols = [
        {
            "valid": True,
            "cls": c,
            "alpha": int(data["net_alphas"][c]),
            "sec_idx": data["net_sec_idx"],
            "A": data["net_A"],
            "states": data["net_states"],
        }
        for c in dynamics.CLASS_ORDER
        if data["net_alphas"][c] >= 0
    ]
    ring_scar = next(
        col
        for col in data["ring_cols"]
        if col["valid"] and col["cls"] == sm.SCAR
    )
    ring_thermal = dynamics.ring_thermal_column(model, data)
    return data, net_cols, [ring_scar, ring_thermal]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", choices=list(MODEL_OF), required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_FRAMES)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data, net_cols, ring_present = build_columns(MODEL_OF[args.fig])
    all_cols = net_cols + ring_present
    is_net_col = [True] * len(net_cols) + [False] * len(ring_present)
    total_cols = len(all_cols)
    print(
        f"{args.fig}: {total_cols} columns "
        f"({len(net_cols)} network + {len(ring_present)} ring)"
    )

    frame_times = np.linspace(0.0, DISPLAY_TIME_MAX, N_FRAMES)
    H_net = dynamics.sector_hamiltonian(
        data["net_im"], data["net_h"], data["net_sec_idx"]
    )
    H_ring = dynamics.sector_hamiltonian(
        data["ring_im"], data["ring_h"], ring_present[0]["sec_idx"]
    )
    evals_net, evecs_net = np.linalg.eigh(H_net)
    evals_ring, evecs_ring = np.linalg.eigh(H_ring)

    prob_full, fid_full, z_full, m_full = [], [], [], []
    for col, is_net in zip(all_cols, is_net_col):
        evals, evecs = (
            (evals_net, evecs_net) if is_net else (evals_ring, evecs_ring)
        )
        prob = dynamics.probabilities_at(
            evals, evecs, col["alpha"], frame_times
        )
        prob_full.append(prob)
        fid_full.append(prob[col["alpha"], :])
        z_full.append(col["states"].T @ prob)
        m_full.append(z_full[-1].mean(axis=0))  # m(t) = (1/N) sum_i <Z_i>

    # Network Hilbert panels: the HIL_TOP_K most-occupied states, as in the figure.
    HIL_TOP_K = len(ring_present[0]["sec_idx"])
    hil_cols = []
    for c, (col, is_net) in enumerate(zip(all_cols, is_net_col)):
        sec_h, A_h, alpha_h = col["sec_idx"], col["A"], col["alpha"]
        keep = np.arange(len(sec_h))
        if is_net and len(sec_h) > HIL_TOP_K:
            occ = prob_full[c].mean(axis=1)
            ref = int(sec_h[alpha_h])
            dist = np.array([(int(x) ^ ref).bit_count() for x in sec_h])
            occ_key = np.where(occ >= OCC_FLOOR, occ, 0.0)
            keep = np.lexsort((dist, -occ_key))[:HIL_TOP_K]
            if alpha_h not in keep:
                keep[-1] = alpha_h
            keep = np.sort(keep)
            sec_h, A_h = sec_h[keep], A_h[np.ix_(keep, keep)]
            alpha_h = int(np.flatnonzero(keep == col["alpha"])[0])
        hil_cols.append((sec_h, A_h, alpha_h, keep))

    lay_net = dynamics.network_layout(
        data["net_im"], data["net_h"], use_rcm=True
    )
    lay_ring = dynamics.network_layout(
        data["ring_im"], data["ring_h"], use_rcm=False
    )
    y_lo = min(
        dynamics.network_ylim(lay_net)[0], dynamics.network_ylim(lay_ring)[0]
    )
    y_hi = max(
        dynamics.network_ylim(lay_net)[1], dynamics.network_ylim(lay_ring)[1]
    )
    SHARED_YLIM = (y_lo, y_hi)

    norm_z = mcolors.Normalize(vmin=-1, vmax=1)
    norm_occ = mcolors.LogNorm(vmin=1e-4, vmax=0.1, clip=False)
    cmap_z = cc.cm.CET_D1A
    # A sequential colormap reads better frame by frame than the figure's CET_R2.
    cmap_occ = cc.cm.CET_L18.copy()
    # Unoccupied nodes white, like the background.
    cmap_occ.set_under("white")
    cmap_occ.set_over(cmap_occ(1.0))

    # F(t), m(t) and <Z_i> rows share one height, as in the static figure.
    HEIGHT_RATIOS = [0.80, 0.40, 0.40, 0.40, 0.90]
    UNIT = 5.0 / sum([0.60, 0.40, 1.00, 1.05])
    FIG_H = UNIT * sum(HEIGHT_RATIOS) * 0.92

    fig_m = plt.figure(figsize=(MAX_FIG_WIDTH, FIG_H))
    gs_m = GridSpec(
        5,
        total_cols + 1,
        figure=fig_m,
        height_ratios=HEIGHT_RATIOS,
        width_ratios=[1.0] * total_cols + [0.06],
    )
    fig_m.set_constrained_layout_pads(
        h_pad=0.02, w_pad=0.01, hspace=0.06, wspace=0.05
    )
    fig_m.set_layout_engine("constrained")

    axn_m = [fig_m.add_subplot(gs_m[0, c]) for c in range(total_cols)]
    axf_m = [fig_m.add_subplot(gs_m[1, c]) for c in range(total_cols)]
    axm_m = [fig_m.add_subplot(gs_m[2, 0], sharex=axf_m[0])]
    axm_m += [
        fig_m.add_subplot(gs_m[2, c], sharex=axf_m[c], sharey=axm_m[0])
        for c in range(1, total_cols)
    ]
    axs_m = [
        fig_m.add_subplot(gs_m[3, c], sharex=axf_m[c])
        for c in range(total_cols)
    ]
    axh_m = [fig_m.add_subplot(gs_m[4, c]) for c in range(total_cols)]
    cax_z_m = fig_m.add_subplot(gs_m[3, -1])
    cax_o_m = fig_m.add_subplot(gs_m[4, -1])

    lines_m, mlines_m, meshes_m, scatters_m = [], [], [], []
    hilbert_axes_info = []  # (ax_h, XY, alpha_idx)
    for c, (col, is_net) in enumerate(zip(all_cols, is_net_col)):
        colour = CLASS_COLORS[col["cls"]]
        layout_m = lay_net if is_net else lay_ring
        dynamics.draw_network(
            axn_m[c], layout_m, col["states"][col["alpha"]], ylim=SHARED_YLIM
        )
        axn_m[c].set_title(CLASS_LABELS[col["cls"]], color=colour, pad=2)

        ax_f, ax_s, ax_h = axf_m[c], axs_m[c], axh_m[c]
        ax_f.set_xlim(frame_times[0], frame_times[-1])
        ax_f.set_ylim(-0.05, 1.10)
        ax_f.set_yticks([0, 1])
        ax_f.tick_params(labelbottom=False)
        (line,) = ax_f.plot([], [], lw=0.7, color=colour)
        lines_m.append(line)
        if c == 0:
            ax_f.set_ylabel(r"Fidelity $F(t)$")
        else:
            ax_f.set_yticklabels([])

        ax_m = axm_m[c]
        ax_m.set_xlim(frame_times[0], frame_times[-1])
        ax_m.tick_params(labelbottom=False)
        # Full trace drawn invisibly so the y-limits are fixed from the first frame.
        ax_m.plot(frame_times, m_full[c], lw=0.7, alpha=0)
        (mline,) = ax_m.plot([], [], lw=0.7, color=colour)
        mlines_m.append(mline)
        if c == 0:
            ax_m.set_ylabel("Magnetization\n" r"$m(t)$")
        else:
            ax_m.tick_params(labelleft=False)  # sharey: not set_yticklabels

        n_spin = z_full[c].shape[0]
        z_masked0 = np.ma.masked_array(z_full[c], mask=True)
        mesh = ax_s.pcolormesh(
            frame_times,
            np.arange(1, n_spin + 1),
            z_masked0,
            cmap=cmap_z,
            norm=norm_z,
            shading="nearest",
            rasterized=True,
        )
        meshes_m.append(mesh)
        ax_s.set_xlim(frame_times[0], frame_times[-1])
        ax_s.set_xlabel(r"Time $Jt$")
        if c == 0:
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

        sec_h, A_h, alpha_h, keep = hil_cols[c]
        XY, EDGES = dynamics.hamming_projection_layout(sec_h, A_h, alpha_h)
        ax_h.axis("off")
        if len(EDGES):
            ax_h.add_collection(
                plt.matplotlib.collections.LineCollection(
                    XY[EDGES], colors="0.88", linewidths=0.25, zorder=1
                )
            )
        sc = ax_h.scatter(
            XY[:, 0],
            XY[:, 1],
            s=4.0,
            c=np.maximum(prob_full[c][keep, 0], 1e-6),
            cmap=cmap_occ,
            norm=norm_occ,
            linewidth=0,
            zorder=3,
        )
        ax_h.scatter(
            XY[alpha_h, 0],
            XY[alpha_h, 1],
            s=64,
            facecolor="none",
            edgecolor=colour,
            linewidth=1.2,
            zorder=5,
        )
        # No equal aspect: x and y are different quantities.
        x_span = XY[:, 0].max() - XY[:, 0].min()
        y_span = max(XY[:, 1].max() - XY[:, 1].min(), 1e-9)
        ax_h.set_xlim(
            XY[:, 0].min() - 0.06 * max(x_span, 1e-9),
            XY[:, 0].max() + 0.06 * max(x_span, 1e-9),
        )
        ax_h.set_ylim(
            XY[:, 1].min() - 0.10 * y_span, XY[:, 1].max() + 0.10 * y_span
        )
        ax_h.margins(0)
        scatters_m.append(sc)
        hilbert_axes_info.append((ax_h, XY, alpha_h))

    fig_m.align_ylabels([axf_m[0], axm_m[0], axs_m[0]])
    cb_z = fig_m.colorbar(
        plt.cm.ScalarMappable(norm=norm_z, cmap=cmap_z),
        cax=cax_z_m,
        label=r"$\langle \hat{Z}_i(t) \rangle$",
    )
    cb_o = fig_m.colorbar(
        plt.cm.ScalarMappable(norm=norm_occ, cmap=cmap_occ),
        cax=cax_o_m,
        label=r"Occupation $\mid \langle \boldsymbol{s} "
        r"\mid \psi(t) \rangle \mid^2$",
        extend="both",
    )

    # Grow the limits so the initial-state ring marker is not clipped.
    fig_m.canvas.draw()
    # Turn off constrained layout, or every saved frame re-applies it.
    fig_m.set_constrained_layout(False)
    renderer = fig_m.canvas.get_renderer()
    MARKER_RADIUS_PT = 7.5  # ring marker radius + edge + margin, pt
    # Two passes: widening the limits changes the data-per-pixel scale.
    for _pass in range(2):
        fig_m.canvas.draw()
        renderer = fig_m.canvas.get_renderer()
        for ax_h, XY, alpha in hilbert_axes_info:
            bbox = ax_h.get_window_extent(renderer)
            xlo, xhi = ax_h.get_xlim()
            ylo, yhi = ax_h.get_ylim()
            px_per_data_x = bbox.width / (xhi - xlo)
            px_per_data_y = bbox.height / (yhi - ylo)
            radius_px = MARKER_RADIUS_PT * fig_m.dpi / 72.0
            need_x = radius_px / px_per_data_x
            need_y = radius_px / px_per_data_y
            ax_x, ay_y = XY[alpha]
            new_xlo = min(xlo, ax_x - need_x)
            new_xhi = max(xhi, ax_x + need_x)
            new_ylo = min(ylo, ay_y - need_y)
            new_yhi = max(yhi, ay_y + need_y)
            if (new_xlo, new_xhi, new_ylo, new_yhi) != (xlo, xhi, ylo, yhi):
                ax_h.set_xlim(new_xlo, new_xhi)
                ax_h.set_ylim(new_ylo, new_yhi)
    for cb in (cb_z, cb_o):
        cb.outline.set_visible(False)
        cb.outline.set_linewidth(0)
        cb.ax.yaxis.label.set_size(5)  # pt, as the figure

    # Shrink the canvas to the content once, leaving room for the time label;
    # axes positions do not change between frames.
    fig_m.canvas.draw()
    renderer = fig_m.canvas.get_renderer()
    _probe = fig_m.text(0.5, 0.5, "Jt = 000.0", ha="center", va="center")
    fig_m.canvas.draw()
    time_text_pad_px = _probe.get_window_extent(renderer).height * 1.3
    _probe.remove()

    top_pad_px, bottom_pad_px, side_pad_px = 10, 16, 8
    tight_boxes = [ax.get_tightbbox(renderer) for ax in fig_m.axes]
    x0 = min(b.x0 for b in tight_boxes) - side_pad_px
    x1 = max(b.x1 for b in tight_boxes) + side_pad_px
    y0 = min(b.y0 for b in tight_boxes) - bottom_pad_px
    y1 = max(b.y1 for b in tight_boxes) + top_pad_px + time_text_pad_px
    new_w_px, new_h_px = x1 - x0, y1 - y0

    for ax in fig_m.axes:
        old_box = ax.get_window_extent(renderer)
        ax.set_position(
            [
                (old_box.x0 - x0) / new_w_px,
                (old_box.y0 - y0) / new_h_px,
                old_box.width / new_w_px,
                old_box.height / new_h_px,
            ]
        )
    fig_m.set_size_inches(
        new_w_px / fig_m.dpi, new_h_px / fig_m.dpi, forward=True
    )
    fig_m.canvas.draw()

    time_text_m = fig_m.text(0.5, 0.995, "", ha="center", va="top")

    def _update(k):
        time_text_m.set_text(f"$Jt = {frame_times[k]:6.1f}$")
        for c in range(len(all_cols)):
            lines_m[c].set_data(frame_times[: k + 1], fid_full[c][: k + 1])
            mlines_m[c].set_data(frame_times[: k + 1], m_full[c][: k + 1])
            mask = np.zeros(len(frame_times), dtype=bool)
            mask[k + 1 :] = True
            z_masked = np.ma.masked_array(
                z_full[c], mask=np.broadcast_to(mask, z_full[c].shape)
            )
            meshes_m[c].set_array(z_masked.ravel())
            scatters_m[c].set_array(
                np.maximum(prob_full[c][hil_cols[c][3], k], 1e-6)
            )
        return []

    frame_range = range(args.start, args.end)
    print(f"rendering frames [{args.start}, {args.end}) -> {args.out}")
    ani_m = FuncAnimation(fig_m, _update, frames=frame_range, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ani_m.save(args.out, writer=FFMpegWriter(fps=FPS), dpi=300)
    print("saved", args.out, args.out.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
