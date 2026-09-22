#!/usr/bin/env python3
"""
plot_pair.py

Poster variant of plot_metric.py: two experiment folders in ONE figure,
side by side, sharing the same axes.

    +---------------------------------------------------------------+
    |      title of panel 1            title of panel 2             |
    | y |  [bars]                   |  [bars]                       |
    |   +-----------------------    +-----------------------        |
    |          GPUs                        GPUs                     |
    +---------------------------------------------------------------+

- same y scale for both panels, y ticks + y label drawn ONCE (left panel)
- same x positions for both panels
- no legend by default (use --legend bottom if you do want one)
- horizontal rules more marked than in the single-panel figures

Colors, bar widths/offsets, method ordering, filtering and auto-titles are
imported from plot_metric.py, so this stays in sync with the other figures.

Usage
-----
python plot_pair.py \
    ../experiments_frontier/wideresnet/cifar10/strongScaling \
    ../experiments_polaris/wideresnet/cifar10/strongScaling \
    --mode strong --scope all --metric time --global-batch 128 --png
    
python plot_pair.py \
    ../experiments_frontier_rccl/wideresnet/cifar10/strongScaling \
    ../experiments_polaris_nccl/wideresnet/cifar10/strongScaling \
    --mode strong --scope all --metric time --global-batch 128 --png
    
python plot_pair.py \
    ../experiments_frontier/wideresnet/cifar10/strongScaling \
    ../experiments_frontier_rccl/wideresnet/cifar10/strongScaling \
    --mode strong --scope all --metric time --global-batch 128 --png
       

python plot_pair.py \
    ../experiments_frontier/wideresnet/cifar10/weakScaling \
    ../experiments_polaris/wideresnet/cifar10/weakScaling \
    --mode weak --scope all --metric time --batch-per-rank 8 --png
    

python plot_pair.py \
    ../experiments_frontier_rccl/wideresnet/cifar10/weakScaling \
    ../experiments_polaris_nccl/wideresnet/cifar10/weakScaling \
    --mode weak --scope all --metric time --batch-per-rank 8 --png
"""
from __future__ import annotations
 
import argparse
import math
import textwrap
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
 
from exp_parser import load_experiment_folder, ensure_plot_dir
from plot_metric import (
    METHOD_COLORS,
    METHOD_GROUP,
    METHOD_ORDER,
    apply_style,
    detect_backend,
    family_offsets,
    filter_df,
    make_default_title,
    metric_spec,
    ordered_methods,
    prefer_online_rate8,
    prepare_batch_columns,
    validate_for_scope,
)
 
 
# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
 
def load_panel(root: Path, args, value_col: str) -> tuple[pd.DataFrame, str]:
    """Load + filter one experiment folder and build its panel title."""
    df = load_experiment_folder(root)
    if df.empty:
        raise SystemExit(f"No data found in {root}")
 
    df = prepare_batch_columns(df)
    df = prefer_online_rate8(df)
 
    df = filter_df(
        df,
        mode=args.mode,
        global_batch=args.global_batch,
        batch_per_rank=args.batch_per_rank,
        gpus=args.gpus if args.scope == "fixed" else None,
    )
 
    needed = ["method", "ranks", value_col]
    needed.append("global_batch" if args.mode == "strong" else "batch_per_rank")
    df = df.dropna(subset=needed)
 
    if df.empty:
        raise SystemExit(f"No rows left after filtering for {root}")
 
    meta = validate_for_scope(df, args.mode, args.scope)
    backend = args.backend or detect_backend(root, df)
 
    title = make_default_title(
        root,
        args.mode,
        args.scope,
        gpus=args.gpus,
        global_batch=meta["global_batch"],
        batch_per_rank=meta["batch_per_rank"],
        varying_batches=meta["varying_batches"],
        backend=backend,
    )
    return df, title
 
 
def aggregate(df: pd.DataFrame, x_col: str, value_col: str) -> pd.DataFrame:
    return df.groupby([x_col, "method"], as_index=False)[value_col].median()
 
 
# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
 
def slot_key(method):
    """
    Methods that plot_metric.family_offsets() puts at the SAME offset share one
    bar slot. That is how 'Ring+ZFP online (rate:10)' and '(rate:8)' are
    treated as one series (rate 8 replaced rate 10 where it was re-run).
    """
    _, legacy = family_offsets()
    return ("offset", legacy[method]) if method in legacy else ("name", method)
 
 
def compute_offsets(methods, group_fill=0.86, family_gap=0.9):
    """
    Wide-bar replacement for plot_metric.family_offsets().
 
    Same idea (bars of one family touch each other, a gap separates families,
    rate:10 and rate:8 share a slot) but the whole group is stretched to fill
    `group_fill` of the x slot, so the bars are as fat as they can be for the
    number of slots present. Returns (width, {method: offset}).
 
    group_fill : fraction of the 1.0-wide x slot the group occupies
    family_gap : gap between families, in units of one bar width
    """
    # unique slots, in method order
    slots, slot_family = [], {}
    for m in methods:
        k = slot_key(m)
        if k not in slot_family:
            slots.append(k)
            slot_family[k] = METHOD_GROUP.get(m, "other")
 
    if not slots:
        return 0.1, {}
 
    fams = []
    for k in slots:
        if slot_family[k] not in fams:
            fams.append(slot_family[k])
 
    units = len(slots) + max(0, len(fams) - 1) * family_gap
    width = group_fill / units
 
    slot_offset = {}
    cursor = -group_fill / 2.0
    prev_fam = None
    for k in slots:
        fam = slot_family[k]
        if prev_fam is not None and fam != prev_fam:
            cursor += family_gap * width
        slot_offset[k] = cursor + width / 2.0
        cursor += width
        prev_fam = fam
 
    return width, {m: slot_offset[slot_key(m)] for m in methods}
 
 
def draw_panel(ax, agg, x_col, value_col, x_values, methods, grid_alpha, minor_grid,
               bar_edge_lw=0.6, grid_lw=1.4, bar_layout="wide",
               group_fill=0.86, family_gap=0.9):
    """Draw one bar panel on the shared x positions `x_values`."""
    x = np.arange(len(x_values))
    if bar_layout == "legacy":
        width, offset_map = family_offsets()
    else:
        width, offset_map = compute_offsets(methods, group_fill, family_gap)
 
    for method in methods:
        vals = []
        for xv in x_values:
            sub = agg[(agg[x_col] == xv) & (agg["method"] == method)]
            vals.append(sub[value_col].iloc[0] if not sub.empty else np.nan)
 
        ax.bar(
            x + offset_map.get(method, 0.0),
            vals,
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=bar_edge_lw,
        )
 
    ax.set_xticks(x)
    # squeeze the side margins so the groups use the full panel
    ax.set_xlim(-0.5 - 0.02, len(x_values) - 0.5 + 0.02)
 
    # Horizontal rules are more marked than in the single-panel figures, so a
    # bar in the right panel can be read against the left panel's y axis.
    ax.grid(axis="y", which="major", color="#4d4d4d",
            alpha=grid_alpha, linewidth=grid_lw, linestyle="-")
    if minor_grid:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="y", which="minor", color="#9a9a9a",
                alpha=grid_alpha * 0.5, linewidth=grid_lw * 0.6, linestyle=(0, (4, 4)))
        ax.tick_params(axis="y", which="minor", left=False)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_axisbelow(True)
 
 
def wrap_title(title, panel_width_in, fontsize):
    """
    Keep a panel title inside its own panel: wrap each line to the number of
    characters that fit at this font size, so the two titles never collide.
    """
    # serif average glyph width ~ 0.48 em
    max_chars = max(18, int((panel_width_in * 72.0) / (0.46 * fontsize)))
    out = []
    for line in str(title).split("\n"):
        out.extend(textwrap.wrap(line, width=max_chars) or [""])
    return "\n".join(out)
 
 
def merged_label(names):
    """'Ring+ZFP online (rate:10)' + '(rate:8)' -> 'Ring+ZFP online (rate:10/8)'."""
    if len(names) == 1:
        return names[0]
    stems = {n.split(" (rate:")[0] for n in names}
    if len(stems) == 1 and all(" (rate:" in n for n in names):
        rates = [n.split(" (rate:")[1].rstrip(")") for n in names]
        return f"{stems.pop()} (rate:{'/'.join(rates)})"
    return " / ".join(names)
 
 
def legend_handles(methods, edge_lw=0.6):
    """One proxy handle per bar slot, in canonical method order."""
    groups = {}
    order = []
    for m in methods:
        k = slot_key(m)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(m)
 
    handles, labels = [], []
    for k in order:
        names = groups[k]
        handles.append(
            Patch(facecolor=METHOD_COLORS.get(names[-1], "#999999"),
                  edgecolor="black", linewidth=edge_lw)
        )
        labels.append(merged_label(names))
    return handles, labels
 
 
def row_major(handles, labels, ncol):
    """
    Matplotlib fills legend columns top-to-bottom. Re-shuffle so the entries
    read left-to-right, one after the other.
    """
    n = len(labels)
    nrows = math.ceil(n / ncol)
    order = []
    for col in range(ncol):
        for row in range(nrows):
            i = row * ncol + col
            if i < n:
                order.append(i)
    return [handles[i] for i in order], [labels[i] for i in order]
 
 
def plot_pair(
    panels,            # list of (agg, title), drawn left -> right
    x_col,
    x_label,
    x_values,
    value_col,
    ylabel,
    methods,
    output,
    ymax=None,
    figsize=(19.5, 6.6),
    wspace=0.10,
    xlabel_mode="each",       # each | once | none
    legend="none",            # none | bottom
    legend_ncol=None,
    legend_fontsize=16,
    grid_alpha=0.45,
    minor_grid=False,
    ytick_step=None,
    nyticks=None,
    lower_better="none",      # none | ylabel | legend | corner
    title_fontsize=24,
    label_fontsize=23,
    tick_fontsize=20,
    scale=1.0,
    bar_edge_lw=0.6,
    bar_layout="wide",
    group_fill=0.86,
    family_gap=0.9,
):
    # One knob to blow everything up for a poster.
    title_fontsize *= scale
    label_fontsize *= scale
    tick_fontsize *= scale
    legend_fontsize *= scale
 
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True,
        layout="constrained",
    )
    fig.get_layout_engine().set(wspace=wspace, w_pad=0.03, h_pad=0.03)
 
    # usable width of ONE panel, used to wrap the titles
    panel_width_in = (figsize[0] * 0.92) / 2.0
 
    for ax, (agg, title) in zip(axes, panels):
        draw_panel(ax, agg, x_col, value_col, x_values, methods, grid_alpha, minor_grid,
                   bar_edge_lw=bar_edge_lw, grid_lw=1.4 * scale,
                   bar_layout=bar_layout, group_fill=group_fill, family_gap=family_gap)
        ax.set_title(
            wrap_title(title, panel_width_in, title_fontsize),
            fontsize=title_fontsize, pad=12,
        )
        ax.set_xticklabels([str(v) for v in x_values], fontsize=tick_fontsize)
        ax.tick_params(axis="x", length=6 * scale, width=1.4 * scale, labelsize=tick_fontsize)
        for sp in ax.spines.values():
            sp.set_linewidth(1.6 * scale)
 
    # Shared y limits (sharey=True -> setting one sets both).
    if ymax is None:
        data_max = 0.0
        for agg, _ in panels:
            v = agg[value_col].max()
            if np.isfinite(v):
                data_max = max(data_max, float(v))
        ymax = data_max * 1.08 if data_max > 0 else None
    if ymax is not None:
        axes[0].set_ylim(0, ymax)
 
    # Bigger tick labels make matplotlib drop ticks; let the user force them back.
    if ytick_step:
        axes[0].yaxis.set_major_locator(MultipleLocator(ytick_step))
    elif nyticks:
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=nyticks, steps=[1, 2, 2.5, 5, 10]))
 
    # One y axis: ticks and label on the left panel only.
    axes[0].tick_params(axis="y", labelsize=tick_fontsize,
                        length=6 * scale, width=1.4 * scale)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.spines["left"].set_visible(False)
 
    y_text = ylabel
    if lower_better == "ylabel":
        y_text = f"{ylabel}\n(lower is better)"
    axes[0].set_ylabel(y_text, fontsize=label_fontsize, labelpad=6)
 
    # x label under each panel, once under the whole figure, or not at all.
    if xlabel_mode == "each":
        for ax in axes:
            ax.set_xlabel(x_label, fontsize=label_fontsize)
    elif xlabel_mode == "once":
        fig.supxlabel(x_label, fontsize=label_fontsize)
 
    if lower_better == "corner":
        axes[0].text(
            0.01, 0.97, r"$\downarrow$ lower is better",
            transform=axes[0].transAxes, ha="left", va="top",
            fontsize=legend_fontsize, style="italic",
        )
 
    # Optional single legend for the whole figure, laid out horizontally.
    if legend == "bottom":
        handles, labels = legend_handles(methods, edge_lw=bar_edge_lw)
        ncol = legend_ncol or (len(labels) if len(labels) <= 5 else math.ceil(len(labels) / 2))
        handles, labels = row_major(handles, labels, ncol)
        lg = fig.legend(
            handles, labels,
            loc="outside lower center",
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
            columnspacing=1.2,
            handlelength=1.6,
            handletextpad=0.6,
            borderaxespad=0.2,
        )
        if lower_better == "legend":
            lg.set_title(r"$\downarrow$ lower is better")
            lg.get_title().set_fontsize(legend_fontsize)
            lg.get_title().set_style("italic")
 
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")
 
 
def plot_legend_only(
    methods,
    output,
    width=19.5,
    legend_ncol=None,
    legend_fontsize=16,
    scale=1.0,
    bar_edge_lw=1.0,
    lower_better="legend",
):
    """
    Render ONLY the legend, as its own file, so it can be placed once on the
    poster next to two (or more) legend-less figures. Entries are laid out
    horizontally, one after the other, in METHOD_ORDER.
    """
    legend_fontsize *= scale
 
    handles, labels = legend_handles(methods, edge_lw=bar_edge_lw)
    # default: two rows, so the strip is not absurdly wide
    ncol = legend_ncol or (len(labels) if len(labels) <= 5 else math.ceil(len(labels) / 2))
    nrows = math.ceil(len(labels) / ncol)
 
    fig = plt.figure(figsize=(width, 0.45 * nrows + 0.3))
    handles, labels = row_major(handles, labels, ncol)
 
    lg = fig.legend(
        handles, labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        columnspacing=1.4,
        handlelength=1.6,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    if lower_better != "none":
        lg.set_title(r"$\downarrow$ lower is better")
        lg.get_title().set_fontsize(legend_fontsize)
        lg.get_title().set_style("italic")
 
    fig.savefig(output, bbox_inches="tight", bbox_extra_artists=[lg])
    plt.close(fig)
    print(f"[INFO] wrote {output}")
 
 
# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
 
def main():
    ap = argparse.ArgumentParser(
        description="Two experiment folders side by side in one figure: shared axes, no duplicated legend."
    )
    ap.add_argument("root_left", nargs="?", default=None,
                    help="Experiment folder drawn in the LEFT panel (not needed with --legend-only)")
    ap.add_argument("root_right", nargs="?", default=None,
                    help="Experiment folder drawn in the RIGHT panel (not needed with --legend-only)")
 
    ap.add_argument("--mode", choices=["strong", "weak"], default=None)
    ap.add_argument("--scope", choices=["fixed", "all"], default="all")
    ap.add_argument("--metric", choices=["time", "hook", "tail"], default=None)
 
    ap.add_argument("--gpus", type=int, default=None, help="Required for --scope fixed")
    ap.add_argument("--global-batch", type=int, default=None, help="Strong-scaling filter")
    ap.add_argument("--batch-per-rank", type=int, default=None, help="Weak-scaling filter")
 
    ap.add_argument("--title-left", default=None, help="Override left panel title")
    ap.add_argument("--title-right", default=None, help="Override right panel title")
    ap.add_argument("--backend", default=None, help="Override backend label in titles (NCCL, RCCL, MPI)")
 
    ap.add_argument("--ymax", type=float, default=None,
                    help="Fixed shared upper y limit (default: max over both panels + 8%%)")
    ap.add_argument("--width", type=float, default=19.5, help="Total figure width in inches")
    ap.add_argument("--height", type=float, default=6.6, help="Figure height in inches")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiply ALL font sizes / line widths (1.2 = 20%% bigger)")
    ap.add_argument("--title-fontsize", type=float, default=24)
    ap.add_argument("--label-fontsize", type=float, default=23)
    ap.add_argument("--tick-fontsize", type=float, default=20)
    ap.add_argument("--bar-edge-lw", type=float, default=1.0, help="Bar outline width")
    ap.add_argument("--bar-layout", choices=["wide", "legacy"], default="wide",
                    help="'wide' fattens the bars to fill the slot; 'legacy' keeps plot_metric's geometry")
    ap.add_argument("--group-fill", type=float, default=0.86,
                    help="Fraction of the x slot one GPU-count group occupies (wide layout)")
    ap.add_argument("--family-gap", type=float, default=0.9,
                    help="Gap between Baseline / Ring / RD families, in bar widths (wide layout)")
    ap.add_argument("--wspace", type=float, default=0.10, help="Horizontal gap between the two panels")
 
    ap.add_argument("--xlabel", choices=["each", "once", "none"], default="each",
                    help="Draw the x label under each panel, once for the figure, or not at all")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated method labels for --legend-only without folders "
                         "(default: every method plot_metric knows)")
    ap.add_argument("--legend-only", action="store_true",
                    help="Write ONLY the legend (horizontal, with the 'lower is better' note) and stop")
    ap.add_argument("--legend", choices=["none", "bottom"], default="none",
                    help="No legend (default) or one shared horizontal legend under both panels")
    ap.add_argument("--legend-ncol", type=int, default=None)
    ap.add_argument("--legend-fontsize", type=float, default=16)
    ap.add_argument("--lower-better", choices=["none", "ylabel", "legend", "corner"], default="none",
                    help="Where to print the 'lower is better' note, if anywhere")
 
    ap.add_argument("--grid-alpha", type=float, default=0.45, help="Strength of the horizontal rules")
    ap.add_argument("--ytick-step", type=float, default=None,
                    help="Force a y tick / horizontal rule every N units (e.g. 25)")
    ap.add_argument("--nyticks", type=int, default=None,
                    help="Approximate number of y ticks, if --ytick-step is not used")
    ap.add_argument("--minor-grid", action="store_true",
                    help="Add lighter dashed rules between the major ones")
 
    ap.add_argument("--png", action="store_true", help="Also save PNG besides PDF")
    ap.add_argument("--csv", action="store_true", help="Save the two filtered dataframes")
    ap.add_argument("--out", default=None)
 
    args = ap.parse_args()
 
    apply_style()
 
    # ---- legend-only: no experiment folders needed --------------------------
    if args.legend_only:
        if args.root_left and args.root_right:
            # use exactly the methods present in the two runs
            if args.mode is None or args.metric is None:
                raise SystemExit("--mode and --metric are required when passing experiment folders")
            value_col, _ = metric_spec(args.mode, args.metric)
            df_l, _ = load_panel(Path(args.root_left), args, value_col)
            df_r, _ = load_panel(Path(args.root_right), args, value_col)
            methods = ordered_methods(set(df_l["method"]) | set(df_r["method"]))
            outdir = ensure_plot_dir(Path(args.root_left))
        elif args.methods:
            methods = ordered_methods([m.strip() for m in args.methods.split(",") if m.strip()])
            outdir = Path(".")
        else:
            # no folders given: every method plot_metric knows about
            methods = list(METHOD_ORDER)
            outdir = Path(".")
 
        out_pdf = Path(args.out) if args.out else outdir / "pair_legend.pdf"
        legend_kwargs = dict(
            width=args.width,
            legend_ncol=args.legend_ncol,
            legend_fontsize=args.legend_fontsize,
            scale=args.scale,
            bar_edge_lw=args.bar_edge_lw,
            lower_better="legend" if args.lower_better == "none" else args.lower_better,
        )
        plot_legend_only(methods, output=out_pdf, **legend_kwargs)
        if args.png:
            plot_legend_only(methods, output=out_pdf.with_suffix(".png"), **legend_kwargs)
        return
 
    # ---- normal two-panel figure -------------------------------------------
    if not (args.root_left and args.root_right):
        raise SystemExit("two experiment folders are required (left panel, right panel)")
    if args.mode is None or args.metric is None:
        raise SystemExit("--mode and --metric are required")
    if args.scope == "fixed" and args.gpus is None:
        raise SystemExit("--gpus is required when --scope fixed")
 
    root_left = Path(args.root_left)
    root_right = Path(args.root_right)
 
    value_col, ylabel = metric_spec(args.mode, args.metric)
 
    df_l, auto_title_l = load_panel(root_left, args, value_col)
    df_r, auto_title_r = load_panel(root_right, args, value_col)
 
    title_l = args.title_left or auto_title_l
    title_r = args.title_right or auto_title_r
 
    if args.scope == "all":
        x_col, x_label = "ranks", "GPUs"
    else:
        x_col = "global_batch" if args.mode == "strong" else "batch_per_rank"
        x_label = "Global batch" if args.mode == "strong" else "Batch per rank"
 
    agg_l = aggregate(df_l, x_col, value_col)
    agg_r = aggregate(df_r, x_col, value_col)
 
    # shared x positions = union of both panels, so both use identical ticks
    x_values = sorted(set(agg_l[x_col].dropna()) | set(agg_r[x_col].dropna()))
    # shared method set, in canonical order
    methods = ordered_methods(set(agg_l["method"]) | set(agg_r["method"]))
 
    if args.out:
        out_pdf = Path(args.out)
    else:
        outdir = ensure_plot_dir(root_left)
        stem = f"pair_{args.mode}_{args.scope}"
        if args.scope == "fixed":
            stem += f"_{args.gpus}gpu"
        out_pdf = outdir / f"{stem}_{args.metric}.pdf"
 
    kwargs = dict(
        x_col=x_col,
        x_label=x_label,
        x_values=x_values,
        value_col=value_col,
        ylabel=ylabel,
        methods=methods,
        ymax=args.ymax,
        figsize=(args.width, args.height),
        wspace=args.wspace,
        xlabel_mode=args.xlabel,
        legend=args.legend,
        legend_ncol=args.legend_ncol,
        legend_fontsize=args.legend_fontsize,
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize,
        scale=args.scale,
        bar_edge_lw=args.bar_edge_lw,
        bar_layout=args.bar_layout,
        group_fill=args.group_fill,
        family_gap=args.family_gap,
        grid_alpha=args.grid_alpha,
        minor_grid=args.minor_grid,
        ytick_step=args.ytick_step,
        nyticks=args.nyticks,
        lower_better=args.lower_better,
    )
    panels = [(agg_l, title_l), (agg_r, title_r)]
 
    plot_pair(panels, output=out_pdf, **kwargs)
 
    if args.png:
        plot_pair(panels, output=out_pdf.with_suffix(".png"), **kwargs)
 
    if args.csv:
        df_l.to_csv(out_pdf.with_suffix(".left.csv"), index=False)
        df_r.to_csv(out_pdf.with_suffix(".right.csv"), index=False)
        print(f"[INFO] wrote {out_pdf.with_suffix('.left.csv')} and {out_pdf.with_suffix('.right.csv')}")
 
 
if __name__ == "__main__":
    main()
 