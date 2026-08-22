"""Runner for coarse-fine DTC simulations.

Edit CONFIG to tune coarse/fine parameters and choose CS/VS/DL per block.
"""

from pathlib import Path
import sys
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np

from plot_style import apply_science_style

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_style import apply_science_style, _multi_panel_figsize, maybe_title, maybe_suptitle
from coarse_fine.coarse_fine_core import (
    build_coarse_fine_dtc,
    optimize_split_loop,
    run_mc_mismatch_analysis,
)


apply_science_style()


MODE_ABBREVIATIONS = {
    "constant": "CS",
    "variable": "VS",
    "delay_line": "DL",
}


CONFIG = {
    "save_dir": Path(
        r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\python simulation\coarse_fine_python"
    ),
    "max_delay_ns": 8.0,
    "mc_runs": 100,
    "dnl_limit_lsb": 0.5,
    # Voltage scaling applied directly to both Vdd and Vth:
    # - constant slope uses voltage_scale_factor_constant
    # - variable slope and delay-line use voltage_scale_factor_variable_delay_line
    "voltage_scale_factor_constant": 1,
    "voltage_scale_factor_variable_delay_line": 1,
    "split_n_total": 13,
    "split_n_coarse_values": np.arange(2, 11),
    "include_delay_line_modes": True,
    # Vdd sweep used for final optimization summary plots.
    # For each Vdd and each mode configuration, we keep the best split
    # (same criterion as the printed table) and plot P_max and P_avg.
    "optimization_vdd_values": [1.1, 0.88, 0.55],
    "plot_from_csv_only": True,  # Set to True to skip the optimization and just plot from the saved CSV.
    "optimization": False,
    "coarse_values": {
        "n": 8,
        "Cu": 1e-15,
        "Vdd": 1.1,
        "Vth":0.55,
        "f": 50e6,
        "Ich" : 92e-6,
        "Cramp": 284e-15,
        "Cramp_dl": 2.228e-15,  # Delay-line ramp capacitance (defaults to Cramp if omitted)
        "C_ramp_cu":3.280e-15,  # VS CDAC unit capacitance (defaults to Cramp if omitted)
        "self_power_down_cs": "yes",  # VS extra power reduction: "yes" or "no"
        "self_power_down_vs": "yes",  # VS extra power reduction: "yes" or "no"
        "self_power_down_dl": "no",  # DL extra power reduction: "yes" or "no"
        "Ac": 5e-3,
        "A": 3.280,
        "C0": 30.9e-15,
        "dac_mode": "binary", #binary or thermometer or segmented
        "slope_mode": "variable",  # "constant" (CS), "variable" (VS), or "delay_line" (DL)
        "delay_line_selection_mode": "tapped",  # "tapped" (constant power) or "accumulated" (rising power)
        "C_fixed": 100e-15
    },
    "fine_values": {
        "n": 5,
        "Cu": 1e-15,
        "Vdd": 1.1,
        "Vth":0.55,
        "f": 50e6,
        "Ich": 50.49e-6,
        "Cramp": 1e-15,
        "Cramp_dl": 1e-15,  # Delay-line ramp capacitance (defaults to Cramp if omitted)
        "C_ramp_cu": 1e-15,  # VS CDAC unit capacitance (defaults to Cramp if omitted)
        "self_power_down_cs": "yes",  # VS extra power reduction: "yes" or "no"
        "self_power_down_vs": "yes",  # VS extra power reduction: "yes" or "no"
        "self_power_down_dl": "no",  # DL extra power reduction: "yes" or "no"
        "Ac": 5e-3,
        "A": 1,
        "C0":14e-15,
        "C0_scale": 1,
        "dac_mode": "binary",
        "slope_mode": "constant",  # "constant" (CS), "variable" (VS), or "delay_line" (DL)
        "delay_line_selection_mode": "tapped",  # "tapped" (constant power) or "accumulated" (rising power)
        "C_fixed": 10e-15
    },
}


def _plot_optimization_metric_vs_vdd(
    records: list,
    mode_configs: list,
    vdd_values: list,
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: str,
    exclude_delay_lines: bool = False,
) -> None:
    """Plot one optimization metric versus Vdd for all mode configurations with shared custom legend."""
    
    # 1. Initialize the single plot canvas using your standard aspect ratios
    base_w, base_h = plt.rcParams.get("figure.figsize", (3.3, 2.5))
    fig, ax = plt.subplots(figsize=(base_w * 1.5, base_h * 1.5))

    # 2. Extract configuration metadata and set uniform categorical color maps
    mode_labels = [cfg_label for (cfg_label, _, _) in mode_configs]
    if exclude_delay_lines:
        mode_labels = [l for l in mode_labels if "DL" not in str(l).upper()]

    palette = [plt.cm.tab10(i) for i in range(10)]
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(mode_labels)}

    # 3. Filter and parse valid plot lines sequentially
    plot_subset = []
    for cfg_label in mode_labels:
        cfg_rows = [
            r for r in records
            if r["configuration"] == cfg_label and float(r["vdd"]) in [float(v) for v in vdd_values]
        ]
        cfg_rows = sorted(cfg_rows, key=lambda r: float(r["vdd"]))
        if len(cfg_rows) == 0:
            continue

        # Append rows to our cumulative list to hand off for cluster grouping annotations later
        plot_subset.extend(cfg_rows)

        x_vdd = [float(r["vdd"]) for r in cfg_rows]
        y_metric = [float(r[metric_key]) * 1e6 for r in cfg_rows]

        ax.plot(
            x_vdd,
            y_metric,
            marker='o',
            markersize=7,
            linewidth=2,
            color=color_map[cfg_label],
            label=cfg_label,
        )

    # 4. FIX: Invoke vertical column clustering outside the loop passing the full data collection
    if plot_subset:
        _annotate_bits(ax, plot_subset, metric_key, color_map)

    # 5. Core plot typography, styling frames, and mathematical axis scaling
    maybe_title(ax, title, fontweight='bold', pad=12)
    ax.set_xlabel(r'$\mathrm{V_\mathrm{DD}}$ $\mathrm{[V]}$', fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)

    # 6. Extract legend handles dynamically and format the identical frame structure 
    handles, labels = ax.get_legend_handles_labels()
    
    # Compress internal margins to create a dedicated empty space on the right for the floating box
    fig.subplots_adjust(left=0.12, right=0.75, top=0.88, bottom=0.16)
    
    fig.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(0.78, 0.5), # Anchor positioned cleanly outside the right frame boundary
        framealpha=0.96,
        edgecolor='black',
        title=rf"$\mathrm{{Mode}}$",
    )

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

def _annotate_bits(ax, records: list, metric_key: str, color_map: dict) -> None:
    """
    Groups configurations by VDD and sorts them dynamically by y-value.
    
    If analyzing the delay line panel (detected via its size/subset characteristics),
    it spreads annotations tightly above and below the points using a 2-element split.
    Otherwise, it applies the compact 5-point centered stack layout.
    """
    from collections import defaultdict
    
    # 1. Group ALL records by their precise VDD voltage column
    vdd_clusters = defaultdict(list)
    for record in records:
        x_vdd = round(float(record["vdd"]), 3)
        vdd_clusters[x_vdd].append(record)
        
    # 2. Process each vertical VDD column independently
    for x_vdd, cluster in vdd_clusters.items():
        
        # 3. Sort the entire vertical column cluster by its plotted Y-value
        sorted_cluster = sorted(cluster, key=lambda r: float(r[metric_key]))
        cluster_size = len(sorted_cluster)
        
        # 4. Apply conditional stacking geometry based on data density
        for stack_index, record in enumerate(sorted_cluster):
            y_metric = float(record[metric_key]) * 1e6
            bit_text = f"{int(record['n_coarse'])}+{int(record['n_fine'])}"
            
            # --- CUSTOM GEOMETRY FOR DELAY LINES (Usually 2 elements in the subset) ---
            if stack_index >= 4:
                # stack_index = 0 -> -1 -> -11 (tightly below the marker)
                # stack_index = 1 ->  1 ->  11 (tightly above the marker)
                centered_index = -1 if stack_index == 5 else 1
                box_spacing = 11
                y_offset = centered_index * box_spacing
                
            # --- STANDARD CENTERING MATRIX FOR MULTI-CURVE CLUSTERS (5 elements) ---
            else:
                # Maps index directly to a balanced centered tower array [-22, -11, 0, 11, 22]
                centered_index = stack_index - 2
                box_spacing = 11  # Kept small to bring bounding boxes close together
                y_offset = centered_index * box_spacing
            
            ax.annotate(
                bit_text,
                xy=(float(record["vdd"]), y_metric),
                textcoords="offset points",
                xytext=(0, y_offset),  # Perfect vertical line projection alignment
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color=color_map.get(record["configuration"], 'black'),
                bbox=dict(
                    boxstyle='round,pad=0.15', # Tight interior text box padding
                    facecolor='white', 
                    edgecolor='black', 
                    alpha=0.75,                 # Solid backdrop prevents grid overlap
                    lw=0.8
                ),
                arrowprops=dict(
                    arrowstyle="->", 
                    color="gray", 
                    lw=0.6, 
                    alpha=0.4,
                    shrinkA=1
                )
            )


def _plot_shared_legend_three_panel(
    records: list,
    title: str,
    save_path: str,
    include_delay_lines: bool = True,
) -> None:
    """Create a 3-panel figure with one shared legend on the right."""
    title_size = plt.rcParams.get("axes.titlesize", 9)
    if not records:
        print(f"No valid rows found for {save_path}")
        return

    if include_delay_lines:
        plot_records = list(records)
        title_suffix = ""
    else:
        plot_records = [r for r in records if not _has_delay_line_mode(r)]
        title_suffix = " (No Delay Lines)"

    if not plot_records:
        print(f"No records available for {save_path}")
        return

    mode_labels = sorted({r["configuration"] for r in plot_records if r.get("configuration")})
    palette = [
       plt.cm.tab10(i) for i in range(10)
    ]
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(mode_labels)}
    vdd_values = sorted({float(r["vdd"]) for r in plot_records})

    base_w, base_h = plt.rcParams.get("figure.figsize", (3.3, 2.5))
    fig = plt.figure(figsize=(base_w * 3.7, base_h * 1.9))
    gs = fig.add_gridspec(1, 3, wspace=0.4)
    ax_pmax_dl = fig.add_subplot(gs[0, 0])
    ax_pmax_no = fig.add_subplot(gs[0, 1])
    ax_pavg_no = fig.add_subplot(gs[0, 2])

    handles = []
    labels = []
    for mode_label in mode_labels:
        handle = plt.Line2D([0], [0], color=color_map[mode_label], marker='o', linewidth=2.4, markersize=7)
        handles.append(handle)
        labels.append(mode_label)

    def plot_panel(ax, metric_key: str, ylabel: str, panel_title: str, record_subset: list, annotate: bool = True) -> None:
        # 1. Plot the lines for each configuration independently
        for mode_label in mode_labels:
            cfg_rows = [r for r in record_subset if r["configuration"] == mode_label]
            cfg_rows = sorted(cfg_rows, key=lambda r: float(r["vdd"]))
            if not cfg_rows:
                continue
            x_vdd = [float(r["vdd"]) for r in cfg_rows]
            y_metric = [float(r[metric_key]) * 1e6 for r in cfg_rows]
            ax.plot(
                x_vdd,
                y_metric,
                marker='o',
                markersize=7,
                linewidth=2,
                color=color_map[mode_label],
                label=rf"$\mathrm{{{mode_label}}}$",
            )

        # 2. CRITICAL FIX: Move the annotation block OUTSIDE the loop!
        # This passes the cumulative subset so Biber/Matplotlib groups them correctly.
        if annotate and record_subset:
            _annotate_bits(ax, record_subset, metric_key, color_map)

        maybe_title(ax, panel_title, fontweight='bold', pad=12)
        ax.set_xlabel(r'$\mathrm{V_\mathrm{DD}}$ $\mathrm{[V]}$', fontsize=20, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
        ax.set_axisbelow(True)

    plot_panel(
        ax_pmax_dl,
        metric_key="max_total_power_w",
        ylabel=r"$P_\mathrm{max}$ [$\mu W$]",
        panel_title=f"P_\mathrm{{max}} with delay lines{title_suffix}",
        record_subset=plot_records,
        annotate=True,
    )

    no_dl_records = [r for r in plot_records if not _has_delay_line_mode(r)]
    plot_panel(
        ax_pmax_no,
        metric_key="max_total_power_w",
        ylabel=r"$P_\mathrm{max}$ [$\mu W$]",
        panel_title=f"P_\mathrm{{max}} without delay lines{title_suffix}",
        record_subset=no_dl_records,
        annotate=True,
    )
    plot_panel(
        ax_pavg_no,
        metric_key="avg_total_power_w",
        ylabel=r"$P_{\mathrm{avg}}$ [$\mu W$]",
        panel_title=f"P_\mathrm{{avg}} without delay lines{title_suffix}",
        record_subset=no_dl_records,
        annotate=True,
    )

    fig.subplots_adjust(left=0.07, right=0.80, top=0.88, bottom=0.16, wspace=0.34)
    fig.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(0.82, 0.5),
        framealpha=0.96,
        edgecolor='black',
        title=rf"$\mathrm{{Mode}}$",
    )
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def _load_optimization_rows_from_csv(csv_path: Path) -> list:
    """Load optimization results from CSV into records expected by plotter."""
    records = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vdd = float(row.get("vdd", ""))
                n_coarse = int(float(row.get("n_coarse", "")))
                n_fine = int(float(row.get("n_fine", "")))
                p_avg_uW = float(row.get("p_avg_uW", ""))
                p_max_uW = float(row.get("p_max_uW", ""))
            except (TypeError, ValueError):
                continue

            records.append({
                "vdd": vdd,
                "configuration": row.get("configuration", "").strip(),
                "coarse_mode": row.get("coarse_mode", "").strip(),
                "fine_mode": row.get("fine_mode", "").strip(),
                "n_coarse": n_coarse,
                "n_fine": n_fine,
                "avg_total_power_w": p_avg_uW * 1e-6,
                "max_total_power_w": p_max_uW * 1e-6,
            })
    return records


def _has_delay_line_mode(record: dict) -> bool:
    """Return True when either block uses delay-line mode."""
    coarse_mode = str(record.get("coarse_mode", "")).strip().lower()
    fine_mode = str(record.get("fine_mode", "")).strip().lower()
    config_label = str(record.get("configuration", "")).upper()
    return coarse_mode == "delay_line" or fine_mode == "delay_line" or "DL" in config_label


def _mode_label(record: dict) -> str:
    """Build a compact label like CS-VS from coarse/fine mode fields."""
    coarse_mode = str(record.get("coarse_mode", "")).strip().lower()
    fine_mode = str(record.get("fine_mode", "")).strip().lower()
    coarse_abbr = MODE_ABBREVIATIONS.get(coarse_mode, str(record.get("coarse_mode", "")).strip().upper()[:2])
    fine_abbr = MODE_ABBREVIATIONS.get(fine_mode, str(record.get("fine_mode", "")).strip().upper()[:2])
    return f"{coarse_abbr}-{fine_abbr}"


def _bits_text(record: dict) -> str:
    """Return the bit split as a short annotation, e.g. 8+5."""
    return f"{int(record['n_coarse'])}+{int(record['n_fine'])}"


def _plot_annotated_metric(
    ax,
    records: list,
    metric_key: str,
    title: str,
    ylabel: str,
    color_map: dict,
    show_legend: bool = False,
    legend_handles=None,
    legend_labels=None,
) -> None:
    """Plot one CSV-derived metric with per-point bit annotations."""
    for record in records:
        label = _mode_label(record)
        color = color_map.get(label, "#1F77B4")
        x_val = float(record["vdd"])
        y_val = float(record[metric_key]) * 1e6
        ax.plot(
            [x_val],
            [y_val],
            marker='o',
            markersize=7,
            linewidth=0,
            color=color,
        )
        ax.annotate(
            _bits_text(record),
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 7),
            ha='center',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75),
        )

    maybe_title(ax, title, fontweight='bold', pad=12)
    ax.set_xlabel(r'$\mathrm{V_\mathrm{DD}}$ $\mathrm{[V]}$', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    if show_legend and legend_handles is not None and legend_labels is not None:
        ax.legend(legend_handles, legend_labels, fontsize=9, framealpha=0.96, edgecolor='black', loc='upper left')


def plot_optimization_metrics_from_csv(csv_path: Path, save_dir: Path) -> None:
    """Plot Pmax/Pavg vs Vdd using the saved optimization CSV."""
    records = _load_optimization_rows_from_csv(csv_path)
    if not records:
        print(f"No valid rows found in {csv_path}")
        return

    mode_labels = sorted({r["configuration"] for r in records if r.get("configuration")})
    mode_configs = [(label, None, None) for label in mode_labels]
    vdd_values = sorted({float(r["vdd"]) for r in records})

    pmax_plot_path = save_dir / "coarse_fine_opt_pmax_vs_vdd.csv_only.pdf"
    _plot_optimization_metric_vs_vdd(
        records=records,
        mode_configs=mode_configs,
        vdd_values=vdd_values,
        metric_key="max_total_power_w",
        ylabel=rf" $P_\mathrm{{max}}$ $[\mathrm{{\mu W}}]$",
        title=rf"Best Split P_\mathrm{{max}}$ vs Vdd by Configuration",
        save_path=str(pmax_plot_path),
    )

    records_no_dl = [r for r in records if not _has_delay_line_mode(r)]
    if records_no_dl:
        pmax_plot_path_no_dl = save_dir / "coarse_fine_opt_pmax_vs_vdd.csv_only_no_dl.pdf"
        _plot_optimization_metric_vs_vdd(
            records=records_no_dl,
            mode_configs=[(label, None, None) for label in sorted({r["configuration"] for r in records_no_dl if r.get("configuration")})],
            vdd_values=sorted({float(r["vdd"]) for r in records_no_dl}),
            metric_key="max_total_power_w",
            ylabel=r"$P_\mathrm{max}$ $[\mathrm{\mu W}]$",
            title="Best Split P_max vs Vdd by Configuration (No Delay Lines)",
            save_path=str(pmax_plot_path_no_dl),
            exclude_delay_lines=True,
        )

    pavg_plot_path = save_dir / "coarse_fine_opt_pavg_vs_vdd.csv_only.pdf"
    _plot_optimization_metric_vs_vdd(
        records=records,
        mode_configs=mode_configs,
        vdd_values=vdd_values,
        metric_key="avg_total_power_w",
        ylabel=r" $P_\mathrm{avg}$ $[\mathrm{\mu W}]$",
        title="Best Split P_avg vs Vdd by Configuration",
        save_path=str(pavg_plot_path),
    )

    if records_no_dl:
        pavg_plot_path_no_dl = save_dir / "coarse_fine_opt_pavg_vs_vdd.csv_only_no_dl.pdf"
        _plot_optimization_metric_vs_vdd(
            records=records_no_dl,
            mode_configs=[(label, None, None) for label in sorted({r["configuration"] for r in records_no_dl if r.get("configuration")})],
            vdd_values=sorted({float(r["vdd"]) for r in records_no_dl}),
            metric_key="avg_total_power_w",
            ylabel=r" $P_\mathrm{avg}$ $[\mathrm{\mu W}]$",
            title="Best Split P_avg vs Vdd by Configuration (No Delay Lines)",
            save_path=str(pavg_plot_path_no_dl),
            exclude_delay_lines=True,
        )

    # Combined 3-panel figure with one shared legend on the left.
    records_no_dl_by_mode = [r for r in records if not _has_delay_line_mode(r)]
    if records and records_no_dl_by_mode:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True)

        unique_labels = sorted({_mode_label(r) for r in records})
        color_cycle = [plt.cm.tab10(i) for i in range(plt.cm.tab10.N)]
        color_map = {label: color_cycle[i % len(color_cycle)] for i, label in enumerate(unique_labels)}

        # Build a single legend from the label/color map.
        legend_handles = []
        legend_labels = []
        for label in unique_labels:
            handle = plt.Line2D([0], [0], color=color_map[label], marker='o', linewidth=2.0, markersize=7)
            legend_handles.append(handle)
            legend_labels.append(label)

        # _plot_annotated_metric(
        #     axes[0],
        #     records,
        #     metric_key="max_total_power_w",
        #     title="P_max with delay lines",
        #     ylabel=r"$P_\mathrm{max}$ $[\mathrm{\mu W}]$",
        #     color_map=color_map,
        # )
        # _plot_annotated_metric(
        #     axes[1],
        #     records_no_dl_by_mode,
        #     metric_key="max_total_power_w",
        #     title=r"P_\mathrm{max} without delay lines",
        #     ylabel=r"$P_\mathrm{max}$ $[\mathrm{\mu W}]$",
        #     color_map=color_map,
        # )
        # _plot_annotated_metric(
        #     axes[2],
        #     records_no_dl_by_mode,
        #     metric_key="avg_total_power_w",
        #     title=r"P_\mathrm{avg} without delay lines",
        #     ylabel=r"$P_\mathrm{avg}$ $[\mathrm{\mu W}]$",
        #     color_map=color_map,
        # )

        for ax in axes:
            ax.set_xlim(min(vdd_values) - 0.03, max(vdd_values) + 0.03)

        fig.subplots_adjust(left=0.20, right=0.98, wspace=0.28)
        fig.legend(
            legend_handles,
            legend_labels,
            loc='center left',
            bbox_to_anchor=(0.02, 0.5),
            framealpha=0.96,
            edgecolor='black',
            title='Mode',
        )
        combined_path = save_dir / "coarse_fine_opt_combined_three_panel.csv_only.pdf"
        plt.tight_layout(rect=(0.14, 0.0, 1.0, 1.0))
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {combined_path}")
        plt.close(fig)

    combined_three_panel = save_dir / "coarse_fine_opt_three_panel_csv_only.pdf"
    _plot_shared_legend_three_panel(
        records=records,
        title="Best Split Power Summary vs Vdd",
        save_path=str(combined_three_panel),
        include_delay_lines=True,
    )

    combined_three_panel_no_dl = save_dir / "coarse_fine_opt_three_panel_csv_only_no_dl.pdf"
    _plot_shared_legend_three_panel(
        records=records_no_dl if records_no_dl else records,
        title="Best Split Power Summary vs Vdd",
        save_path=str(combined_three_panel_no_dl),
        include_delay_lines=False,
    )


def main() -> None:
    save_dir = CONFIG["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    def out_path(name: str) -> str:
        return str(save_dir / name)
    
    def mode_param_text(mode: str, row: dict, prefix: str) -> str:
        mode_key = str(mode).strip().lower()
        ich_ua = float(row.get(f"ich_{prefix}_a", np.nan)) * 1e6
        if mode_key == "variable":
            cramp_cu_ff = float(row.get(f"cramp_cu_{prefix}_f", np.nan)) * 1e15
            return f"Cramp_cu={cramp_cu_ff:.3f} fF, Ich={ich_ua:.3f} uA"
        if mode_key == "delay_line":
            cramp_dl_ff = float(row.get(f"cramp_dl_{prefix}_f", np.nan)) * 1e15
            return f"Cramp_dl={cramp_dl_ff:.3f} fF, Ich={ich_ua:.3f} uA"

        cramp_ff = float(row.get(f"cramp_{prefix}_f", np.nan)) * 1e15
        c0_ff = float(row.get(f"c0_{prefix}_f", np.nan)) * 1e15
        return f"Cramp={cramp_ff:.3f} fF, C0={c0_ff:.3f} fF, Ich={ich_ua:.3f} uA"


    coarse_values = dict(CONFIG["coarse_values"])
    fine_values = dict(CONFIG["fine_values"])


    architecture = build_coarse_fine_dtc(coarse_values, fine_values)

    #Try to produce a certain taget delay
    target_ns = 5.0

    coarse_idx, fine_idx_per_coarse, policy_meta = architecture._fine_indices_by_transition_policy()
    print(f"Coarse indices: {type(coarse_idx)}")
    print(f"Fine indices per coarse: {type(fine_idx_per_coarse)}")
    result = architecture.synthesize_delay(target_ns * 1e-9, coarse_idx=coarse_idx, fine_idx_per_coarse=fine_idx_per_coarse)

    print("Coarse-Fine DTC result")
    print("-" * 50)
    print(f"Target delay: {result['target_delay_s'] * 1e9:.3f} ns")
    print(f"Total delay : {result['total_delay_s'] * 1e9:.3f} ns")
    print(f"Error       : {result['error_s'] * 1e12:.2f} ps")
    print(f"Coarse code : {int(result['coarse_code'])}")
    print(f"Fine code   : {int(result['fine_code'])}")
    print(f"Total power : {result['p_total_w'] * 1e6:.3f} uW")

    max_delay_ns = float(CONFIG["max_delay_ns"])
    ch = architecture.combined_characteristic(coarse_period_s=max_delay_ns*1e-9, coarse_codes=coarse_idx, fine_idx_per_coarse=fine_idx_per_coarse, policy_meta=policy_meta)
    
    architecture.plot_characteristic_vs_code(t_range_ns=max_delay_ns, save_path=out_path('coarse_fine_delay_vs_code.pdf'), ch=ch, policy_meta=policy_meta)
    architecture.plot_power_vs_code(avg_over_target_range_ns=max_delay_ns, avg_num_points=200, save_path=out_path('coarse_fine_power_vs_code.pdf'), ch =ch)

    architecture.plot_single_block_characteristic('coarse', save_path=out_path('coarse_only_delay_vs_code.pdf'))
    architecture.plot_single_block_nonlinearity('coarse', save_path=out_path('coarse_only_dnl_inl.pdf'))
    architecture.plot_coarse_nonlinearity_ps(mc_runs=int(CONFIG["mc_runs"]), save_path=out_path('coarse_only_dnl_inl_ps.pdf'))
    architecture.plot_single_block_characteristic('fine', save_path=out_path('fine_only_delay_vs_code.pdf'))
    architecture.plot_single_block_nonlinearity('fine', save_path=out_path('fine_only_dnl_inl.pdf'))


    mc_stats = run_mc_mismatch_analysis(
        coarse_values=coarse_values,
        fine_values=fine_values,
        mc_runs=int(CONFIG["mc_runs"]),
        dnl_limit_lsb=float(CONFIG["dnl_limit_lsb"]),
        t_range_ns=max_delay_ns,
        save_path=out_path('coarse_fine_mc_mismatch.pdf'),
    )
    # print(f"Probability of staying below {CONFIG['dnl_limit_lsb']:.1f} LSB = {mc_stats['pass_probability_percent']:.2f}%")

    optimization = CONFIG.get("optimization", False)   # Set to False to skip the optimization loop and just run the main characterization.
    
    if optimization:
        print("\n" + "=" * 70)
        print("Split Optimization Loop - Mode Configurations")
        print("=" * 70)

        mode_configs = [
            ("VS-CS", "variable", "constant"),
            ("VS-VS", "variable", "variable"),
            ("CS-CS", "constant", "constant"),
            ("CS-VS", "constant", "variable"),
        ]
        if bool(CONFIG.get("include_delay_line_modes", False)):
            mode_configs.extend([
                # ("CS-DL", "constant", "delay_line"),
                ("DL-CS", "delay_line", "constant"),
                # ("VS-DL", "variable", "delay_line"),
                ("DL-VS", "delay_line", "variable"),
                # ("DL-DL", "delay_line", "delay_line"),
            ])

        best_rows = []
        optimization_rows = []
        optimization_vdd_values = [float(v) for v in CONFIG.get("optimization_vdd_values", [1.1, 0.8, 0.55])]

        for vdd_target in optimization_vdd_values:
            print("\n" + "=" * 70)
            print(f"Optimization at Vdd = {vdd_target:.3f} V")
            print("=" * 70)

            for cfg_label, coarse_mode, fine_mode in mode_configs:
                cfg_coarse = dict(CONFIG["coarse_values"])
                cfg_fine = dict(CONFIG["fine_values"])
                cfg_coarse["slope_mode"] = coarse_mode
                cfg_fine["slope_mode"] = fine_mode

                cfg_coarse["Vdd"] = float(vdd_target)
                cfg_fine["Vdd"] = float(vdd_target)
                if cfg_coarse.get("Vth") is not None:
                    cfg_coarse["Vth"] = float(vdd_target) / 2.0
                if cfg_fine.get("Vth") is not None:
                    cfg_fine["Vth"] = float(vdd_target) / 2.0


                split_results, best_split = optimize_split_loop(
                    n_total=int(CONFIG["split_n_total"]),
                    base_coarse_values=cfg_coarse,
                    base_fine_values=cfg_fine,
                    n_coarse_values=np.array(CONFIG["split_n_coarse_values"]),
                    max_delay_ns=max_delay_ns,
                    mc_runs=int(CONFIG["mc_runs"]),
                    dnl_limit_lsb=float(CONFIG["dnl_limit_lsb"]),
                    num_points_power=200,
                )

                print(f"\n[{cfg_label}] coarse={coarse_mode}, fine={fine_mode}")
                print(
                    f"{'Vdd (V)':<10} {'n_coarse':<10} {'n_fine':<8} "
                    f"{'Coarse params':<56} {'Fine params':<56} "
                    f"{'P_avg (uW)':<14} {'P_max (uW)':<14} {'Pass <0.5LSB (%)':<18}"
                )
                print("-" * 206)
                for row in split_results:
                    coarse_param_desc = mode_param_text(coarse_mode, row, "coarse")
                    fine_param_desc = mode_param_text(fine_mode, row, "fine")
                    print(
                        f"{vdd_target:<10.3f} "
                        f"{row['n_coarse']:<10d} "
                        f"{row['n_fine']:<8d} "
                        f"{coarse_param_desc:<56.56} "
                        f"{fine_param_desc:<56.56} "
                        f"{row['avg_total_power_w']*1e6:<14.3f} "
                        f"{row['max_total_power_w']*1e6:<14.3f} "
                        f"{row['pass_probability_percent']:<18.2f}"
                    )

                best_entry = dict(best_split)
                best_entry["configuration"] = cfg_label
                best_entry["coarse_mode"] = coarse_mode
                best_entry["fine_mode"] = fine_mode
                best_entry["vdd"] = float(vdd_target)
                best_rows.append(best_entry)
                optimization_rows.append(best_entry)

        ranked = sorted(
            best_rows,
            key=lambda r: (
                round(r["max_total_power_w"] * 1e6, 3),
                -round(r["pass_probability_percent"], 2),
                round(r["avg_total_power_w"] * 1e6, 3),
            ),
        )

        print("\n" + "=" * 70)
        print("Best Configuration Comparison")
        print("=" * 70)
        print(
            f"{'Config':<8} {'Vdd (V)':<10} {'n_coarse':<10} {'n_fine':<8} "
            f"{'Coarse params':<56} {'Fine params':<56} "
            f"{'P_avg (uW)':<14} {'P_max (uW)':<14} {'Pass <0.5LSB (%)':<18}"
        )
        print("-" * 214)
        for row in ranked:
            coarse_param_desc = mode_param_text(row['coarse_mode'], row, "coarse")
            fine_param_desc = mode_param_text(row['fine_mode'], row, "fine")
            print(
                f"{row['configuration']:<8} "
                f"{float(row.get('vdd', np.nan)):<10.3f} "
                f"{row['n_coarse']:<10d} "
                f"{row['n_fine']:<8d} "
                f"{coarse_param_desc:<56.56} "
                f"{fine_param_desc:<56.56} "
                f"{row['avg_total_power_w']*1e6:<14.3f} "
                f"{row['max_total_power_w']*1e6:<14.3f} "
                f"{row['pass_probability_percent']:<18.2f}"
            )

        winner = ranked[0]
        winner_coarse_param_desc = mode_param_text(winner['coarse_mode'], winner, "coarse")
        winner_fine_param_desc = mode_param_text(winner['fine_mode'], winner, "fine")
        print("\nRecommended winner (priority: P_max, then pass %, then P_avg):")
        print(
            f"{winner['configuration']} at Vdd={float(winner.get('vdd', np.nan)):.3f} V "
            f"with n_coarse={winner['n_coarse']}, n_fine={winner['n_fine']}, "
            f"coarse[{winner_coarse_param_desc}], fine[{winner_fine_param_desc}], "
            f"Pass={winner['pass_probability_percent']:.2f}%, "
            f"P_avg={winner['avg_total_power_w']*1e6:.3f} uW, "
            f"P_max={winner['max_total_power_w']*1e6:.3f} uW"
        )

        pmax_plot_path = out_path("coarse_fine_opt_pmax_vs_vdd.pdf")
        _plot_optimization_metric_vs_vdd(
            records=optimization_rows,
            mode_configs=mode_configs,
            vdd_values=optimization_vdd_values,
            metric_key="max_total_power_w",
            ylabel=r"$P_{{max}} [\mu W]$",
            title="Best Split P_max vs Vdd by Configuration",
            save_path=pmax_plot_path,
        )

        pavg_plot_path = out_path("coarse_fine_opt_pavg_vs_vdd.pdf")
        _plot_optimization_metric_vs_vdd(
            records=optimization_rows,
            mode_configs=mode_configs,
            vdd_values=optimization_vdd_values,
            metric_key="avg_total_power_w",
            ylabel=r"$P_{{avg}} [\mu W]$",
            title="Best Split P_avg vs Vdd by Configuration",
            save_path=pavg_plot_path,
        )

        csv_path = out_path("coarse_fine_configuration_comparison.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "vdd",
                "configuration",
                "coarse_mode",
                "fine_mode",
                "n_coarse",
                "n_fine",
                "coarse_params",
                "fine_params",
                "p_avg_uW",
                "p_max_uW",
                "pass_probability_percent",
            ])
            for row in ranked:
                writer.writerow([
                    row.get("vdd", ""),
                    row["configuration"],
                    row["coarse_mode"],
                    row["fine_mode"],
                    row["n_coarse"],
                    row["n_fine"],
                    mode_param_text(row["coarse_mode"], row, "coarse"),
                    mode_param_text(row["fine_mode"], row, "fine"),
                    row["avg_total_power_w"] * 1e6,
                    row["max_total_power_w"] * 1e6,
                    row["pass_probability_percent"],
                ])
        print(f"Saved: {csv_path}")

        csv_vdd_path = out_path("coarse_fine_best_per_config_per_vdd.csv")
        with open(csv_vdd_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "vdd",
                "configuration",
                "coarse_mode",
                "fine_mode",
                "n_coarse",
                "n_fine",
                "coarse_params",
                "fine_params",
                "p_avg_uW",
                "p_max_uW",
                "pass_probability_percent",
            ])
            for row in sorted(
                optimization_rows,
                key=lambda r: (float(r["vdd"]), str(r["configuration"])),
            ):
                writer.writerow([
                    float(row["vdd"]),
                    row["configuration"],
                    row["coarse_mode"],
                    row["fine_mode"],
                    row["n_coarse"],
                    row["n_fine"],
                    mode_param_text(row["coarse_mode"], row, "coarse"),
                    mode_param_text(row["fine_mode"], row, "fine"),
                    row["avg_total_power_w"] * 1e6,
                    row["max_total_power_w"] * 1e6,
                    row["pass_probability_percent"],
                ])
        print(f"Saved: {csv_vdd_path}")

    if CONFIG["plot_from_csv_only"]:
        csv_path = CONFIG["save_dir"] / "coarse_fine_configuration_comparison.csv"
        print("Plotting---")
        plot_optimization_metrics_from_csv(csv_path, CONFIG["save_dir"])
    
