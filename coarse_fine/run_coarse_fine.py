"""Runner for coarse-fine DTC simulations.

Edit CONFIG to tune coarse/fine parameters and choose CS/VS/DL per block.
"""

from pathlib import Path
import sys
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coarse_fine.coarse_fine_core import (
    build_coarse_fine_dtc,
    optimize_split_loop,
    run_mc_mismatch_analysis,
)


CONFIG = {
    "save_dir": Path(
        r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\python simulation\coarse_fine_python"
    ),
    "max_delay_ns": 6.0,
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
    "optimization_vdd_values": [0.55, 0.88, 1.1],
    "optimization": False,
    "coarse_values": {
        "n": 8,
        "Cu": 2e-15,
        "Vdd": 1.1,
        "f": 50e6,
        "Ich" : 62.5e-6,
        #"Ich": 203.702e-9,
        "Cramp": 120e-15,
        "Cramp_dl": 2e-15,  # Delay-line ramp capacitance (defaults to Cramp if omitted)
        "C_ramp_cu": 2.228e-15,  # VS CDAC unit capacitance (defaults to Cramp if omitted)
        "self_power_down_vs": "yes",  # VS extra power reduction: "yes" or "no"
        "self_power_down_dl": "no",  # DL extra power reduction: "yes" or "no"
        "Ac": 5.218e-3,
        "A": 3.5,
        "C0": 8e-15,
        "dac_mode": "binary", #binary or thermometer or segmented
        "slope_mode": "variable",  # "constant" (CS), "variable" (VS), or "delay_line" (DL)
        "delay_line_selection_mode": "tapped",  # "tapped" (constant power) or "accumulated" (rising power)
        "C_fixed": 10e-15
    },
    "fine_values": {
        "n": 5,
        "Cu": 2e-15,
        "Vdd": 1.1,
        "f": 100e6,
        "Ich": 100.98e-6,
        "Cramp": 2e-15,
        "Cramp_dl": 1e-15,  # Delay-line ramp capacitance (defaults to Cramp if omitted)
        "C_ramp_cu": 2e-15,  # VS CDAC unit capacitance (defaults to Cramp if omitted)
        "self_power_down_vs": "yes",  # VS extra power reduction: "yes" or "no"
        "self_power_down_dl": "no",  # DL extra power reduction: "yes" or "no"
        "Ac": 5.218e-3,
        "A": 3.5,
        "C0":28e-15,
        "C0_scale": 1,
        "dac_mode": "binary",
        "slope_mode": "constant",  # "constant" (CS), "variable" (VS), or "delay_line" (DL)
        "delay_line_selection_mode": "tapped",  # "tapped" (constant power) or "accumulated" (rising power)
        "C_fixed": 0
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
) -> None:
    """Plot one optimization metric versus Vdd for all mode configurations."""
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = [
        '#1F77B4', '#D62728', '#2CA02C', '#FF7F0E', '#8C564B',
        '#17BECF', '#E377C2', '#7F7F7F', '#BCBD22',
    ]

    for idx, (cfg_label, _, _) in enumerate(mode_configs):
        cfg_rows = [
            r for r in records
            if r["configuration"] == cfg_label and float(r["vdd"]) in [float(v) for v in vdd_values]
        ]
        cfg_rows = sorted(cfg_rows, key=lambda r: float(r["vdd"]))
        if len(cfg_rows) == 0:
            continue

        x_vdd = [float(r["vdd"]) for r in cfg_rows]
        y_metric = [float(r[metric_key]) * 1e6 for r in cfg_rows]

        split_text = ", ".join(
            f"{float(r['vdd']):.2f}V:{int(r['n_coarse'])}+{int(r['n_fine'])}"
            for r in cfg_rows
        )
        label = f"{cfg_label} [{split_text}]"

        ax.plot(
            x_vdd,
            y_metric,
            marker='o',
            markersize=6,
            linewidth=2.2,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Vdd [V]', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.96, edgecolor='black', loc='best')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def main() -> None:
    save_dir = CONFIG["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    def out_path(name: str) -> str:
        return str(save_dir / name)

    def apply_mode_voltage_scaling(values: dict) -> dict:
        mode = str(values.get("slope_mode", "constant")).strip().lower()
        scale_constant = float(CONFIG["voltage_scale_factor_constant"])
        scale_var_dl = float(CONFIG["voltage_scale_factor_variable_delay_line"])
        scale = scale_constant if mode == "constant" else scale_var_dl

        out = dict(values)
        out["Vdd"] = float(out["Vdd"]) * scale
        if out.get("Vth") is not None:
            out["Vth"] = float(out["Vth"]) * scale
        return out

    coarse_values = apply_mode_voltage_scaling(dict(CONFIG["coarse_values"]))
    fine_values = apply_mode_voltage_scaling(dict(CONFIG["fine_values"]))

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


    # Keep internal mode-dependent scaling neutral so runner values are used as-is.
    for blk in (coarse_values, fine_values):
        blk["vdd_vth_factor_variable"] = 1.0
        blk["vdd_vth_factor_constant"] = 1.0

    architecture = build_coarse_fine_dtc(coarse_values, fine_values)

    target_ns = 5.0
    result = architecture.synthesize_delay(target_ns * 1e-9)

    print("Coarse-Fine DTC result")
    print("-" * 50)
    print(f"Target delay: {result['target_delay_s'] * 1e9:.3f} ns")
    print(f"Total delay : {result['total_delay_s'] * 1e9:.3f} ns")
    print(f"Error       : {result['error_s'] * 1e12:.2f} ps")
    print(f"Coarse code : {int(result['coarse_code'])}")
    print(f"Fine code   : {int(result['fine_code'])}")
    print(f"Total power : {result['p_total_w'] * 1e6:.3f} uW")

    max_delay_ns = float(CONFIG["max_delay_ns"])

    architecture.plot_characteristic_vs_code(t_range_ns=max_delay_ns, save_path=out_path('coarse_fine_delay_vs_code.png'))
    architecture.plot_power_vs_code(
        avg_over_target_range_ns=max_delay_ns,
        avg_num_points=200,
        save_path=out_path('coarse_fine_power_vs_code.png'),
    )

    architecture.plot_single_block_characteristic('coarse', save_path=out_path('coarse_only_delay_vs_code.png'))
    architecture.plot_single_block_nonlinearity('coarse', save_path=out_path('coarse_only_dnl_inl.png'))
    architecture.plot_coarse_nonlinearity_ps(mc_runs=int(CONFIG["mc_runs"]), save_path=out_path('coarse_only_dnl_inl_ps.png'))
    architecture.plot_single_block_characteristic('fine', save_path=out_path('fine_only_delay_vs_code.png'))
    architecture.plot_single_block_nonlinearity('fine', save_path=out_path('fine_only_dnl_inl.png'))

    ch_all = architecture.combined_characteristic()
    code_last = len(ch_all['combined_code']) - 1
    code_mid = code_last // 2

    # architecture.plot_phase_noise(
    #     codes=[0, code_mid, code_last],
    #     num_samples=2**14,
    #     nperseg=1024,
    #     temperature_k=100e-3,
    #     seed=42,
    #     save_path=out_path('coarse_fine_phase_noise.png'),
    # )

    # sigma_from_target = architecture.max_sigma_jitter_tolerated_from_lf(
    #     l_dbc_hz=-100.0,
    #     integration_bw_hz=100e6,
    # )
    # print(
    #     f"Max tolerated sigma_jitter from flat L(f)=-100 dBc/Hz over 1 MHz: "
    #     f"{sigma_from_target*1e12:.3f} ps"
    # )

    _, _, mc_stats = run_mc_mismatch_analysis(
        coarse_values=coarse_values,
        fine_values=fine_values,
        mc_runs=int(CONFIG["mc_runs"]),
        delay_range_ns=max_delay_ns,
        num_points=120,
        dnl_limit_lsb=float(CONFIG["dnl_limit_lsb"]),
        t_range_ns=max_delay_ns,
        save_path=out_path('coarse_fine_mc_mismatch.png'),
    )
    print(f"Probability of staying below {CONFIG['dnl_limit_lsb']:.1f} LSB = {mc_stats['pass_probability_percent']:.2f}%")

    optimization = CONFIG.get("optimization", False)   # Set to False to skip the optimization loop and just run the main characterization.
    if optimization:
        print("\n" + "=" * 70)
        print("Split Optimization Loop - Mode Configurations")
        print("=" * 70)

        mode_configs = [
            ("CS-CS", "constant", "constant"),
            ("CS-VS", "constant", "variable"),
            ("VS-CS", "variable", "constant"),
            ("VS-VS", "variable", "variable"),
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

                cfg_coarse = apply_mode_voltage_scaling(cfg_coarse)
                cfg_fine = apply_mode_voltage_scaling(cfg_fine)

                for blk in (cfg_coarse, cfg_fine):
                    blk["vdd_vth_factor_variable"] = 1.0
                    blk["vdd_vth_factor_constant"] = 1.0

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

        pmax_plot_path = out_path("coarse_fine_opt_pmax_vs_vdd.png")
        _plot_optimization_metric_vs_vdd(
            records=optimization_rows,
            mode_configs=mode_configs,
            vdd_values=optimization_vdd_values,
            metric_key="max_total_power_w",
            ylabel="Best P_max [uW]",
            title="Best Split P_max vs Vdd by Configuration",
            save_path=pmax_plot_path,
        )

        pavg_plot_path = out_path("coarse_fine_opt_pavg_vs_vdd.png")
        _plot_optimization_metric_vs_vdd(
            records=optimization_rows,
            mode_configs=mode_configs,
            vdd_values=optimization_vdd_values,
            metric_key="avg_total_power_w",
            ylabel="Best P_avg [uW]",
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


if __name__ == "__main__":
    optimization = False  # Set to False to skip the optimization loop and just run the main characterization.<
    main()
