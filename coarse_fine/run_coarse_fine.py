"""Runner for coarse-fine DTC simulations.

Edit CONFIG to tune coarse/fine parameters and choose CS/VS/DL per block.
"""

from pathlib import Path
import sys
import csv

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
    "voltage_scale_factor_constant": 0.5,
    "voltage_scale_factor_variable_delay_line": 0.5,
    "split_n_total": 11,
    "split_n_coarse_values": np.arange(2, 9),
    "include_delay_line_modes": True,
    "coarse_values": {
        "n": 6,
        "Cu": 2e-15,
        "Vdd": 1.1,
        "f": 100e6,
        "Ich" : 1.733e-6,
        #"Ich": 203.702e-9,
        "Cramp": 1e-15,
        "Cramp_dl": 0.5e-15,  # Delay-line ramp capacitance (defaults to Cramp if omitted)
        "C_ramp_cu": 2e-15,  # VS CDAC unit capacitance (defaults to Cramp if omitted)
        "self_power_down": "yes",  # VS-only extra power reduction: "yes" or "no"
        "Ac": 5.218e-3,
        "A": 3.5,
        "C0_scale": 1,
        "dac_mode": "thermometer",
        "slope_mode": "delay_line",  # "constant" (CS), "variable" (VS), or "delay_line" (DL)
    },
    "fine_values": {
        "n": 5,
        "Cu": 2e-15,
        "Vdd": 1.1,
        "f": 100e6,
        "Ich": 41.580e-6,
        "Cramp": 0.5e-15,
        "Cramp_dl": 0.5e-15,  # Delay-line ramp capacitance (defaults to Cramp if omitted)
        "C_ramp_cu": 2e-15,  # VS CDAC unit capacitance (defaults to Cramp if omitted)
        "self_power_down": "yes",  # VS-only extra power reduction: "yes" or "no"
        "Ac": 5.218e-3,
        "A": 3.5,
        "C0_scale": 1,
        "dac_mode": "binary",
        "slope_mode": "delay_line",  # "constant" (CS), "variable" (VS), or "delay_line" (DL)
    },
}


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
    architecture.plot_phase_noise(
        codes=[0, code_mid, code_last],
        num_samples=2**14,
        nperseg=1024,
        temperature_k=100e-3,
        seed=42,
        save_path=out_path('coarse_fine_phase_noise.png'),
    )

    sigma_from_target = architecture.max_sigma_jitter_tolerated_from_lf(
        l_dbc_hz=-100.0,
        integration_bw_hz=100e6,
    )
    print(
        f"Max tolerated sigma_jitter from flat L(f)=-100 dBc/Hz over 1 MHz: "
        f"{sigma_from_target*1e12:.3f} ps"
    )

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
            ("CS-DL", "constant", "delay_line"),
            ("DL-CS", "delay_line", "constant"),
            ("VS-DL", "variable", "delay_line"),
            ("DL-VS", "delay_line", "variable"),
            ("DL-DL", "delay_line", "delay_line"),
        ])

    best_rows = []

    for cfg_label, coarse_mode, fine_mode in mode_configs:
        cfg_coarse = dict(CONFIG["coarse_values"])
        cfg_fine = dict(CONFIG["fine_values"])
        cfg_coarse["slope_mode"] = coarse_mode
        cfg_fine["slope_mode"] = fine_mode

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
        print(f"{'n_coarse':<10} {'n_fine':<8} {'Ich_coarse (uA)':<16} {'Ich_fine (uA)':<14} {'P_avg (uW)':<14} {'P_max (uW)':<14} {'Pass <0.5LSB (%)':<18}")
        print("-" * 110)
        for row in split_results:
            print(
                f"{row['n_coarse']:<10d} "
                f"{row['n_fine']:<8d} "
                f"{row['ich_coarse_a']*1e6:<16.3f} "
                f"{row['ich_fine_a']*1e6:<14.3f} "
                f"{row['avg_total_power_w']*1e6:<14.3f} "
                f"{row['max_total_power_w']*1e6:<14.3f} "
                f"{row['pass_probability_percent']:<18.2f}"
            )

        best_entry = dict(best_split)
        best_entry["configuration"] = cfg_label
        best_entry["coarse_mode"] = coarse_mode
        best_entry["fine_mode"] = fine_mode
        best_rows.append(best_entry)

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
    print(f"{'Config':<8} {'n_coarse':<10} {'n_fine':<8} {'Ich_coarse (uA)':<16} {'Ich_fine (uA)':<14} {'P_avg (uW)':<14} {'P_max (uW)':<14} {'Pass <0.5LSB (%)':<18}")
    print("-" * 118)
    for row in ranked:
        print(
            f"{row['configuration']:<8} "
            f"{row['n_coarse']:<10d} "
            f"{row['n_fine']:<8d} "
            f"{row['ich_coarse_a']*1e6:<16.3f} "
            f"{row['ich_fine_a']*1e6:<14.3f} "
            f"{row['avg_total_power_w']*1e6:<14.3f} "
            f"{row['max_total_power_w']*1e6:<14.3f} "
            f"{row['pass_probability_percent']:<18.2f}"
        )

    winner = ranked[0]
    print("\nRecommended winner (priority: P_max, then pass %, then P_avg):")
    print(
        f"{winner['configuration']} with n_coarse={winner['n_coarse']}, n_fine={winner['n_fine']}, "
        f"Pass={winner['pass_probability_percent']:.2f}%, "
        f"P_avg={winner['avg_total_power_w']*1e6:.3f} uW, "
        f"P_max={winner['max_total_power_w']*1e6:.3f} uW"
    )

    csv_path = out_path("coarse_fine_configuration_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration",
            "coarse_mode",
            "fine_mode",
            "n_coarse",
            "n_fine",
            "ich_coarse_uA",
            "ich_fine_uA",
            "p_avg_uW",
            "p_max_uW",
            "pass_probability_percent",
        ])
        for row in ranked:
            writer.writerow([
                row["configuration"],
                row["coarse_mode"],
                row["fine_mode"],
                row["n_coarse"],
                row["n_fine"],
                row["ich_coarse_a"] * 1e6,
                row["ich_fine_a"] * 1e6,
                row["avg_total_power_w"] * 1e6,
                row["max_total_power_w"] * 1e6,
                row["pass_probability_percent"],
            ])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
