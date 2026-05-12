from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dtc.dtc_core import (
    ConstantSlopeDTC,
    VariableSlopeDTC,
    compute_sigma_c,
    lsb_from_curve,
    plot_aux_effects,
    plot_delay_power,
    plot_mc_dnl_inl,
    print_summary,
    run_constant_slope_simulation,
    run_variable_slope_simulation,
)
from plot_style import apply_science_style


# ---------------------------
# Easy configuration section
# ---------------------------
CONFIG = {
    "save_dir": Path(r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP"),
    "plot_folder": "plot_python",
    "run_flags": {
        "DAC_mismatch": True,
        "CLM": False,
        "Non-linearities-capacitor": False,
    },
    "n_bits": 11,
    "cu": 2e-15,
    "dac_mode": "binary",  # baseline mode for delay/power/summaries
    "segmented_thermo_bits": 4,
    "ac_nm": 5.218e-3,
    "area_um2": 3.5,
    "vdd": 1.1,
    "vth": 0.55,
    # Apply directly to Vdd and Vth in runner.
    "voltage_scale_factor_constant": 1,
    "voltage_scale_factor_variable": 1,
    "f_hz":100e6,
    "ich_constant": 26.4e-6,
    "ich_variable": 6.9e-6,
    "divide_ich_by_c0_factor": True,
    "self_power_down_variable": "yes",  # "yes" or "no"
    "cramp": 2e-15,
    "cramp_u": 2e-15,  # Only used for variable slope DAC. If not provided, "cramp" value will be used.
    "c1": 0.323,
    "c2": -0.09,
    "i1": 0.184,
    "c0_factor": 1.0,  # used when c0_factors is not provided
    "c0_factors": [0.5, 1.0, 1.5],
    "mc_runs": 100,
    "mc_modes": ["binary", "segmented"],
    "C_fixed" : 120e-15, # Fixed capacitance added to each DAC code (e.g. from routing or intentional cap)
}


def build_constant_sim(config, sigma_c, dac_mode, vdd_eff, vth_eff):
    return ConstantSlopeDTC(
        n_bits=config["n_bits"],
        cu=config["cu"],
        dac_mode=dac_mode,
        thermo_bits=config["segmented_thermo_bits"],
        run_flags=config["run_flags"],
        sigma_c=sigma_c,
        vdd=vdd_eff,
        vth=vth_eff,
        ich=config["ich_constant"],
        cramp=config["cramp"],
        i1=config["i1"],
        c1=config["c1"],
        c2=config["c2"],
    )


def build_variable_sim(config, sigma_c, dac_mode, vdd_eff, vth_eff):
    return VariableSlopeDTC(
        n_bits=config["n_bits"],
        dac_mode=dac_mode,
        thermo_bits=config["segmented_thermo_bits"],
        run_flags=config["run_flags"],
        sigma_c=sigma_c,
        vdd=vdd_eff,
        vth=vth_eff,
        ich=config["ich_variable"],
        cramp_u=config["cramp_u"],
        i1=config["i1"],
        c1=config["c1"],
        c2=config["c2"],
        self_power_down=config.get("self_power_down_variable", True),
        C_fixed=config.get("C_fixed", 0),
    )


def main():
    apply_science_style()

    save_path = CONFIG["save_dir"] / CONFIG["plot_folder"]
    save_path.mkdir(parents=True, exist_ok=True)

    sigma_c = compute_sigma_c(CONFIG["ac_nm"], CONFIG["area_um2"], CONFIG["cu"])

    vdd_cs = CONFIG["vdd"] * float(CONFIG["voltage_scale_factor_constant"])
    vth_cs = CONFIG["vth"] * float(CONFIG["voltage_scale_factor_constant"])
    vdd_vs = CONFIG["vdd"] * float(CONFIG["voltage_scale_factor_variable"])
    vth_vs = CONFIG["vth"] * float(CONFIG["voltage_scale_factor_variable"])

    c0_factors = CONFIG.get("c0_factors")
    if c0_factors is None:
        c0_factors = [float(CONFIG["c0_factor"])]
    elif np.isscalar(c0_factors):
        c0_factors = [float(c0_factors)]
    else:
        c0_factors = [float(f) for f in c0_factors]

    power_sweep_cs = []
    power_sweep_vs = []

    for c0_factor in c0_factors:
        if c0_factor == 0:
            raise ValueError("C0 factor cannot be 0 when sweeping.")

        ich_scale = (2.0 / (1+c0_factor)) if CONFIG.get("divide_ich_by_c0_factor", False) else 1.0
        sim_config = {**CONFIG}
        sim_config["ich_constant"] = CONFIG["ich_constant"] * ich_scale
        sim_config["ich_variable"] = CONFIG["ich_variable"] * ich_scale

        factor_tag = f"c0_{c0_factor:g}".replace(".", "p")
        factor_root = save_path / factor_tag
        constant_save_path = factor_root / "constant_slope"
        variable_save_path = factor_root / "variable_slope"

        print(
            f"\n=== Running sweep for C0 factor = {c0_factor:g} "
            f"| Ich scale = x{ich_scale:g} ==="
        )

        sim = build_constant_sim(sim_config, sigma_c, sim_config["dac_mode"], vdd_cs, vth_cs)
        sim_var = build_variable_sim(sim_config, sigma_c, sim_config["dac_mode"], vdd_vs, vth_vs)

        const_results = run_constant_slope_simulation(
            sim=sim,
            freq_hz=CONFIG["f_hz"],
            mismatch_enable=CONFIG["run_flags"]["DAC_mismatch"],
            c0_factor=c0_factor,
            cramp=CONFIG["cramp"]
        )

        var_results = run_variable_slope_simulation(
            sim=sim_var,
            vst_array=np.full(2**CONFIG["n_bits"] - 1, CONFIG["vdd"]),
            freq_hz=CONFIG["f_hz"],
            mismatch_enable=CONFIG["run_flags"]["DAC_mismatch"],
        )

        power_sweep_cs.append(
            {
                "c0_factor": c0_factor,
                "codes": const_results["codes"],
                "power": const_results["power_array"],
            }
        )
        power_sweep_vs.append(
            {
                "c0_factor": c0_factor,
                "codes": var_results["codes"],
                "power": var_results["power_array"],
            }
        )

        plot_delay_power(
            const_results["codes"],
            const_results["delay_array"],
            const_results["power_array"],
            constant_save_path,
            f"Constant Slope (C0 x{c0_factor:g})",
        )
        plot_delay_power(
            var_results["codes"],
            var_results["delay_array"],
            var_results["power_array"],
            variable_save_path,
            f"Variable Slope (C0 x{c0_factor:g})",
        )

        # Plot CLM/nonlinearity auxiliary curves only when their flags are enabled.
        cs_ich_plot = const_results["ich_array"] if CONFIG["run_flags"]["CLM"] else np.zeros_like(const_results["ich_array"])
        cs_cramp_plot = (
            const_results["cramp_array"]
            if CONFIG["run_flags"]["Non-linearities-capacitor"]
            else np.zeros_like(const_results["cramp_array"])
        )
        vs_ich_plot = var_results["ich_array"] if CONFIG["run_flags"]["CLM"] else np.zeros_like(var_results["ich_array"])
        vs_cramp_plot = (
            var_results["cramp_array"]
            if CONFIG["run_flags"]["Non-linearities-capacitor"]
            else np.zeros_like(var_results["cramp_array"])
        )

        plot_aux_effects(
            const_results["codes"],
            cs_ich_plot,
            cs_cramp_plot,
            constant_save_path,
            f"Constant Slope (C0 x{c0_factor:g})",
        )
        plot_aux_effects(
            var_results["codes"],
            vs_ich_plot,
            vs_cramp_plot,
            variable_save_path,
            f"Variable Slope (C0 x{c0_factor:g})",
        )

        for mc_mode in CONFIG["mc_modes"]:
            sim_mc = build_constant_sim(sim_config, sigma_c, mc_mode, vdd_cs, vth_cs)
            sim_var_mc = build_variable_sim(sim_config, sigma_c, mc_mode, vdd_vs, vth_vs)

            mc_cs = []
            mc_vs = []
            for _ in range(CONFIG["mc_runs"]):
                cs_mc = run_constant_slope_simulation(
                    sim=sim_mc,
                    freq_hz=CONFIG["f_hz"],
                    mismatch_enable=True,
                    cramp=CONFIG["cramp"],
                    c0_factor=c0_factor,
                    report_mismatch=False,
                )
                
                vs_mc = run_variable_slope_simulation(
                    sim=sim_var_mc,
                    vst_array=np.full(2**CONFIG["n_bits"] - 1, CONFIG["vdd"]),
                    freq_hz=CONFIG["f_hz"],
                    mismatch_enable=True,
                    report_mismatch=False,
                )
                mc_cs.append(cs_mc["delay_array"])
                mc_vs.append(vs_mc["delay_array"])

            mode_label = mc_mode.capitalize()
            plot_mc_dnl_inl(
                mc_cs,
                constant_save_path,
                f"Constant Slope {mode_label} (C0 x{c0_factor:g})",
                CONFIG["mc_runs"],
            )
            plot_mc_dnl_inl(
                mc_vs,
                variable_save_path,
                f"Variable Slope {mode_label} (C0 x{c0_factor:g})",
                CONFIG["mc_runs"],
            )

        delay_cs_clm = sim.compute_delay(const_results["vst_array_full"], clm_enabled=True, nonlin_enabled=False)[0]
        delay_cs_nonlin = sim.compute_delay(const_results["vst_array_full"], clm_enabled=False, nonlin_enabled=True)[0]
        delay_cs_both = sim.compute_delay(const_results["vst_array_full"], clm_enabled=True, nonlin_enabled=True)[0]

        delay_cs_clm = delay_cs_clm[[i for i in range(len(delay_cs_clm)) if i != const_results["half"]]]
        delay_cs_nonlin = delay_cs_nonlin[[i for i in range(len(delay_cs_nonlin)) if i != const_results["half"]]]
        delay_cs_both = delay_cs_both[[i for i in range(len(delay_cs_both)) if i != const_results["half"]]]

        delay_vs_clm = sim_var.compute_delay(
            const_results["vst_array_full"], var_results["cap_array"], clm_enabled=True, nonlin_enabled=False
        )[0]
        delay_vs_nonlin = sim_var.compute_delay(
            const_results["vst_array_full"], var_results["cap_array"], clm_enabled=False, nonlin_enabled=True
        )[0]
        delay_vs_both = sim_var.compute_delay(
            const_results["vst_array_full"], var_results["cap_array"], clm_enabled=True, nonlin_enabled=True
        )[0]

        print_summary(
            "CS",
            const_results["delay_array"],
            const_results["power_array_full"],
            CONFIG["f_hz"],
            const_results["vst_array_full"],
            lsb_from_curve(delay_cs_clm),
            lsb_from_curve(delay_cs_nonlin),
            lsb_from_curve(delay_cs_both),
        )

        print_summary(
            "VS",
            var_results["delay_array"],
            var_results["power_array_full"],
            CONFIG["f_hz"],
            const_results["vst_array_full"],
            lsb_from_curve(delay_vs_clm),
            lsb_from_curve(delay_vs_nonlin),
            lsb_from_curve(delay_vs_both),
        )

        print(f"All requested plots saved to:\n- {constant_save_path}\n- {variable_save_path}")

    fig, (ax_cs, ax_vs) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for item in power_sweep_cs:
        ax_cs.plot(item["codes"], item["power"] * 1e6, lw=2, label=f"C0 x{item['c0_factor']:g}")
    ax_cs.set_title("Constant Slope Power vs Code (C0 Sweep)")
    ax_cs.set_ylabel("Power (uW)")
    ax_cs.grid(True, alpha=0.3)
    ax_cs.legend(frameon=False)

    for item in power_sweep_vs:
        ax_vs.plot(item["codes"], item["power"] * 1e6, lw=2, label=f"C0 x{item['c0_factor']:g}")
    ax_vs.set_title("Variable Slope Power vs Code (C0 Sweep)")
    ax_vs.set_xlabel("Digital Code")
    ax_vs.set_ylabel("Power (uW)")
    ax_vs.grid(True, alpha=0.3)
    ax_vs.legend(frameon=False)

    fig.tight_layout()
    combined_power_path = save_path / "power_comparison_c0_sweep.png"
    fig.savefig(combined_power_path, dpi=300)
    plt.close(fig)

    print(f"Combined power sweep figure saved to:\n- {combined_power_path}")


if __name__ == "__main__":
    main()
