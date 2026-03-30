from pathlib import Path

import numpy as np

from dtc.dtc_core import (
    ConstantSlopeDTC,
    VariableSlopeDTC,
    compute_sigma_c,
    configure_plot_style,
    lsb_from_curve,
    plot_aux_effects,
    plot_delay_power,
    plot_mc_dnl_inl,
    print_summary,
    run_constant_slope_simulation,
    run_variable_slope_simulation,
)


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
    "n_bits": 5,
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
    "f_hz": 66.67e6,
    "ich_constant": 0.22e-6,
    "ich_variable": 6.9e-6,
    "self_power_down_variable": "yes",  # "yes" or "no"
    "cramp": 2e-15,
    "c1": 0.323,
    "c2": -0.09,
    "i1": 0.184,
    "c0_factor": 1.0,
    "mc_runs": 100,
    "mc_modes": ["binary", "segmented"],
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
        cramp_u=config["cramp"],
        i1=config["i1"],
        c1=config["c1"],
        c2=config["c2"],
        self_power_down=config.get("self_power_down_variable", True),
    )


def main():
    configure_plot_style()

    save_path = CONFIG["save_dir"] / CONFIG["plot_folder"]
    constant_save_path = save_path / "constant_slope"
    variable_save_path = save_path / "variable_slope"

    sigma_c = compute_sigma_c(CONFIG["ac_nm"], CONFIG["area_um2"], CONFIG["cu"])

    vdd_cs = CONFIG["vdd"] * float(CONFIG["voltage_scale_factor_constant"])
    vth_cs = CONFIG["vth"] * float(CONFIG["voltage_scale_factor_constant"])
    vdd_vs = CONFIG["vdd"] * float(CONFIG["voltage_scale_factor_variable"])
    vth_vs = CONFIG["vth"] * float(CONFIG["voltage_scale_factor_variable"])

    sim = build_constant_sim(CONFIG, sigma_c, CONFIG["dac_mode"], vdd_cs, vth_cs)
    sim_var = build_variable_sim(CONFIG, sigma_c, CONFIG["dac_mode"], vdd_vs, vth_vs)

    const_results = run_constant_slope_simulation(
        sim=sim,
        freq_hz=CONFIG["f_hz"],
        mismatch_enable=CONFIG["run_flags"]["DAC_mismatch"],
        c0_factor=CONFIG["c0_factor"],
    )

    var_results = run_variable_slope_simulation(
        sim=sim_var,
        vst_array=const_results["vst_array_full"],
        freq_hz=CONFIG["f_hz"],
        mismatch_enable=CONFIG["run_flags"]["DAC_mismatch"],
        remove_index=None,
    )

    plot_delay_power(
        const_results["codes"],
        const_results["delay_array"],
        const_results["power_array"],
        constant_save_path,
        "Constant Slope",
    )
    plot_delay_power(
        var_results["codes"],
        var_results["delay_array"],
        var_results["power_array"],
        variable_save_path,
        "Variable Slope",
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
        "Constant Slope",
    )
    plot_aux_effects(
        var_results["codes"],
        vs_ich_plot,
        vs_cramp_plot,
        variable_save_path,
        "Variable Slope",
    )

    for mc_mode in CONFIG["mc_modes"]:
        sim_mc = build_constant_sim(CONFIG, sigma_c, mc_mode, vdd_cs, vth_cs)
        sim_var_mc = build_variable_sim(CONFIG, sigma_c, mc_mode, vdd_vs, vth_vs)

        mc_cs = []
        mc_vs = []
        for _ in range(CONFIG["mc_runs"]):
            cs_mc = run_constant_slope_simulation(
                sim=sim_mc,
                freq_hz=CONFIG["f_hz"],
                mismatch_enable=True,
                c0_factor=CONFIG["c0_factor"],
                report_mismatch=False,
            )
            vs_mc = run_variable_slope_simulation(
                sim=sim_var_mc,
                vst_array=cs_mc["vst_array_full"],
                freq_hz=CONFIG["f_hz"],
                mismatch_enable=True,
                remove_index=None,
                report_mismatch=False,
            )
            mc_cs.append(cs_mc["delay_array"])
            mc_vs.append(vs_mc["delay_array"])

        mode_label = mc_mode.capitalize()
        plot_mc_dnl_inl(mc_cs, constant_save_path, f"Constant Slope {mode_label}", CONFIG["mc_runs"])
        plot_mc_dnl_inl(mc_vs, variable_save_path, f"Variable Slope {mode_label}", CONFIG["mc_runs"])

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

    print(f"\nAll requested plots saved to:\n- {constant_save_path}\n- {variable_save_path}")


if __name__ == "__main__":
    main()
