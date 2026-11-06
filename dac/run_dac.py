from pathlib import Path

import numpy as np

from dac.dac_core import (
    ChargeSharingDAC,
    SignMagnitudeDAC,
    analytic_n_sigma,
    mc_n_sigma,
    mc_worst_case,
    plot_mc_linearity,
    plot_power_vs_vout,
    plot_voltage_power,
    plot_ramp_time_vs_code,
    plot_vref_sweep,
    run_mc_linearity,
)
from plot_style import apply_science_style


# ---------------------------
# Easy configuration section
# ---------------------------
CONFIG = {
    # Python plots live outside results/ (that one is for Cadence exports).
    "save_dir": Path(__file__).resolve().parents[1] / "plots_python",
    "plot_folder": "dac_comparison",
    "cload": 1.1e-12,
    "dv_max": 1e-3,          # max allowed step (resolution) for both DACs
    "v_range_dac1": 0.3,     # DAC1 output covers 0 .. +v_range
    "v_fs_dac2": 0.3,        # DAC2 output covers -v_fs .. +v_fs
    "dac_mode": "binary",    # "binary" or "thermometer"
    "ac_nm": 6.3e-3,         # Pelgrom coefficient (sigma_C/C = Ac / sqrt(area))
    "cap_density": 1.21e-15,    # F/um^2, same as DTC config (1 fF on 1 um^2)
    "cload_term_factor": 0.5,  # DAC2 extra energy alpha * Cload * Vout^2
    "f_hz": 50e6,
    "vref_points": [0.35, 0.6, 0.8, 1.1],  # operating points for the per-code plots
    "mc_runs": 100,
    "vref_sweep": np.linspace(0.35, 1.2, 35),
    "sweep_mc_runs": 100,
    "sweep_percentile": 99,  # linearity reported as this percentile of max |error|
    "slow": True,            # DAC1 only: constant-current ramp sized on the full-scale code
    "ramp_times": [50e-9],  # full-scale ramp time
}


def build_dac1(config, vref, dac_mode):
    return ChargeSharingDAC(
        vref=vref,
        cload=config["cload"],
        dv_max=config["dv_max"],
        v_range=config["v_range_dac1"],
        dac_mode=dac_mode,
        ac_nm=config["ac_nm"],
        cap_density=config["cap_density"],
    )


def build_dac2(config, vref, dac_mode):
    return SignMagnitudeDAC(
        vref=vref,
        cload=config["cload"],
        dv_max=config["dv_max"],
        v_fs=config["v_fs_dac2"],
        dac_mode=dac_mode,
        ac_nm=config["ac_nm"],
        cap_density=config["cap_density"],
        cload_term_factor=config["cload_term_factor"],
    )


def print_ramp(q_min, q_fs, ramp_times):
    """I is sized on full scale (largest charge), so code 1 is q_fs / q_min faster."""
    print(f"Q(code 1) = {q_min*1e18:.1f} aC | Q(FS) = {q_fs*1e15:.1f} fC | t_FS / t_code1 = {q_fs/q_min:.0f}")
    for t in ramp_times:
        i_ramp = q_fs / t
        print(f"  t_FS = {t*1e9:4g} ns -> I = {i_ramp*1e6:6.2f} uA | t_code1 = {q_min/i_ramp*1e12:7.1f} ps")


def print_summary(label, dac, v_out, power_w, dnl_runs, inl_runs, percentile):
    steps = np.diff(v_out)
    print(f"\n[{label}]  Vref = {dac.vref:.3f} V")
    print(f"Bits = {dac.n_bits} | Cu = {dac.cu*1e15:.3f} fF | sigma_Cu/Cu = {dac.sigma_c/dac.cu*100:.3f} %")
    print(f"C_DAC,tot = {dac.total_cap()*1e15:.1f} fF | Area = {dac.total_area_um2():.0f} um^2 | codes used = {len(v_out)}")
    print(f"Vout range = {v_out[0]*1e3:.2f} .. {v_out[-1]*1e3:.2f} mV")
    print(f"Max step = {steps.max()*1e3:.4f} mV | Min step = {steps.min()*1e3:.4f} mV")
    print(f"Worst-case power = {power_w.max()*1e6:.3f} uW | Mean power = {power_w.mean()*1e6:.3f} uW")
    print(
        f"MC {percentile}th pct: max|DNL| = {mc_worst_case(dnl_runs, percentile):.3f} LSB, "
        f"max|INL| = {mc_worst_case(inl_runs, percentile):.3f} LSB"
    )
    print(
        f"MC 3-sigma (worst code): DNL = {mc_n_sigma(dnl_runs):.3f} LSB, "
        f"INL = {mc_n_sigma(inl_runs):.3f} LSB"
    )
    dnl_an, inl_an = analytic_n_sigma(dac)
    print(f"Analytic 3-sigma (worst code): DNL = {dnl_an:.3f} LSB, INL = {inl_an:.3f} LSB")


def run_vref_sweep(config):
    sweep = {}
    builders = {"DAC1 (charge sharing)": build_dac1, "DAC2 (sign-magnitude)": build_dac2}
    for label, builder in builders.items():
        rows = []
        for vref in config["vref_sweep"]:
            dac = builder(config, vref, config["dac_mode"])
            arr = dac.build_dac_array(mismatch_enable=False)
            v_out = dac.voltage(arr)
            energy = dac.energy(v_out) if isinstance(dac, ChargeSharingDAC) else dac.energy(arr)
            dnl_runs, inl_runs = run_mc_linearity(dac, config["sweep_mc_runs"])
            rows.append((
                vref,
                energy.max() * config["f_hz"],
                dac.total_cap(),
                dac.total_area_um2(),
                dac.n_bits,
                mc_worst_case(dnl_runs, config["sweep_percentile"]),
                mc_worst_case(inl_runs, config["sweep_percentile"]),
                mc_n_sigma(dnl_runs),
                mc_n_sigma(inl_runs),
                *analytic_n_sigma(dac),
            ))
        cols = np.array(rows).T
        sweep[label] = dict(zip(["vref", "power_wc", "total_cap", "area_um2", "n_bits", "dnl_wc", "inl_wc", "dnl_3s", "inl_3s", "dnl_3s_an", "inl_3s_an"], cols))
    return sweep


def main():
    apply_science_style()
    np.random.seed(0)

    save_path = CONFIG["save_dir"] / CONFIG["plot_folder"]
    save_path.mkdir(parents=True, exist_ok=True)
    mode = CONFIG["dac_mode"]

    for vref in CONFIG["vref_points"]:
        out_dir = save_path / f"{mode}_vref_{vref:.2f}V".replace(".", "p")
        print(f"\n=== Vref = {vref:g} V | {mode} ===")

        dac1 = build_dac1(CONFIG, vref, mode)
        arr1 = dac1.build_dac_array(mismatch_enable=False)
        v1 = dac1.voltage(arr1)
        p1 = dac1.energy(v1) * CONFIG["f_hz"]

        dac2 = build_dac2(CONFIG, vref, mode)
        arr2 = dac2.build_dac_array(mismatch_enable=False)
        v2 = dac2.voltage(arr2)
        p2 = dac2.energy(arr2) * CONFIG["f_hz"]

        plot_voltage_power(dac1.codes, v1, p1, out_dir, f"DAC1 {mode}", "dac1")
        plot_voltage_power(dac2.codes, v2, p2, out_dir, f"DAC2 {mode}", "dac2")
        plot_power_vs_vout(
            [
                {"label": "DAC1", "v_out": v1, "power": p1},
                {"label": "DAC2", "v_out": v2, "power": p2},
            ],
            out_dir,
            "dac1_vs_dac2",
        )

        dnl1, inl1 = run_mc_linearity(dac1, CONFIG["mc_runs"])
        dnl2, inl2 = run_mc_linearity(dac2, CONFIG["mc_runs"])
        plot_mc_linearity(dac1.codes, dnl1, inl1, out_dir, f"DAC1 {mode}", "dac1", "LSB_{local}")
        plot_mc_linearity(dac2.codes, dnl2, inl2, out_dir, f"DAC2 {mode}", "dac2", "LSB")

        print_summary("DAC1", dac1, v1, p1, dnl1, inl1, CONFIG["sweep_percentile"])
        print_summary("DAC2", dac2, v2, p2, dnl2, inl2, CONFIG["sweep_percentile"])

        if CONFIG["slow"]:
            q1 = dac1.ramp_charge(v1)
            print("\n[DAC1 slow: constant-current ramp]")
            print_ramp(q1[1], q1[-1], CONFIG["ramp_times"])
            plot_ramp_time_vs_code(dac1.codes, q1, CONFIG["ramp_times"], CONFIG["f_hz"], out_dir, "dac1_ramp_time")

    print("\n=== Vref sweep ===")
    sweep = run_vref_sweep(CONFIG)
    plot_vref_sweep(sweep, save_path, f"vref_sweep_{mode}", CONFIG["cap_density"])
    for label, s in sweep.items():
        print(f"\n[{label}]")
        for row in zip(s["vref"], s["power_wc"], s["total_cap"], s["area_um2"], s["n_bits"], s["dnl_wc"], s["inl_wc"], s["dnl_3s"], s["inl_3s"], s["dnl_3s_an"], s["inl_3s_an"]):
            print(
                f"Vref={row[0]:.3f} V | P_wc={row[1]*1e6:7.3f} uW | C_DAC={row[2]*1e12:7.3f} pF | "
                f"Area={row[3]:7.0f} um^2 | bits={int(row[4]):2d} | DNL={row[5]:.3f} | INL={row[6]:.3f} | "
                f"3s DNL={row[7]:.3f} | 3s INL={row[8]:.3f} | 3s DNL an={row[9]:.3f} | 3s INL an={row[10]:.3f}"
            )

    print(f"\nAll plots saved to:\n- {save_path}")


if __name__ == "__main__":
    main()
