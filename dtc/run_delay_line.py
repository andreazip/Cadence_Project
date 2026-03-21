from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dtc.dtc_core import DelayLineDTC, compute_dnl_inl, configure_plot_style, save_figure_to


CONFIG = {
    "save_dir": Path(r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP"),
    "plot_folder": "plot_python",
    "delay_line_folder": "delay_line",
    "n_bits": 11,
    "vdd": 1.1,
    "vth": 0.55,
    # Apply directly to Vdd and Vth in runner (typ. 0.5 for delay line).
    "voltage_scale_factor_delay_line": 0.5,
    "ich": 225e-6,
    "cramp": 0.5e-15,
    "f_hz": 100e6,
    # Set sigma_cramp > 0 to model global Cramp mismatch.
    "sigma_cramp": 0.0,
    "mc_runs": 100,
}


def plot_delay_line(replica_axis, delay_s, power_w, out_dir):
    fig_delay, ax_delay = plt.subplots(figsize=(10, 6))
    ax_delay.plot(replica_axis, delay_s * 1e9, color="#D62728", linewidth=2.6, label="Delay")
    ax_delay.set_title("Delay Line Total Delay vs Replica Count", fontsize=14, fontweight="bold", pad=12)
    ax_delay.set_xlabel("Replica Count N", fontsize=12, fontweight="bold")
    ax_delay.set_ylabel("Delay [ns]", fontsize=12, fontweight="bold")
    ax_delay.grid(True, linestyle="--", alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax_delay.set_axisbelow(True)
    ax_delay.legend(fontsize=10, framealpha=0.96, edgecolor="black")
    save_figure_to(fig_delay, "delay_line_delay_vs_replicas.png", out_dir)
    plt.close(fig_delay)

    fig_power, ax_power = plt.subplots(figsize=(10, 6))
    ax_power.plot(replica_axis, power_w * 1e6, color="#1F77B4", linewidth=2.6, label="Power")
    ax_power.set_title("Delay Line Total Power vs Replica Count", fontsize=14, fontweight="bold", pad=12)
    ax_power.set_xlabel("Replica Count N", fontsize=12, fontweight="bold")
    ax_power.set_ylabel("Power [uW]", fontsize=12, fontweight="bold")
    ax_power.grid(True, linestyle="--", alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax_power.set_axisbelow(True)
    ax_power.legend(fontsize=10, framealpha=0.96, edgecolor="black")
    save_figure_to(fig_power, "delay_line_power_vs_replicas.png", out_dir)
    plt.close(fig_power)


def plot_delay_line_dnl_inl(replica_axis, delay_s, out_dir):
    dnl, inl, lsb = compute_dnl_inl(delay_s)
    if len(dnl) == 0:
        return

    x = replica_axis[1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    ax1.plot(x, dnl, color="#D62728", linewidth=2.2, label="DNL")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("DNL (LSB)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Delay Line DNL/INL (LSB = {lsb:.2e} s)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(True, linestyle="--", alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, framealpha=0.96, edgecolor="black")

    ax2.plot(x, inl, color="#1F77B4", linewidth=2.2, label="INL")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax2.set_ylabel("INL (LSB)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Replica Count N", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10, framealpha=0.96, edgecolor="black")

    plt.tight_layout()
    save_figure_to(fig, "delay_line_dnl_inl.png", out_dir)
    plt.close(fig)


def main():
    configure_plot_style()

    out_dir = CONFIG["save_dir"] / CONFIG["plot_folder"] / CONFIG["delay_line_folder"]

    scale_dl = float(CONFIG["voltage_scale_factor_delay_line"])
    vdd_eff = float(CONFIG["vdd"]) * scale_dl
    vth_eff = float(CONFIG["vth"]) * scale_dl

    sim = DelayLineDTC(
        n_bits=CONFIG["n_bits"],
        vdd=vdd_eff,
        vth=vth_eff,
        ich=CONFIG["ich"],
        cramp=CONFIG["cramp"],
        freq_hz=CONFIG["f_hz"],
        sigma_cramp=CONFIG["sigma_cramp"],
    )

    nominal = sim.evaluate_total(mismatch_enable=False)
    curve = sim.characterize_by_replica(mismatch_enable=False)

    print("Delay-Line DTC nominal summary")
    print("-" * 50)
    print(f"n_bits: {CONFIG['n_bits']}")
    print(f"N replicas: {int(nominal['N'])}")
    print(f"Delay @ N: {nominal['delay_s']*1e9:.6f} ns")
    print(f"Power @ N: {nominal['power_w']*1e6:.6f} uW")

    plot_delay_line(curve["N_axis"], curve["delay_s"], curve["power_w"], out_dir)
    plot_delay_line_dnl_inl(curve["N_axis"], curve["delay_s"], out_dir)

    if CONFIG["sigma_cramp"] > 0.0 and CONFIG["mc_runs"] > 0:
        delay_samples = np.zeros(int(CONFIG["mc_runs"]))
        power_samples = np.zeros(int(CONFIG["mc_runs"]))
        for i in range(int(CONFIG["mc_runs"])):
            r = sim.evaluate_total(mismatch_enable=True)
            delay_samples[i] = r["delay_s"]
            power_samples[i] = r["power_w"]

        print("\nCramp mismatch Monte Carlo (global scaling only)")
        print(f"sigma_cramp: {CONFIG['sigma_cramp']*1e15:.6f} fF")
        print(f"Delay mean/std: {np.mean(delay_samples)*1e9:.6f} / {np.std(delay_samples)*1e12:.6f} ns/ps")
        print(f"Power mean/std: {np.mean(power_samples)*1e6:.6f} / {np.std(power_samples)*1e6:.6f} uW")

    print(f"\nSaved delay-line plots to: {out_dir}")


if __name__ == "__main__":
    main()
