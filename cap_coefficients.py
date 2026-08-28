import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_science_style


apply_science_style()
if shutil.which("latex") is None:
    plt.rcParams["text.usetex"] = False


def fit_capacitance_series(voltage, capacitance, degree=2):
    """Fit C(Vds) and return polynomial and normalized model coefficients."""
    valid = np.isfinite(voltage) & np.isfinite(capacitance)
    voltage = np.asarray(voltage[valid], dtype=float)
    capacitance = np.asarray(capacitance[valid], dtype=float)
    order = np.argsort(voltage)
    voltage = voltage[order]
    capacitance = capacitance[order]

    if voltage.size < 2:
        raise ValueError("Each X/Y pair needs at least two valid samples")
    fit_degree = min(degree, voltage.size - 1)
    polynomial = np.polyfit(voltage, capacitance, fit_degree)
    if fit_degree != 2:
        raise ValueError("At least three samples are required for the quadratic model")

    p2, p1, c0 = polynomial
    if c0 == 0:
        raise ValueError("The fitted C0 is zero; normalized coefficients are undefined")
    return voltage, capacitance, polynomial, (c0, p1 / c0, p2 / c0)


def fit_current_series(voltage, current):
    """Fit I(Vds) = I0 + I1*Vds and return its intercept and slope."""
    valid = np.isfinite(voltage) & np.isfinite(current)
    voltage = np.asarray(voltage[valid], dtype=float)
    current = np.asarray(current[valid], dtype=float)
    order = np.argsort(voltage)
    voltage = voltage[order]
    current = current[order]
    if voltage.size < 2 or np.unique(voltage).size < 2:
        raise ValueError("Each current X/Y pair needs at least two distinct samples")
    i1, i0 = np.polyfit(voltage, current, 1)
    return voltage, current, i0, i1


def main():
    parser = argparse.ArgumentParser(description="Fit and extrapolate Cadence C(Vds) sweeps")
    parser.add_argument("--input", type=Path, default=Path("results_cadence/cap_data.csv"))
    parser.add_argument(
        "--current-input",
        type=Path,
        default=Path("results_cadence/current_source_pre.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results_cadence/cap_extrapolated.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results_cadence/cap_extrapolated.png"))
    parser.add_argument(
        "--delay-plot",
        type=Path,
        default=Path("results_cadence/capacitance_delay.png"),
    )
    parser.add_argument(
        "--ideal-delay-plot",
        type=Path,
        default=Path("results_cadence/ideal_delay.png"),
    )
    parser.add_argument(
        "--inl-plot",
        type=Path,
        default=Path("results_cadence/capacitance_inl.png"),
    )
    parser.add_argument("--current-plot", type=Path, default=Path("results_cadence/current_extrapolated.png"))
    parser.add_argument("--current-delay-plot", type=Path, default=Path("results_cadence/current_delay.png"))
    parser.add_argument("--current-ideal-delay-plot", type=Path, default=Path("results_cadence/current_ideal_delay.png"))
    parser.add_argument("--current-inl-plot", type=Path, default=Path("results_cadence/current_inl.png"))
    parser.add_argument("--vds-min", type=float, default=None)
    parser.add_argument("--vds-max", type=float, default=None)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument(
        "--i0",
        type=float,
        default=None,
        help="Charging current I0 in A; if supplied, print evaluated delay and INL formulas",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    series = []
    for x_column in df.columns:
        if not x_column.endswith(" X"):
            continue
        y_column = x_column[:-2] + " Y"
        if y_column not in df.columns:
            raise ValueError(f"Missing capacitance column for {x_column}: {y_column}")
        voltage = pd.to_numeric(df[x_column], errors="coerce").to_numpy()
        capacitance = pd.to_numeric(df[y_column], errors="coerce").to_numpy()
        fit = fit_capacitance_series(voltage, capacitance)
        series.append((x_column[:-2], *fit))

    if not series:
        raise ValueError("No matching '<name> X' / '<name> Y' column pairs found")

    current_df = pd.read_csv(args.current_input)
    current_series = []
    for x_column in current_df.columns:
        if not x_column.endswith(" X"):
            continue
        y_column = x_column[:-2] + " Y"
        if y_column not in current_df.columns:
            raise ValueError(f"Missing current column for {x_column}: {y_column}")
        voltage = pd.to_numeric(current_df[x_column], errors="coerce").to_numpy()
        current = pd.to_numeric(current_df[y_column], errors="coerce").to_numpy()
        current_series.append((x_column[:-2], *fit_current_series(voltage, current)))
    if not current_series:
        raise ValueError("No matching current '<name> X' / '<name> Y' pairs found")

    measured_min = min(values[1].min() for values in series)
    measured_max = max(values[1].max() for values in series)
    vds_min = measured_min if args.vds_min is None else args.vds_min
    vds_max = measured_max if args.vds_max is None else args.vds_max
    if vds_min >= vds_max or args.points < 2:
        raise ValueError("Require vds-min < vds-max and at least two output points")

    extrapolated_vds = np.linspace(vds_min, vds_max, args.points)
    output = {"Vds": extrapolated_vds}
    fig, ax = plt.subplots()
    delay_fig, delay_ax = plt.subplots()
    ideal_delay_fig, ideal_delay_ax = plt.subplots()
    inl_fig, inl_ax = plt.subplots()
    current_fig, current_ax = plt.subplots()
    current_delay_fig, current_delay_ax = plt.subplots()
    current_ideal_fig, current_ideal_ax = plt.subplots()
    current_inl_fig, current_inl_ax = plt.subplots()
    delay_scale = 1 if args.i0 is None else 1 / args.i0
    delay_unit = "I0 * delay (F V)" if args.i0 is None else "Delay (s)"
    inl_unit = "I0 * INL (F V)" if args.i0 is None else "INL (s)"
    for name, voltage, capacitance, polynomial, normalized in series:
        c0, vc1, vc2 = normalized
        p2, p1, _ = polynomial
        td_max = (
            c0 * (vds_max - vds_min)
            + p1 / 2 * (vds_max**2 - vds_min**2)
            + p2 / 3 * (vds_max**3 - vds_min**3)
        )
        print(f"{name}: C0={c0:.6e} F, VC1={vc1:.6e} V^-1, VC2={vc2:.6e} V^-2")
        print(
            "  Eq. 3: td(Vds)|C(V)/I = "
            f"({c0:.6e}/I0)(Vds - {vds_min:g}) "
            f"+ ({p1:.6e}/(2 I0))(Vds^2 - {vds_min:g}^2) "
            f"+ ({p2:.6e}/(3 I0))(Vds^3 - {vds_min:g}^3)"
        )
        print(
            "  Eq. 4: td(Vds)|C/I = "
            f"({td_max:.6e}/I0)/({vds_max:g} - {vds_min:g}) "
            f"(Vds - {vds_min:g})"
        )
        print(
            "  Eq. 5: INL(Vds)|C(V)/I = "
            f"(Vds - {vds_min:g})(Vds - {vds_max:g}) "
            f"[{p1:.6e}/(2 I0) + {p2:.6e}/(3 I0) "
            f"(Vds + {vds_min:g} + {vds_max:g})]"
        )
        if args.i0 is not None:
            if args.i0 == 0:
                raise ValueError("I0 must be nonzero")
            print(f"  With I0={args.i0:.6e} A: td,max={td_max / args.i0:.6e} s")
        output[name] = np.polyval(polynomial, extrapolated_vds)
        ax.scatter(voltage, capacitance, label=f"{name} data")
        ax.plot(extrapolated_vds, output[name], "--", label=f"{name} quadratic")

        variable_delay = (
            c0 * (extrapolated_vds - vds_min)
            + p1 / 2 * (extrapolated_vds**2 - vds_min**2)
            + p2 / 3 * (extrapolated_vds**3 - vds_min**3)
        ) * delay_scale
        ideal_delay = (
            td_max / (vds_max - vds_min) * (extrapolated_vds - vds_min)
        ) * delay_scale
        inl = (
            (extrapolated_vds - vds_min)
            * (extrapolated_vds - vds_max)
            * (p1 / 2 + p2 / 3 * (extrapolated_vds + vds_min + vds_max))
            * delay_scale
        )
        delay_ax.plot(extrapolated_vds, variable_delay, label=f"{name} Eq. 3")
        ideal_delay_ax.plot(extrapolated_vds, ideal_delay, label=f"{name} Eq. 4")
        inl_ax.plot(extrapolated_vds, inl, label=f"{name} Eq. 5")

    current_output = {"Vds": extrapolated_vds}
    for current_name, current_voltage, current, i0, i1 in current_series:
        current_values = i0 + i1 * extrapolated_vds
        if np.any(current_values <= 0) or i1 == 0:
            raise ValueError(f"Current model for {current_name} must be positive and have nonzero I1")
        current_output[current_name] = current_values
        current_ax.scatter(current_voltage, current, label=f"{current_name} data")
        current_ax.plot(extrapolated_vds, current_values, "--", label=f"{current_name} Eq. 6")
        print(f"{current_name}: Eq. 6 I(Vds) = {i0:.6e} + ({i1:.6e}) Vds A")

        for cap_name, _, _, _, normalized in series:
            c0 = normalized[0]
            delay = c0 / i1 * np.log(current_values / (i0 + i1 * vds_min))
            delay_max = c0 / i1 * np.log(current_values[-1] / current_values[0])
            ideal_delay = delay_max / (vds_max - vds_min) * (extrapolated_vds - vds_min)
            current_inl = delay - ideal_delay
            label = f"{cap_name}, {current_name}"
            current_delay_ax.plot(extrapolated_vds, delay, label=label)
            current_ideal_ax.plot(extrapolated_vds, ideal_delay, label=label)
            current_inl_ax.plot(extrapolated_vds, current_inl, label=label)
            print(
                f"  {label}: Eq. 7 td = ({c0:.6e}/{i1:.6e}) ln(("
                f"{i0:.6e} + {i1:.6e} Vds)/({i0:.6e} + {i1:.6e} * {vds_min:g}))"
            )
            print(
                f"  {label}: ideal delay = ({delay_max:.6e}/({vds_max:g} - {vds_min:g})) "
                f"(Vds - {vds_min:g})"
            )
            print(f"  {label}: Eq. 8 INL = Eq. 7 delay - ideal delay")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.output, index=False)
    ax.set_xlabel("Vds (V)")
    ax.set_ylabel("Capacitance (F)")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=300, bbox_inches="tight")
    delay_ax.set_xlabel("Vds (V)")
    delay_ax.set_ylabel(delay_unit)
    delay_ax.set_title("Variable-capacitance delay (Eq. 3)")
    delay_ax.legend(fontsize=6, loc="best")
    delay_fig.tight_layout()
    args.delay_plot.parent.mkdir(parents=True, exist_ok=True)
    delay_fig.savefig(args.delay_plot, dpi=300, bbox_inches="tight")
    ideal_delay_ax.set_xlabel("Vds (V)")
    ideal_delay_ax.set_ylabel(delay_unit)
    ideal_delay_ax.set_title("Ideal linear delay (Eq. 4)")
    ideal_delay_ax.legend(fontsize=6, loc="best")
    ideal_delay_fig.tight_layout()
    args.ideal_delay_plot.parent.mkdir(parents=True, exist_ok=True)
    ideal_delay_fig.savefig(args.ideal_delay_plot, dpi=300, bbox_inches="tight")
    inl_ax.set_xlabel("Vds (V)")
    inl_ax.set_ylabel(inl_unit)
    inl_ax.set_title("Integral nonlinearity (Eq. 5)")
    inl_ax.legend(fontsize=6, loc="best")
    inl_fig.tight_layout()
    args.inl_plot.parent.mkdir(parents=True, exist_ok=True)
    inl_fig.savefig(args.inl_plot, dpi=300, bbox_inches="tight")
    current_ax.set_xlabel("Vds (V)")
    current_ax.set_ylabel("Current (A)")
    current_ax.set_title("Extrapolated current (Eq. 6)")
    current_ax.legend(fontsize=5, loc="best")
    current_fig.tight_layout()
    args.current_plot.parent.mkdir(parents=True, exist_ok=True)
    current_fig.savefig(args.current_plot, dpi=300, bbox_inches="tight")
    for plot_ax, plot_fig, path, ylabel, title in [
        (current_delay_ax, current_delay_fig, args.current_delay_plot, "Delay (s)", "Current-dependent delay (Eq. 7)"),
        (current_ideal_ax, current_ideal_fig, args.current_ideal_delay_plot, "Delay (s)", "Ideal linear delay"),
        (current_inl_ax, current_inl_fig, args.current_inl_plot, "INL (s)", "Current-dependent INL (Eq. 8)"),
    ]:
        plot_ax.set_xlabel("Vds (V)")
        plot_ax.set_ylabel(ylabel)
        plot_ax.set_title(title)
        plot_ax.legend(fontsize=5, loc="best")
        plot_fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Wrote {args.output} ({vds_min:g} to {vds_max:g} Vds)")
    print(f"Wrote {args.plot}")
    print(f"Wrote {args.delay_plot}")
    print(f"Wrote {args.ideal_delay_plot}")
    print(f"Wrote {args.inl_plot}")
    print(f"Wrote {args.current_plot}")
    print(f"Wrote {args.current_delay_plot}")
    print(f"Wrote {args.current_ideal_delay_plot}")
    print(f"Wrote {args.current_inl_plot}")


if __name__ == "__main__":
    main()

