#!/usr/bin/env python3
"""
DNL / INL of a binary-weighted capacitor bank from a Cadence MC CSV export,
checked against the analytic mismatch model.

Input format: one header row + one data row. Each (code, mcparamset) pair
contributes three columns: X (frequency), YRe, YReImag (real / imaginary
part of the probed current).

C is extracted as  C = Im(I) / (2*pi*f*V), with V = VAC (unit AC magnitude
on the stimulus by default).

DNL is computed only at consecutive-code transitions present in the sweep,
with a per-run endpoint LSB  C_unit = (C_max - C_min) / (NUM_CODES - 1).
INL is the running sum of those DNL values (gaps contribute nothing).

Analytic model: sigma_u is extracted from the 0->1 step (one unit cap, C0
cancels in the difference). The predicted DNL sigma at transition k-1 -> k is

    sigma_DNL(k) = (sigma_u / Cu) * sqrt(N_switched(k))

where N_switched is the number of unit caps that change state (units switching
on plus units switching off, from the XOR of the two codes). At a carry into
2^m this reduces to sqrt(2^(m+1) - 1); at a plain LSB step it is 1.

Usage (from the project root):
    python -m dac.cap_bank_mc path/to/current_mc_sim_cap.csv
"""

import csv
import re
import sys
from math import erf, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dac.dac_core import _style_grid, save_figure_to
from plot_style import apply_science_style, maybe_title


CSV_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "current_mc_sim_cap.csv")
VAC = 1.0        # AC stimulus amplitude [V]
NUM_CODES = 512  # full-scale code count, for the ideal LSB
# Python plots live outside results/ (that one is for Cadence exports).
SAVE_DIR = Path(__file__).resolve().parents[1] / "plots_python" / "cap_bank_mc"

HDR_RE = re.compile(r"code=(\d+),\s*mcparamset=(\d+)\)\s+(\S+)")


# ---------------------------
# Data
# ---------------------------
def load(path):
    """Return freq, sorted codes, sorted mc ids, and C[mc, code] in farad."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        row = [float(v) for v in next(reader)]

    freq = {}
    imag = {}
    for col, name in enumerate(header):
        m = HDR_RE.search(name)
        if not m:
            continue
        key = (int(m.group(2)), int(m.group(1)))  # (mc, code)
        if m.group(3) == "X":
            freq[key] = row[col]
        elif m.group(3) == "YReImag":
            imag[key] = row[col]

    codes = sorted({c for _, c in imag})
    mcs = sorted({m for m, _ in imag})

    if len(set(freq.values())) != 1:
        print(f"warning: {len(set(freq.values()))} distinct frequencies in file", file=sys.stderr)
    f0 = freq[next(iter(freq))]

    C = np.full((len(mcs), len(codes)), np.nan)
    for i, mc in enumerate(mcs):
        for j, code in enumerate(codes):
            C[i, j] = imag[(mc, code)] / (2 * np.pi * f0 * VAC)
    return f0, np.array(codes), np.array(mcs), C


def n_switched(k):
    """Unit caps changing state on the transition k-1 -> k.

    Binary weights: the integer value of a bit mask equals its number of unit caps.
    """
    a, b = k - 1, k
    x = a ^ b
    return (b & x) + (a & x)


def dnl_inl(codes, c_row):
    """DNL at consecutive-code transitions only; INL = running sum of DNL.

    Returns (dnl, inl, valid, c_unit). Non-consecutive gaps are NaN in dnl
    and contribute nothing to the accumulated inl.
    """
    c_unit = (c_row.max() - c_row.min()) / (NUM_CODES - 1)

    dnl = np.full(len(codes), np.nan)
    valid = np.zeros(len(codes), dtype=bool)
    for k in range(1, len(codes)):
        if codes[k] - codes[k - 1] == 1:
            dnl[k] = (c_row[k] - c_row[k - 1]) / c_unit - 1.0
            valid[k] = True

    inl = np.nancumsum(np.where(valid, dnl, 0.0))
    return dnl, inl, valid, c_unit


# ---------------------------
# Reports
# ---------------------------
def report_dnl_inl(codes, C, dnl_all, inl_all, c_units):
    print(f"\nC_unit    : {c_units.mean():.4e} F  "
          f"(sigma {c_units.std(ddof=1):.3e}, "
          f"{100*c_units.std(ddof=1)/c_units.mean():.3f} %)")
    print(f"C_full    : {C[:, -1].mean():.4e} F")

    print("\nper-run extremes over the sampled transitions [LSB]")
    print(f"  max DNL : {np.nanmax(dnl_all):+.4f}   "
          f"mean of per-run max {np.nanmax(dnl_all, axis=1).mean():+.4f}  "
          f"sigma {np.nanmax(dnl_all, axis=1).std(ddof=1):.4f}")
    print(f"  min DNL : {np.nanmin(dnl_all):+.4f}   "
          f"mean of per-run min {np.nanmin(dnl_all, axis=1).mean():+.4f}  "
          f"sigma {np.nanmin(dnl_all, axis=1).std(ddof=1):.4f}")
    print(f"  max INL : {np.nanmax(inl_all):+.4f}")
    print(f"  min INL : {np.nanmin(inl_all):+.4f}")


def analytic_model(codes, C, dnl_all, c_units):
    """Compare measured DNL sigma per transition with sigma_u/Cu * sqrt(N_sw).

    Returns rows (code_from, code_to, n_sw, measured, predicted) and sigma_u/Cu.
    """
    idx0 = np.where(codes == 0)[0]
    idx1 = np.where(codes == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Codes 0 and 1 are needed to extract sigma_u from the 0->1 step.")

    cu = c_units.mean()
    c0 = C[:, idx0[0]]
    step01 = C[:, idx1[0]] - c0  # C0 cancels in the difference
    sigma_u = step01.std(ddof=1)
    su_rel = sigma_u / cu

    print(f"\nC0 (code 0)        : {c0.mean():.4e} F  "
          f"sigma {c0.std(ddof=1):.3e} ({100*c0.std(ddof=1)/c0.mean():.4f} %)")
    print(f"Cu                 : {cu:.4e} F")
    print(f"sigma_u            : {sigma_u:.4e} F")
    print(f"sigma_u / Cu       : {su_rel:.5f}  ({100*su_rel:.3f} %)")
    print(f"  [from C(1) alone : {C[:, idx1[0]].std(ddof=1)/cu:.5f} "
          f"- inflated by C0 variation, do not use]")

    print(f"\n{'transition':>12} {'Nsw':>5} {'mean':>10} {'measured':>10} "
          f"{'model':>10} {'meas/model':>11} {'min':>10} {'max':>10}")
    rows = []
    for k in range(1, len(codes)):
        col = dnl_all[:, k]
        if np.all(np.isnan(col)):
            continue
        code_to = int(codes[k])
        nsw = n_switched(code_to)
        meas = np.nanstd(col, ddof=1)
        pred = su_rel * np.sqrt(nsw)
        rows.append((int(codes[k - 1]), code_to, nsw, meas, pred))
        print(f"{f'{codes[k-1]}->{code_to}':>12} {nsw:5d} {np.nanmean(col):+10.4f} {meas:10.5f} "
              f"{pred:10.5f} {meas/pred:11.3f} {np.nanmin(col):+10.4f} {np.nanmax(col):+10.4f}")

    worst = max(rows, key=lambda r: r[3])
    print(f"\nworst transition {worst[0]}->{worst[1]} (Nsw = {worst[2]}): "
          f"measured {worst[3]:.4f} vs model {worst[4]:.4f} LSB "
          f"({worst[3]/worst[4]:.2f}x)")
    p = 2 * (1 - 0.5 * (1 + erf(0.5 / (worst[3] * sqrt(2)))))
    print(f"P(|DNL| > 0.5 LSB) at worst transition : {100*p:.2f} %"
          f"   (0.5 LSB = {0.5/worst[3]:.2f} sigma)")
    return rows, su_rel


# ---------------------------
# Plots
# ---------------------------
def transition_label(code_from, code_to):
    return rf"{code_from}$\rightarrow${code_to}"


def plot_dnl_inl(codes, dnl_all, inl_all, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(codes))

    for i in range(dnl_all.shape[0]):
        ax1.plot(x, dnl_all[i], color=plt.cm.tab10(0), alpha=0.18, lw=0.9, marker="o", ms=2.5)
        ax2.plot(x, inl_all[i], color=plt.cm.tab10(1), alpha=0.18, lw=0.9, marker="o", ms=2.5)

    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dnl_mean = np.nanmean(dnl_all, axis=0)
            inl_mean = np.nanmean(inl_all, axis=0)
    ax1.plot(x, dnl_mean, color="k", lw=1.6, label="mean")
    ax2.plot(x, inl_mean, color="k", lw=1.6, label="mean")

    ax1.set_ylabel(r"$\mathrm{DNL} \; [\mathrm{LSB}]$")
    ax2.set_ylabel(r"$\mathrm{INL} \; [\mathrm{LSB}]$")
    ax2.set_xlabel(r"$\mathrm{Digital} \; \mathrm{Code}$")
    for ax in (ax1, ax2):
        ax.axhline(0, color="0.6", lw=0.7)
        _style_grid(ax)
        ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax2.set_xticks(x)
    ax2.set_xticklabels(codes, rotation=90, fontsize=7)

    maybe_title(ax1, f"Cap-bank DNL / INL, {dnl_all.shape[0]} MC runs (gaps = non-consecutive codes)")
    fig.tight_layout()
    save_figure_to(fig, "dnl_inl.pdf", out_dir)
    plt.close(fig)


def plot_analytic_vs_mc(rows, su_rel, out_dir):
    nsw = np.array([r[2] for r in rows], float)
    meas = np.array([r[3] for r in rows])
    pred = np.array([r[4] for r in rows])
    lbl = [transition_label(r[0], r[1]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    nn = np.logspace(0, np.log10(nsw.max() * 1.3), 100)
    ax1.plot(nn, su_rel * np.sqrt(nn), "k--", lw=1.4, label=r"model $\sigma_u/C_u\cdot\sqrt{N_{sw}}$")
    ax1.plot(nsw, meas, "o", color=plt.cm.tab10(3), ms=7, label="MC measured")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$N_{sw}$ (unit caps switching)")
    ax1.set_ylabel(r"$\sigma_{DNL}$ [LSB]")
    _style_grid(ax1)
    ax1.legend(fontsize=10, frameon=False)
    maybe_title(ax1, "scaling check")

    x = np.arange(len(rows))
    w = 0.38
    ax2.bar(x - w/2, meas, w, color=plt.cm.tab10(3), label="MC measured")
    ax2.bar(x + w/2, pred, w, color="0.6", label="analytic model")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lbl, rotation=90, fontsize=7)
    ax2.set_ylabel(r"$\sigma_{DNL}$ [LSB]")
    _style_grid(ax2)
    ax2.legend(fontsize=10, frameon=False)
    maybe_title(ax2, "per transition")

    fig.tight_layout()
    save_figure_to(fig, "analytic_vs_mc.pdf", out_dir)
    plt.close(fig)


def main():
    apply_science_style()
    f0, codes, mcs, C = load(CSV_PATH)
    out_dir = SAVE_DIR / CSV_PATH.stem

    print(f"file      : {CSV_PATH}")
    print(f"frequency : {f0/1e6:g} MHz")
    print(f"codes     : {len(codes)}  {[int(c) for c in codes]}")
    print(f"MC runs   : {len(mcs)}")

    n_mc = len(mcs)
    dnl_all = np.full((n_mc, len(codes)), np.nan)
    inl_all = np.full((n_mc, len(codes)), np.nan)
    c_units = np.zeros(n_mc)
    for i in range(n_mc):
        dnl_all[i], inl_all[i], _, c_units[i] = dnl_inl(codes, C[i])

    report_dnl_inl(codes, C, dnl_all, inl_all, c_units)
    rows, su_rel = analytic_model(codes, C, dnl_all, c_units)

    plot_dnl_inl(codes, dnl_all, inl_all, out_dir)
    plot_analytic_vs_mc(rows, su_rel, out_dir)
    print(f"\nAll plots saved to:\n- {out_dir}")


if __name__ == "__main__":
    main()
