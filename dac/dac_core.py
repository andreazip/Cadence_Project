import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import apply_science_style, _multi_panel_figsize, maybe_title, maybe_suptitle


apply_science_style()


def compute_sigma_c(ac_nm, cu_f, cap_density_f_per_um2):
    """Pelgrom mismatch of one unit cap: sigma_C/C = Ac/sqrt(area), area = Cu / density."""
    area_um2 = cu_f / cap_density_f_per_um2
    return ac_nm / np.sqrt(area_um2) * cu_f


def build_cap_array(n_bits, cu, sigma_c, dac_mode, mismatch_enable=False):
    """Binary: n_bits weighted caps (each a sum of 2**j mismatched units).
    Thermometer: 2**n_bits - 1 identical units."""
    sigma_val = sigma_c if mismatch_enable else 0.0
    mode_key = dac_mode.lower()

    if mode_key == "binary":
        arr = np.zeros(n_bits)
        for j in range(n_bits):
            units = cu + np.random.randn(2**j) * sigma_val  # generate 2**j units
            arr[j] = np.sum(units)
        return arr

    if mode_key == "thermometer":
        n_units = (2**n_bits) - 1
        return cu + np.random.randn(n_units) * sigma_val

    raise ValueError("dac_mode must be 'binary' or 'thermometer'")


def switched_cap(codes, arr, dac_mode):
    """Capacitance connected to Vref for every code (vectorised)."""
    codes = np.asarray(codes, dtype=int)
    mode_key = dac_mode.lower()

    if mode_key == "binary":
        bits = (codes[:, None] >> np.arange(len(arr))[None, :]) & 1
        return bits @ arr

    if mode_key == "thermometer":
        cumulative = np.concatenate([[0.0], np.cumsum(arr)])
        return cumulative[codes]

    raise ValueError("dac_mode must be 'binary' or 'thermometer'")


class ChargeSharingDAC:
    """DAC 1: only the selected caps are connected (unselected ones float).

      V[k] = Vref * C[k] / (Cload + C[k])
      E[k] = Vref * V[k] * Cload

    The transfer is compressive, so the largest step is the first one:
      Cu = Cload * dv_max / (Vref - dv_max)
    The number of units is the smallest that reaches v_range.
    """

    def __init__(self, vref, cload, dv_max, v_range, dac_mode, ac_nm, cap_density):
        if vref <= v_range:
            raise ValueError(f"DAC1 needs Vref > v_range ({vref} <= {v_range})")
        self.vref = vref
        self.cload = cload
        self.dv_max = dv_max
        self.v_range = v_range
        self.dac_mode = dac_mode

        self.cu = cload * dv_max / (vref - dv_max)
        c_needed = cload * v_range / (vref - v_range)
        self.k_max = int(np.ceil(c_needed / self.cu - 1e-9))  # last code needed to cover the range
        self.n_bits = int(np.ceil(np.log2(self.k_max + 1)))
        self.codes = np.arange(self.k_max + 1)
        self.cap_density = cap_density
        self.sigma_c = compute_sigma_c(ac_nm, self.cu, cap_density)

    def build_dac_array(self, mismatch_enable=False):
        return build_cap_array(self.n_bits, self.cu, self.sigma_c, self.dac_mode, mismatch_enable)

    def total_cap(self):
        """Physical array capacitance (binary builds all 2**n - 1 units)."""
        return ((2**self.n_bits) - 1) * self.cu

    def total_area_um2(self):
        """Array area from the assumed cap density (Cload and routing excluded)."""
        return self.total_cap() / self.cap_density

    def voltage(self, arr):
        c_k = switched_cap(self.codes, arr, self.dac_mode)
        return self.vref * c_k / (self.cload + c_k)

    def energy(self, v_out):
        return self.vref * v_out * self.cload

    def ramp_charge(self, v_out):
        """Charge the source delivers per code: all of it ends on Cload (E = Vref * Q).
        Q = Cload * C[k] / (Cload + C[k]) * Vref"""
        return self.cload * v_out


class SignMagnitudeDAC:
    """DAC 2: bottom-plate switched CDAC, unselected caps at ground, MSB sets the sign.

      V[k]  = sign * Vref * C[k] / (Ctot + Cload)
      E[k]  = C[k] (Cload + Ctot - C[k]) / (Cload + Ctot) * Vref^2 + alpha * Cload * V[k]^2

    C[k] uses the magnitude code (no MSB). Energy is symmetric in the sign.
    Linear transfer, so: LSB = v_fs / (2**m - 1) <= dv_max  and  Cu = LSB * Cload / (Vref - v_fs).
    """

    def __init__(self, vref, cload, dv_max, v_fs, dac_mode, ac_nm, cap_density, cload_term_factor):
        if vref <= v_fs:
            raise ValueError(f"DAC2 needs Vref > v_fs ({vref} <= {v_fs})")
        self.vref = vref
        self.cload = cload
        self.dv_max = dv_max
        self.v_fs = v_fs
        self.dac_mode = dac_mode
        self.cload_term_factor = cload_term_factor

        self.m = int(np.ceil(np.log2(v_fs / dv_max + 1 - 1e-9)))  # magnitude bits
        self.n_bits = self.m + 1
        self.k_max = (2**self.m) - 1
        self.lsb = v_fs / self.k_max
        self.cu = self.lsb * cload / (vref - v_fs)
        self.mag_codes = np.arange(self.k_max + 1)
        # Signed codes -k_max..+k_max, the duplicated zero (MSB=1, mag=0) is removed.
        self.codes = np.arange(-self.k_max, self.k_max + 1)
        self.cap_density = cap_density
        self.sigma_c = compute_sigma_c(ac_nm, self.cu, cap_density)

    def build_dac_array(self, mismatch_enable=False):
        return build_cap_array(self.m, self.cu, self.sigma_c, self.dac_mode, mismatch_enable)

    def total_cap(self):
        return self.k_max * self.cu

    def total_area_um2(self):
        """Array area from the assumed cap density (Cload and routing excluded)."""
        return self.total_cap() / self.cap_density

    def _signed(self, values_mag, odd):
        neg = -values_mag[:0:-1] if odd else values_mag[:0:-1]
        return np.concatenate([neg, values_mag])

    def voltage(self, arr):
        c_k = switched_cap(self.mag_codes, arr, self.dac_mode)
        v_mag = self.vref * c_k / (np.sum(arr) + self.cload)
        return self._signed(v_mag, odd=True)

    def energy(self, arr):
        c_k = switched_cap(self.mag_codes, arr, self.dac_mode)
        c_sum = np.sum(arr) + self.cload
        v_mag = self.vref * c_k / c_sum
        e_mag = c_k * (c_sum - c_k) / c_sum * self.vref**2 + self.cload_term_factor * self.cload * v_mag**2
        return self._signed(e_mag, odd=False)


# ---------------------------
# Linearity
# ---------------------------
def linear_dnl_inl(v_out):
    """Endpoint DNL/INL in LSB (same definition as the DTC)."""
    steps = np.diff(v_out)
    lsb = (v_out[-1] - v_out[0]) / (len(v_out) - 1)
    dnl = steps / lsb - 1
    inl = np.cumsum(dnl)
    return dnl, inl


def nonlinear_dnl_inl(c_mc, c_ideal, vref, cload):
    """DNL/INL of a nonlinear DAC against its own ideal curve, in local LSB.

    Endpoint correction is done in the capacitor domain (scale the mismatched array so the
    last code matches), which is the equivalent of the endpoint fit for the linear DAC.
      DNL[k] = dV_mc[k] / dV_ideal[k] - 1
      INL[k] = (V_mc[k+1] - V_ideal[k+1]) / dV_ideal[k]
    """
    c_mc = c_mc * (c_ideal[-1] / c_mc[-1])
    v_mc = vref * c_mc / (cload + c_mc)
    v_id = vref * c_ideal / (cload + c_ideal)
    step_id = np.diff(v_id)
    dnl = np.diff(v_mc) / step_id - 1
    inl = (v_mc[1:] - v_id[1:]) / step_id
    return dnl, inl


def run_mc_linearity(dac, n_runs):
    """Returns (dnl_runs, inl_runs) as arrays of shape (n_runs, n_codes - 1)."""
    dnl_runs, inl_runs = [], []
    if isinstance(dac, ChargeSharingDAC):
        c_ideal = switched_cap(dac.codes, dac.build_dac_array(mismatch_enable=False), dac.dac_mode)
        for _ in range(n_runs):
            c_mc = switched_cap(dac.codes, dac.build_dac_array(mismatch_enable=True), dac.dac_mode)
            dnl, inl = nonlinear_dnl_inl(c_mc, c_ideal, dac.vref, dac.cload)
            dnl_runs.append(dnl)
            inl_runs.append(inl)
    else:
        for _ in range(n_runs):
            dnl, inl = linear_dnl_inl(dac.voltage(dac.build_dac_array(mismatch_enable=True)))
            dnl_runs.append(dnl)
            inl_runs.append(inl)
    return np.array(dnl_runs), np.array(inl_runs)


def mc_worst_case(runs, percentile):
    """Percentile (over MC runs) of the per-run max |error|."""
    return np.percentile(np.max(np.abs(runs), axis=1), percentile)


def analytic_n_sigma(dac, n_sigma=3):
    """Analytic n-sigma DNL/INL at the worst code [LSB], from independent unit-cap errors.

    K = last code used (endpoint fit), e(S) = sum of the unit errors in set S, var = sigma_u^2 per unit.
    Binary: code k switches k units and shares (k & K) units with the endpoint code.
    Thermometer: code k's units are always a subset of K's, so the overlap is k.

      INL[k] = e(S_k) - k/K e(S_K)
        var  = sigma_u^2 [ k - 2 k ov(k) / K + k^2 / K ]
      DNL[k] = e(S_k) - e(S_k-1) - e(S_K) / K
        var  = sigma_u^2 [ N_sw(k) - 2 (ov(k) - ov(k-1)) / K + 1 / K ]

    For DAC2 (K = 2**m - 1, all ones) INL reduces to k (K - k) / K and the worst DNL to the MSB carry.
    For DAC1 the errors are in local LSB (small-signal: dV/dV_ideal - 1 ~ dC / Cu).
    """
    su_rel = dac.sigma_c / dac.cu
    k_end = dac.k_max
    k = np.arange(k_end + 1)

    if dac.dac_mode.lower() == "binary":
        overlap = k & k_end
        n_sw = (k[1:] ^ k[:-1]).astype(float)  # value of flipped bits = units switching (on + off)
    else:
        overlap = k
        n_sw = np.ones(k_end)

    var_inl = k - 2 * k * overlap / k_end + k**2 / k_end
    var_dnl = n_sw - 2 * np.diff(overlap) / k_end + 1 / k_end
    return n_sigma * su_rel * np.sqrt(var_dnl.max()), n_sigma * su_rel * np.sqrt(var_inl.max())


def mc_n_sigma(runs, n_sigma=3):
    """n * sigma of the error at the worst code (sigma taken across MC runs, per code)."""
    return n_sigma * np.max(np.std(runs, axis=0, ddof=1))


# ---------------------------
# Plotting
# ---------------------------
def save_figure_to(fig, filename, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")

    # PNG copy in a png/ subfolder, for quick viewing and slides.
    png_dir = out_path / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{Path(filename).stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")


def _style_grid(ax):
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)


def plot_voltage_power(codes, v_out, power_w, out_dir, name_prefix, file_stem):
    fig_v = plt.figure()
    ax = fig_v.add_subplot(111)
    ax.plot(codes, v_out * 1e3, color=plt.cm.tab10(0), linewidth=2)
    maybe_title(ax, f'{name_prefix} Output Voltage vs Digital Code', fontweight='bold', pad=12)
    ax.set_xlabel(r'$\mathrm{Digital} \; \mathrm{Code}$', fontweight='bold')
    ax.set_ylabel(r'$V_\mathrm{out} \; [\mathrm{mV}]$', fontweight='bold')
    _style_grid(ax)
    save_figure_to(fig_v, f'{file_stem}_voltage.pdf', out_dir)
    plt.close(fig_v)

    fig_s = plt.figure()
    ax = fig_s.add_subplot(111)
    ax.plot(codes[1:], np.diff(v_out) * 1e3, color=plt.cm.tab10(2), linewidth=2)
    maybe_title(ax, f'{name_prefix} Step Size vs Digital Code', fontweight='bold', pad=12)
    ax.set_xlabel(r'$\mathrm{Digital} \; \mathrm{Code}$', fontweight='bold')
    ax.set_ylabel(r'$\Delta V_\mathrm{out} \; [\mathrm{mV}]$', fontweight='bold')
    ax.set_ylim(bottom=0)
    _style_grid(ax)
    save_figure_to(fig_s, f'{file_stem}_step.pdf', out_dir)
    plt.close(fig_s)

    fig_p = plt.figure()
    ax = fig_p.add_subplot(111)
    ax.plot(codes, power_w * 1e6, color=plt.cm.tab10(1), linewidth=2)
    maybe_title(ax, f'{name_prefix} Power vs Digital Code', fontweight='bold', pad=12)
    ax.set_xlabel(r'$\mathrm{Digital} \; \mathrm{Code}$', fontweight='bold')
    ax.set_ylabel(r'$\mathrm{Power} \; [\mu W]$', fontweight='bold')
    _style_grid(ax)
    save_figure_to(fig_p, f'{file_stem}_power.pdf', out_dir)
    plt.close(fig_p)


def plot_power_vs_vout(curves, out_dir, file_stem):
    """curves: list of dicts with 'label', 'v_out', 'power'."""
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, item in enumerate(curves):
        ax.plot(item["v_out"] * 1e3, item["power"] * 1e6, lw=2, color=plt.cm.tab10(i), label=item["label"])
    ax.set_xlabel(r'$V_\mathrm{out} \; [\mathrm{mV}]$')
    ax.set_ylabel(r'$\mathrm{Power} \; [\mu\mathrm{W}]$')
    _style_grid(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    save_figure_to(fig, f'{file_stem}_power_vs_vout.pdf', out_dir)
    plt.close(fig)


def plot_mc_linearity(codes, dnl_runs, inl_runs, out_dir, name_prefix, file_stem, lsb_label):
    n_runs = len(dnl_runs)
    x = codes[1:]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=_multi_panel_figsize(2, 1))

    for idx, (dnl, inl) in enumerate(zip(dnl_runs, inl_runs)):
        alpha_value = 0.15 + (idx / max(n_runs - 1, 1)) * 0.45
        ax1.plot(x, dnl, color=plt.cm.tab10(0), linewidth=1.0, alpha=alpha_value)
        ax2.plot(x, inl, color=plt.cm.tab10(1), linewidth=1.0, alpha=alpha_value)

    for ax, name in ((ax1, 'DNL'), (ax2, 'INL')):
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax.set_ylabel(rf'$\mathrm{{{name}}} \; [\mathrm{{{lsb_label}}}]$', fontweight='bold')
        _style_grid(ax)
    ax2.set_xlabel(r'$\mathrm{Digital} \; \mathrm{Code}$', fontweight='bold')
    maybe_suptitle(fig, f'{name_prefix} Monte Carlo DNL/INL ({n_runs} realizations)', fontweight='bold')

    fig.tight_layout()
    save_figure_to(fig, f'{file_stem}_dnl_inl_mc{n_runs}.pdf', out_dir)
    plt.close(fig)


def plot_vref_sweep(sweep, out_dir, file_stem, cap_density, power_floor_w=None):
    """sweep: dict label -> dict of arrays (vref, power_wc, total_cap, n_bits, dnl_3s_an, inl_3s_an).
    The linearity panel shows the analytic 3-sigma (no MC noise)."""
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 9))
    ax_p, ax_c, ax_b, ax_l = axes

    for i, (label, s) in enumerate(sweep.items()):
        color = plt.cm.tab10(i)
        ax_p.plot(s["vref"], s["power_wc"] * 1e6, lw=2, color=color, label=label)
        ax_c.plot(s["vref"], s["total_cap"] * 1e12, lw=2, color=color)
        ax_b.step(s["vref"], s["n_bits"], where='mid', lw=2, color=color)
        ax_l.plot(s["vref"], s["inl_3s_an"], lw=2, color=color, linestyle='-')
        ax_l.plot(s["vref"], s["dnl_3s_an"], lw=2, color=color, linestyle='--')

    if power_floor_w is not None:
        ax_p.axhline(power_floor_w * 1e6, color='black', lw=1.0, alpha=0.5, linestyle=':')

    ax_p.set_ylabel(r'$P_\mathrm{wc} \; [\mu\mathrm{W}]$')
    ax_c.set_ylabel(r'$C_\mathrm{DAC} \; [\mathrm{pF}]$')
    ax_c.set_yscale('log')
    # Same quantity in area units (A = C / density), so a secondary scale, not a second y-variable.
    pf_per_um2 = cap_density * 1e12
    ax_area = ax_c.secondary_yaxis('right', functions=(lambda c: c / pf_per_um2, lambda a: a * pf_per_um2))
    ax_area.set_ylabel(r'$\mathrm{Area} \; [\mu\mathrm{m}^2]$')
    ax_b.set_ylabel(r'$\mathrm{Bits}$')
    ax_l.set_ylabel(r'$3\sigma \; \mathrm{INL}, \mathrm{DNL} \; [\mathrm{LSB}]$')
    ax_l.set_xlabel(r'$V_\mathrm{ref} \; [\mathrm{V}]$')
    ax_p.legend(frameon=False, fontsize=10)
    ax_l.plot([], [], color='gray', lw=2, linestyle='-', label='INL')
    ax_l.plot([], [], color='gray', lw=2, linestyle='--', label='DNL')
    ax_l.legend(frameon=False, fontsize=10)
    for ax in axes:
        _style_grid(ax)

    fig.tight_layout()
    save_figure_to(fig, f'{file_stem}.pdf', out_dir)
    plt.close(fig)


def plot_ramp_time_vs_code(codes, q_code, ramp_times, f_hz, out_dir, file_stem):
    """DAC1 ramp time per code, t[k] = Q[k] / I with I = Q[k_max] / t_ramp (sized on full scale)."""
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = plt.cm.Blues(np.linspace(0.45, 1.0, len(ramp_times)))  # sequential: longer ramp = darker

    for color, t_ramp in zip(colors, ramp_times):
        i_ramp = q_code[-1] / t_ramp
        ax.plot(codes[1:], q_code[1:] / i_ramp * 1e9, lw=2, color=color,
                label=rf'$t_\mathrm{{ramp,FS}} = {t_ramp*1e9:g}\,\mathrm{{ns}}$')

    ax.axhline(1 / f_hz * 1e9, color='black', lw=1.0, alpha=0.6, linestyle=':')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathrm{Digital} \; \mathrm{Code}$')
    ax.set_ylabel(r'$t_\mathrm{ramp} \; [\mathrm{ns}]$')
    _style_grid(ax)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    save_figure_to(fig, f'{file_stem}.pdf', out_dir)
    plt.close(fig)
