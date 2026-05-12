import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import apply_science_style, _multi_panel_figsize


apply_science_style()


def compute_sigma_c(ac_nm, area_um2, cu_f):
    """Compute capacitor mismatch sigma from Pelgrom-style parameters."""
    return ac_nm / np.sqrt(area_um2) * cu_f


class ConstantSlopeDTC:
    """Constant-slope DTC model with pluggable DAC coding and non-ideal effects."""

    def __init__(
        self,
        n_bits,
        cu,
        dac_mode,
        thermo_bits,
        run_flags,
        sigma_c,
        vdd,
        vth,
        ich,
        cramp,
        i1,
        c1,
        c2,
    ):
        self.n_bits = n_bits
        self.m = n_bits - 1
        self.N = 2**n_bits - 1
        self.half = (self.N - 1) // 2
        self.cu = cu
        self.dac_mode = dac_mode
        self.thermo_bits = thermo_bits
        self.run_flags = run_flags
        self.sigma_c = sigma_c
        self.vdd = vdd
        self.vth = vth
        self.ich = ich
        self.cramp = cramp
        self.i1 = i1
        self.c1 = c1
        self.c2 = c2

    def build_dac_array(self, mismatch_enable=False, sigma=None):
        sigma_val = self.sigma_c if sigma is None else sigma
        mode_key = self.dac_mode.lower()

        if mode_key == "binary":
            arr = np.zeros(self.m)
            for j in range(self.m):
                mismatch = np.random.randn() * sigma_val / np.sqrt(2**j) if mismatch_enable else 0.0
                arr[j] = (2**j) * (self.cu + mismatch)
            return arr

        if mode_key == "thermometer":
            n_units = (2**self.m) - 1
            arr = np.zeros(n_units)
            for j in range(n_units):
                mismatch = np.random.randn() * sigma_val if mismatch_enable else 0.0
                arr[j] = self.cu + mismatch
            return arr

        if mode_key == "segmented":
            t = int(self.thermo_bits)
            b = self.m - t

            arr_bin = np.zeros(b)
            for j in range(b):
                mismatch = np.random.randn() * sigma_val / np.sqrt(2**j) if mismatch_enable else 0.0
                arr_bin[j] = (2**j) * (self.cu + mismatch)

            n_units = (2**t) - 1
            unit_val_ideal = (2**b) * self.cu
            arr_therm = np.zeros(n_units)
            for j in range(n_units):
                mismatch = np.random.randn() * sigma_val * np.sqrt(2**b) if mismatch_enable else 0.0
                arr_therm[j] = unit_val_ideal + mismatch

            return np.concatenate([arr_bin, arr_therm])

        raise ValueError("dac_mode must be 'binary', 'thermometer', or 'segmented'")

    def calc_ck(self, code, arr):
        lsb_code = code & ((2**self.m) - 1)
        mode_key = self.dac_mode.lower()

        if mode_key == "binary":
            c_k_local = 0.0
            for j in range(self.m):
                bit = (lsb_code >> j) & 1
                c_k_local += (1 - bit) * arr[j]
            return c_k_local

        if mode_key == "thermometer":
            turned_on = int(min(max(lsb_code, 0), len(arr)))
            return float(np.sum(arr[turned_on:]))

        if mode_key == "segmented":
            t = int(self.thermo_bits)
            b = self.m - t

            binary_code = lsb_code & ((2**b) - 1)
            thermo_index = lsb_code >> b
            arr_bin = arr[:b]
            arr_therm = arr[b:]

            c_k_bin = 0.0
            for j in range(b):
                bit = (binary_code >> j) & 1
                c_k_bin += (1 - bit) * arr_bin[j]

            thermo_on = np.sum(arr_therm[:thermo_index])
            total_thermo = np.sum(arr_therm)
            c_k_therm = total_thermo - thermo_on
            return float(c_k_bin + c_k_therm)

        raise ValueError("dac_mode must be 'binary', 'thermometer', or 'segmented'")

    def energy_msb_0(self, c_k, c0, ca, Vst):
        return (c0 + self.cramp + (ca - c_k)) * (c_k / (c0 + self.cramp + ca)) * self.vdd**2  + self.cramp * (self.vdd**2-(self.vdd)*self.vdd + (self.vdd/2)**2)

    def energy_msb_1(self, ca, c_k, c0, Vst):
        return (ca - c_k) / (c0 + ca) * (c0 + c_k) * self.vdd**2 + self.cramp * (Vst**2-Vst*(self.vdd) + (self.vdd/2)**2)

    def compute_vst_energy(self, cap_array, c0, cramp):
        ca = np.sum(cap_array)
        vst_array = np.zeros(self.N)
        energy_array = np.zeros(self.N)

        for i in range(self.N):
            c_k = self.calc_ck(i, cap_array)
            if i > self.half:
                vst_array[i] = (1 + (ca - c_k ) / (c0 + cramp + ca)) * self.vdd
                energy_array[i] = self.energy_msb_1(ca, c_k, c0, vst_array[i])
            else:
                vst_array[i] = (1 - (c_k) / (c0 + cramp + ca)) * self.vdd
                energy_array[i] = self.energy_msb_0(c_k, c0, ca, vst_array[i])

        return vst_array, energy_array, ca

    def compute_delay(self, vst_array, clm_enabled=None, nonlin_enabled=None, ich_val=None, cramp_val=None):
        if clm_enabled is None:
            clm_enabled = self.run_flags["CLM"]
        if nonlin_enabled is None:
            nonlin_enabled = self.run_flags["Non-linearities-capacitor"]

        ich_base = self.ich if ich_val is None else ich_val
        cramp_base = self.cramp if cramp_val is None else cramp_val

        delay = np.zeros(len(vst_array))
        ich_array = np.zeros(len(vst_array))
        cramp_array = np.zeros(len(vst_array))

        for i, vst in enumerate(vst_array):
            ich_eff = ich_base
            cramp_eff = cramp_base

            if clm_enabled:
                ich_eff = ich_base * (1 + self.i1 * (vst - 0.4))
                ich_array[i] = ich_eff

            if nonlin_enabled:
                cramp_eff = cramp_base * (1 + self.c1 * vst + self.c2 * vst**2)
                cramp_array[i] = cramp_eff

            k_eff = -ich_eff / cramp_eff
            delay[i] = (self.vth - vst) / k_eff

        return delay, ich_array, cramp_array


class VariableSlopeDTC:
    """Variable-slope DTC model where DAC controls ramp capacitance (Cramp)."""

    def __init__(
        self,
        n_bits,
        dac_mode,
        thermo_bits,
        run_flags,
        sigma_c,
        vdd,
        vth,
        ich,
        cramp_u,
        i1,
        c1,
        c2,
        C_fixed,
        self_power_down=True,
    ):
        self.n_bits = n_bits
        self.m = n_bits
        self.N = 2**n_bits - 1
        self.dac_mode = dac_mode
        self.thermo_bits = thermo_bits
        self.run_flags = run_flags
        self.sigma_c = sigma_c
        self.vdd = vdd
        self.vth = vth
        self.ich = ich
        self.cramp_u = cramp_u
        self.i1 = i1
        self.c1 = c1
        self.c2 = c2
        self.C_fixed = C_fixed
        

        if isinstance(self_power_down, str):
            self.self_power_down = self_power_down.strip().lower() in {"yes", "true", "1", "on"}
        else:
            self.self_power_down = bool(self_power_down)

    def build_dac_array(self, mismatch_enable=False, sigma=None):
        sigma_val = self.sigma_c if sigma is None else sigma
        mode_key = self.dac_mode.lower()

        if mode_key == "binary":
            arr = np.zeros(self.m)
            for j in range(self.m):
                mismatch = np.random.randn() * sigma_val / np.sqrt(2**j) if mismatch_enable else 0.0
                arr[j] = (2**j) * (self.cramp_u + mismatch)
            return arr

        if mode_key == "thermometer":
            n_units = (2**self.m) - 1
            arr = np.zeros(n_units)
            for j in range(n_units):
                mismatch = np.random.randn() * sigma_val if mismatch_enable else 0.0
                arr[j] = self.cramp_u + mismatch
            return arr

        if mode_key == "segmented":
            t = int(self.thermo_bits)
            b = self.m - t

            arr_bin = np.zeros(b)
            for j in range(b):
                mismatch = np.random.randn() * sigma_val / np.sqrt(2**j) if mismatch_enable else 0.0
                arr_bin[j] = (2**j) * (self.cramp_u + mismatch)

            n_units = (2**t) - 1
            unit_val_ideal = (2**b) * self.cramp_u
            arr_therm = np.zeros(n_units)
            for j in range(n_units):
                mismatch = np.random.randn() * sigma_val * np.sqrt(2**b) if mismatch_enable else 0.0
                arr_therm[j] = unit_val_ideal + mismatch

            return np.concatenate([arr_bin, arr_therm])

        raise ValueError("dac_mode must be 'binary', 'thermometer', or 'segmented'")

    def calc_ck(self, code, arr):
        lsb_code = code & ((2**self.m) - 1)
        mode_key = self.dac_mode.lower()

        if mode_key == "binary":
            c_k_local = 0.0
            for j in range(self.m):
                bit = (lsb_code >> j) & 1
                c_k_local += bit * arr[j]
            return c_k_local

        if mode_key == "thermometer":
            turned_on = int(min(max(lsb_code, 0), len(arr)))
            return float(np.sum(arr[:turned_on]))

        if mode_key == "segmented":
            t = int(self.thermo_bits)
            b = self.m - t

            binary_code = lsb_code & ((2**b) - 1)
            thermo_index = lsb_code >> b
            arr_bin = arr[:b]
            arr_therm = arr[b:]

            c_k_bin = 0.0
            for j in range(b):
                bit = (binary_code >> j) & 1
                c_k_bin += bit * arr_bin[j]

            c_k_therm = np.sum(arr_therm[:thermo_index])
            return float(c_k_bin + c_k_therm)

        raise ValueError("dac_mode must be 'binary', 'thermometer', or 'segmented'")

    def energy(self, c_k):
        power_scale = 0.25 if self.self_power_down else 1.0
        return c_k * self.vdd**2 * power_scale

    def compute_delay(self, vst_array, cap_array, clm_enabled=None, nonlin_enabled=None, ich_val=None):
        if clm_enabled is None:
            clm_enabled = self.run_flags["CLM"]
        if nonlin_enabled is None:
            nonlin_enabled = self.run_flags["Non-linearities-capacitor"]

        ich_base = self.ich if ich_val is None else ich_val

        delay = np.zeros(len(vst_array))
        ich_array = np.zeros(len(vst_array))
        cramp_array = np.zeros(len(vst_array))
        energy_array = np.zeros(len(vst_array))

        for i, vst in enumerate(vst_array):
            ich_eff = ich_base
            cramp_eff = self.calc_ck(i, cap_array)
            energy_array[i] = self.energy(cramp_eff)

            if clm_enabled:
                ich_eff = ich_base * (1 + self.i1 * (vst - 0.4))
                ich_array[i] = ich_eff

            if nonlin_enabled:
                cramp_eff = cramp_eff * (1 + self.c1 * vst + self.c2 * vst**2)
                cramp_array[i] = cramp_eff
            else:
                cramp_array[i] = cramp_eff

            delay[i] = (self.C_fixed +cramp_eff) * (self.vdd - self.vth) / ich_eff


        return delay, ich_array, cramp_array, energy_array


class DelayLineDTC:
    """Replica delay-line model with Cramp-dominated delay and power.

    Equations:
      t_total = ((Vdd - Vth) / Ich) * Cramp * N
      P_total = Cramp * Vdd^2 * f * N

    where N = 2^n_bits - 1 is the number of active replicas.

    Mismatch is applied directly to Cramp as a global factor. This changes only
    global delay/power scaling and does not introduce code-dependent nonlinearity.
    """

    def __init__(
        self,
        n_bits,
        vdd,
        vth,
        ich,
        cramp,
        run_flags,
        i1,
        c1,
        c2,
        sigma_cramp=0.0,
        self_power_down=False,
        
    ):
        self.n_bits = n_bits
        self.m = n_bits
        self.N = 2**n_bits - 1
        self.run_flags = run_flags
        self.sigma_c = sigma_cramp
        self.cramp = cramp
        self.vdd = vdd
        self.vth = vth
        self.ich = ich
        self.i1 = i1
        self.c1 = c1
        self.c2 = c2

        if isinstance(self_power_down, str):
            self.self_power_down = self_power_down.strip().lower() in {"yes", "true", "1", "on"}
        else:
            self.self_power_down = bool(self_power_down)

    def n_replicas(self):
        return (2**self.n_bits) - 1

    def _sample_cramp(self, mismatch_enable=False):
        if not mismatch_enable or self.sigma_cramp == 0.0:
            return self.cramp
        return self.cramp + np.random.randn() * self.sigma_cramp
    
    def energy(self, cramp_eff):
        energy = cramp_eff * (self.vdd**2)
        if self.self_power_down:
            energy = energy / 4.0
        return energy
    
    def compute_delay(self, vst_array, clm_enabled=None, nonlin_enabled=None, ich_val=None):
        if clm_enabled is None:
            clm_enabled = self.run_flags["CLM"]
        if nonlin_enabled is None:
            nonlin_enabled = self.run_flags["Non-linearities-capacitor"]

        ich_base = self.ich if ich_val is None else ich_val

        delay = np.zeros(len(vst_array))
        ich_array = np.zeros(len(vst_array))
        cramp_array = np.zeros(len(vst_array))
        energy_array = np.zeros(len(vst_array))

        for i, vst in enumerate(vst_array):
            ich_eff = ich_base
            cramp = self.cramp
            energy_array[i] = self.energy(cramp)

            if clm_enabled:
                ich_eff = ich_base * (1 + self.i1 * (vst - 0.4))
                ich_array[i] = ich_eff

            if nonlin_enabled:
                cramp_eff = cramp_eff * (1 + self.c1 * vst + self.c2 * vst**2)
                cramp_array[i] = cramp_eff
            else:
                cramp_array[i] = cramp_eff

            delay[i] = cramp_eff * (self.vdd - self.vth) / ich_eff

        return delay, ich_array, cramp_array, energy_array


def save_figure_to(fig, filename, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")


def compute_dnl_inl(delay_s):
    if len(delay_s) < 3:
        return np.array([]), np.array([]), 0.0

    steps = np.diff(delay_s)
    lsb = (delay_s[-1] - delay_s[0]) / (len(delay_s) - 1)
    if abs(lsb) < 1e-30:
        return np.zeros(len(delay_s) - 1), np.zeros(len(delay_s) - 1), 0.0

    dnl = steps / lsb - 1
    inl = np.cumsum(dnl)
    return dnl, inl, lsb


def plot_delay_power(codes, delay_s, power_w, out_dir, name_prefix):
    fig_delay = plt.figure()
    ax = fig_delay.add_subplot(111)
    ax.plot(codes, delay_s * 1e9, color='#D62728', linewidth=2.6, label='Delay')
    ax.set_title(f'{name_prefix} Delay vs Digital Code', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
    save_figure_to(fig_delay, f'{name_prefix.lower().replace(" ", "_")}_delay_vs_code.png', out_dir)
    plt.close(fig_delay)

    fig_power = plt.figure()
    ax = fig_power.add_subplot(111)
    ax.plot(codes, power_w * 1e6, color='#1F77B4', linewidth=2.6, label='Power')
    ax.set_title(f'{name_prefix} Power vs Digital Code', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power [uW]', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
    save_figure_to(fig_power, f'{name_prefix.lower().replace(" ", "_")}_power_vs_code.png', out_dir)
    plt.close(fig_power)


def plot_mc_dnl_inl(mc_delay_list, out_dir, name_prefix, n_runs):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=_multi_panel_figsize(2, 1))

    for idx, delay_s in enumerate(mc_delay_list):
        dnl, inl, _ = compute_dnl_inl(delay_s)
        x = np.arange(1, len(delay_s))
        alpha_value = 0.15 + (idx / max(len(mc_delay_list) - 1, 1)) * 0.45
        ax1.plot(x, dnl, color='#D62728', linewidth=1.5, alpha=alpha_value)
        ax2.plot(x, inl, color='#1F77B4', linewidth=1.5, alpha=alpha_value)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax1.set_ylabel('DNL (LSB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{name_prefix} Monte Carlo DNL/INL ({n_runs} realizations)', fontsize=14, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax1.set_axisbelow(True)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax2.set_ylabel('INL (LSB)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    save_figure_to(fig, f'{name_prefix.lower().replace(" ", "_")}_mc_dnl_inl_{n_runs}runs.png', out_dir)
    plt.close(fig)


def plot_aux_effects(codes, ich_s, cramp_s, out_dir, name_prefix):
    """Plot CLM current and effective ramp capacitance when enabled."""
    if np.any(np.abs(ich_s) > 0):
        fig_current = plt.figure()
        ax = fig_current.add_subplot(111)
        ax.plot(codes, ich_s * 1e9, color='#2CA02C', linewidth=2.6, label='Effective Current')
        ax.set_title(f'{name_prefix} Effective Current vs Digital Code', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Current [nA]', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
        save_figure_to(fig_current, f'{name_prefix.lower().replace(" ", "_")}_clm_current.png', out_dir)
        plt.close(fig_current)

    if np.any(np.abs(cramp_s) > 0):
        fig_cap = plt.figure()
        ax = fig_cap.add_subplot(111)
        ax.plot(codes, cramp_s * 1e15, color='#FF7F0E', linewidth=2.6, label='Effective Ramp Capacitance')
        ax.set_title(f'{name_prefix} Ramp Capacitance vs Digital Code', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capacitance [fF]', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
        save_figure_to(fig_cap, f'{name_prefix.lower().replace(" ", "_")}_ramp_capacitance.png', out_dir)
        plt.close(fig_cap)


def report_mismatch_stats(cap_array, ideal_array, name_prefix):
    delta = cap_array - ideal_array
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(np.abs(ideal_array) > 0, delta / ideal_array, 0.0)

    rms_rel = np.sqrt(np.mean(rel**2))
    max_rel = np.max(np.abs(rel))
    rms_abs = np.sqrt(np.mean(delta**2))
    print(
        f"[{name_prefix}] mismatch stats: "
        f"RMS rel={rms_rel*100:.4f}% | "
        f"Max rel={max_rel*100:.4f}% | "
        f"RMS abs={rms_abs*1e15:.4e} fF"
    )


def run_constant_slope_simulation(sim, freq_hz, mismatch_enable, cramp, c0_factor, report_mismatch=True):
    cap_array = sim.build_dac_array(mismatch_enable=mismatch_enable)
    if mismatch_enable and report_mismatch:
        ideal_array = sim.build_dac_array(mismatch_enable=False, sigma=0.0)
        report_mismatch_stats(cap_array, ideal_array, 'ConstantSlope DAC')

    ca = np.sum(cap_array)
    c0 = c0_factor*ca - cramp

    vst_array, energy_array, _ = sim.compute_vst_energy(cap_array, c0, cramp)
    delay_array, ich_array, cramp_array = sim.compute_delay(vst_array)

    delay_trim = np.delete(delay_array, sim.half)
    power_full = energy_array * freq_hz
    power_trim = np.delete(energy_array, sim.half) * freq_hz
    ich_trim = np.delete(ich_array, sim.half)
    cramp_trim = np.delete(cramp_array, sim.half)
    codes = np.arange(len(delay_trim))

    return {
        'cap_array': cap_array,
        'ca': ca,
        'c0': c0,
        'vst_array_full': vst_array,
        'delay_array_full': delay_array,
        'power_array_full': power_full,
        'delay_array': delay_trim,
        'power_array': power_trim,
        'ich_array': ich_trim,
        'cramp_array': cramp_trim,
        'cramp_array_full': cramp_array,
        'codes': codes,
        'half': sim.half,
    }


def run_variable_slope_simulation(sim, vst_array, freq_hz, mismatch_enable, report_mismatch=True):
    cap_array = sim.build_dac_array(mismatch_enable=mismatch_enable)
    if mismatch_enable and report_mismatch:
        ideal_array = sim.build_dac_array(mismatch_enable=False, sigma=0.0)
        report_mismatch_stats(cap_array, ideal_array, 'VariableSlope CrampDAC')

    delay_array, ich_array, cramp_array, energy_array = sim.compute_delay(vst_array, cap_array)

    
    delay_trim = delay_array
    
    power_full = energy_array * freq_hz
    power_trim = power_full
    ich_trim = ich_array
    cramp_trim = cramp_array


    codes = np.arange(len(delay_trim))

    return {
        'cap_array': cap_array,
        'delay_array_full': delay_array,
        'power_array_full': power_full,
        'delay_array': delay_trim,
        'power_array': power_trim,
        'ich_array': ich_trim,
        'cramp_array': cramp_trim,
        'cramp_array_full': cramp_array,
        'codes': codes,
    }

def run_delay_line_simulation(sim, vst_array, freq_hz, mismatch_enable, report_mismatch=True):
    cramp = sim._sample_cramp(sim, mismatch_enable=mismatch_enable)
    if mismatch_enable and report_mismatch:
        ideal_cap = sim._sample_cramp(mismatch_enable=False, sigma=0.0)
        report_mismatch_stats(cramp, ideal_cap, 'Delay line Cramp')

    delay_array, ich_array, cramp_array, energy_array = sim.compute_delay(vst_array)


    delay_trim = delay_array
    power_full = energy_array * freq_hz
    power_trim = power_full
    ich_trim = ich_array
    cramp_trim = cramp_array
    

    codes = np.arange(len(delay_trim))

    return {
        'cap_array': cramp_array,
        'delay_array_full': delay_array,
        'power_array_full': power_full,
        'delay_array': delay_trim,
        'power_array': power_trim,
        'ich_array': ich_trim,
        'cramp_array': cramp_trim,
        'cramp_array_full': cramp_array,
        'codes': codes,
    }



def lsb_from_curve(delay_arr):
    return (delay_arr[-1] - delay_arr[0]) / (len(delay_arr) - 1)


def print_summary(label, delay_arr, power_arr_full, f_hz, vst_array_full, lsb_clm_val, lsb_nonlin_val, lsb_both_val):
    delta_v = (vst_array_full[-1] - vst_array_full[0]) * 1e3
    t_offset = delay_arr[0] * 1e9
    t_range = (delay_arr[-1] - delay_arr[0]) * 1e9
    resolution = (delay_arr[-1] - delay_arr[0]) * 1e12 / len(delay_arr)

    print(f"\n[{label}]")
    print(f"Delta V = {delta_v} mV")
    print(f"T_offset (ideal) = {t_offset} ns")
    print(f"T_range (ideal) = {t_range} ns")
    print(f"Resolution (ideal) = {resolution} ps")
    print("")

    print(f"Power @ Code 0 (f={f_hz/1e6:.0f} MHz) = {power_arr_full[0]*1e6:.3e} uW")
    idx = len(power_arr_full) - 1
    print(f"Power @ Code {idx} (f={f_hz/1e6:.0f} MHz) = {power_arr_full[idx]*1e6:.3e} uW")
    print(f"CLM LSB = {lsb_clm_val:.2e}, Non-lin LSB = {lsb_nonlin_val:.2e}, Both LSB = {lsb_both_val:.2e}")
