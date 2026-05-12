import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import apply_science_style


apply_science_style()


def DTC_block(code, config=None):
    """
    DTC block function that returns delay and power consumption for a given digital code.
    
    This function implements a Digital-to-Time Converter that converts a digital code
    to a corresponding time delay and calculates the associated power consumption.
    
    Parameters:
    -----------
    code : int
        Digital code input (0 to 2^n - 1, e.g., 0-255 for 8-bit)
        
    config : dict, optional
        Configuration dictionary with the following optional parameters:
        - n: number of bits (default: 8)
        - Cu: unit capacitance in F (default: 30e-15)
        - C0: reference capacitor in F (default: computed as Ca*8/3)
        - Vdd: supply voltage in V (default: 1.1)
        - f: operating frequency in Hz (default: 20e6)
        - Vth: threshold voltage in V (default: Vdd/2)
        - Ich: charging current in A (default: 300e-9)
        - Cramp: ramp capacitance in F (default: 5e-15)
        - C_array: custom capacitor array (optional, numpy array)
        - enable_CLM: enable channel length modulation (default: False)
        - enable_nonlin: enable capacitor non-linearities (default: False)
        - C1: non-linearity coefficient for capacitor (default: 0.323)
        - C2: non-linearity coefficient for capacitor (default: -0.09)
        - I1: CLM coefficient for current (default: 0.184)
        
    Returns:
    --------
    delay : float
        Delay in seconds
        
    power : float
        Power consumption in watts
        
    Example:
    --------
    >>> # Basic usage with default parameters
    >>> delay, power = DTC_block(127)
    >>> print(f"Delay: {delay*1e9:.2f} ns, Power: {power*1e6:.2f} uW")
    
    >>> # With custom configuration
    >>> config = {'f': 50e6, 'enable_CLM': True}
    >>> delay, power = DTC_block(200, config)
    """
    # Default configuration
    default_config = {
        'n': 8,
        'Cu': 8e-15,
        'Vdd': 1.1,
        'f': 20e6,
        'Ich': 350e-9,
        'Cramp': 5e-15,
        'enable_CLM': False,
        'enable_nonlin': False,
        'C1': 0.323,
        'C2': -0.09,
        'I1': 0.184
    }
    
    # Merge user config with defaults
    if config is None:
        config = {}
    cfg = {**default_config, **config}
    
    # Extract parameters
    n = cfg['n']
    Cu = cfg['Cu']
    Vdd = cfg['Vdd']
    f = cfg['f']
    Ich = cfg['Ich']
    Cramp = cfg['Cramp']
    Vth = cfg.get('Vth', Vdd/2)
    
    # Generate or use provided capacitor array
    if 'C_array' in cfg:
        C_array = cfg['C_array']
    else:
        # Ideal binary-weighted capacitor array
        C_array = np.array([(2**j) * Cu for j in range(n-1)])
    
    Ca = np.sum(C_array)
    C0 = cfg.get('C0', 1.5*Ca)  # Default C0 is 1.5 times total array capacitance
    
    N = 2**n
    
    # Validate code
    if code < 0 or code >= N:
        raise ValueError(f"Code must be between 0 and {N-1}")
    
    # Calculate C_k for this code (7 LSBs)
    C_k = 0
    for j in range(n-1):
        bit = (code >> j) & 1
        C_k += (1 - bit) * C_array[j]
    
    # MSB determines the mode
    msb = (code >> (n-1)) & 1
    
    # Calculate starting voltage (Vst) and energy
    if msb == 1:
        # MSB = 1
        Vst = (1 + (Ca - C_k) / (C0 + Ca)) * Vdd
        E = (Ca - C_k) / (C0 + Ca) * (C0 + C_k) * Vdd**2
    else:
        # MSB = 0
        Vst = (1 - C_k / (C0 + Ca)) * Vdd
        E = (C0 + (Ca - C_k)) * (C_k / (C0 + Ca)) * Vdd**2
    
    # Apply CLM and non-linearities if enabled
    Ich_eff = Ich
    Cramp_eff = Cramp
    
    if cfg['enable_CLM']:
        I1 = cfg['I1']
        Ich_eff = Ich * (1 + I1 * (Vst - 0.4))
    
    if cfg['enable_nonlin']:
        C1 = cfg['C1']
        C2 = cfg['C2']
        Cramp_eff = Cramp * (1 + C1 * Vst + C2 * Vst**2)
    
    # Calculate delay: delay = (Vth - Vst) / k_eff where k_eff = -Ich_eff/Cramp_eff
    k_eff = -Ich_eff / Cramp_eff
    delay = (Vth - Vst) / k_eff
    
    # Calculate power: P = E * f
    power = E * f
    
    return delay, power

def produce_delay(T, f_cnt, C_load, f_DTC, n_DTC, n_cnt, Vdd, dtc_config=None):
    """
    Produce a target delay T using a counter (coarse) and DTC (fine).

    Parameters:
    -----------
    T : float
        Target delay in seconds
    f_cnt : float
        Counter frequency in Hz
    C_load : float
        Load capacitance in F (for counter power calculation)
    f_DTC : float
        DTC operating frequency in Hz
    n_DTC : int
        Number of bits in DTC
    n_cnt : int
        Number of bits in counter
    Vdd : float
        Supply voltage in V
    dtc_config : dict, optional
        Configuration dictionary forwarded to DTC_block. Values in this dict
        can override n/Vdd/f and any DTC non-ideality parameters.

    Returns:
    --------
    P_cnt : float
        Counter power consumption in W
    power_DTC : float
        DTC power consumption in W
    total_delay : float
        Actual delay produced in seconds
    """
    T_period = 1 / f_cnt

    dtc_cfg = {'n': n_DTC, 'Vdd': Vdd, 'f': f_DTC}
    if dtc_config:
        dtc_cfg.update(dtc_config)

    n_dtc_eff = dtc_cfg['n']
    t_min = DTC_block(0, dtc_cfg)[0]
    t_max = DTC_block(2**n_dtc_eff - 1, dtc_cfg)[0]

    if T < t_min:
        T_cnt_cycles = 0
        T_cnt = 0
        code = 0
    else:
        T_cnt_cycles = int((T - t_min) // T_period)
        T_cnt = T_cnt_cycles * T_period

        max_cnt_cycles = 2**n_cnt - 1
        if T_cnt_cycles >= max_cnt_cycles:
            print(f"Warning: T requires {T_cnt_cycles} counter cycles, exceeds maximum {max_cnt_cycles} for n={n_cnt}.")
            return None

        T_DTC_target = T - T_cnt

        if T_DTC_target < t_min:
            if T_cnt_cycles > 0:
                T_cnt_cycles -= 1
                T_cnt = T_cnt_cycles * T_period
                T_DTC_target = T - T_cnt
            else:
                T_DTC_target = t_min
        elif T_DTC_target > t_max:
            T_cnt_cycles += 1
            T_cnt = T_cnt_cycles * T_period
            T_DTC_target = T - T_cnt
            if T_DTC_target < t_min:
                print(f"Warning: Cannot achieve target delay {T*1e9:.3f} ns with given parameters.")
                return None

        t_range = t_max - t_min
        resolution = t_range / (2**n_dtc_eff - 1)
        T_DTC_normalized = T_DTC_target - t_min
        code = int(round(T_DTC_normalized / resolution))

        max_code = 2**n_dtc_eff - 1
        code = max(0, min(code, max_code))

    delay_DTC, power_DTC = DTC_block(code, dtc_cfg)
    total_delay = T_cnt + delay_DTC
    P_cnt = C_load * Vdd**2 * f_cnt

    return P_cnt, power_DTC, total_delay, T_cnt_cycles, code


def plot_delay_system(delay_range_ns=150, f_cnt=100e6, C_load=10e-15, f_DTC=20e6,
                      n_DTC=8, n_cnt=8, Vdd=1.1, config=None, save_path=None):
    """
    Plot complete delay system (counter + DTC) characteristic with publication-ready styling.
    Marks counter period transitions with dots.

    Parameters:
    -----------
    delay_range_ns : float
        Maximum delay range to plot in nanoseconds (default: 150)
    f_cnt : float
        Counter frequency in Hz (default: 100e6)
    C_load : float
        Load capacitance in F (default: 10e-15)
    f_DTC : float
        DTC operating frequency in Hz (default: 20e6)
    n_DTC : int
        Number of bits in DTC (default: 8)
    n_cnt : int
        Number of bits in counter (default: 8)
    Vdd : float
        Supply voltage in V (default: 1.1)
    config : dict, optional
        Additional configuration for DTC (CLM, non-linearities)
    save_path : str or Path, optional
        Path to save the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    num_points = 500
    target_delays = np.linspace(0, delay_range_ns * 1e-9, num_points)
    actual_delays = []
    counter_transitions = []

    dtc_config = {'n': n_DTC, 'Vdd': Vdd, 'f': f_DTC}
    if config:
        dtc_config.update(config)

    prev_cnt_cycles = -1
    for T in target_delays:
        result = produce_delay(T, f_cnt, C_load, f_DTC, n_DTC, n_cnt, Vdd, dtc_config=dtc_config)
        if result is not None:
            _, _, total_delay, cnt_cycles, _ = result
            actual_delays.append(total_delay * 1e9)
            if cnt_cycles != prev_cnt_cycles and prev_cnt_cycles >= 0:
                counter_transitions.append(total_delay * 1e9)
            prev_cnt_cycles = cnt_cycles

    actual_delays = np.array(actual_delays)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(actual_delays, actual_delays, color='#D62728', linewidth=2.6, label='System Delay (Counter + DTC)')

    for transition in counter_transitions:
        ax.plot(transition, transition, 'o', color='#1F77B4', markersize=8,
                markeredgewidth=1.5, markeredgecolor='white', zorder=5)

    ax.plot([], [], 'o', color='#1F77B4', markersize=8, markeredgewidth=1.5,
            markeredgecolor='white', label='Counter Period Transition')

    ax.set_title('Complete Delay System Characteristic (Counter + DTC)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Target Delay [ns]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Delay [ns]', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_delay_vs_code(config=None, save_path=None):
    """
    Plot DTC delay characteristic vs digital code with publication-ready styling.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary for DTC_block (see DTC_block documentation)
    save_path : str or Path, optional
        Path to save the figure. If None, figure is displayed but not saved.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Get configuration
    if config is None:
        config = {}
    n = config.get('n', 8)
    
    # Generate all codes
    N = 2**n
    codes = np.arange(N)
    mid_code = 2**(n - 1)
    codes_shifted = np.arange(N - 1)
    
    # Calculate delay for each code
    delays = np.zeros(len(codes))
    for i, code in enumerate(codes):
        delays[i], _ = DTC_block(code, config)
    
    # Convert to nanoseconds
    delays_ns = np.delete(delays, mid_code) * 1e9
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(codes_shifted, delays_ns, color='#D62728', linewidth=2.6, label='DTC Delay')
    
    ax.set_title('DTC Delay Characteristic vs Digital Code', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_power_vs_code(config=None, save_path=None):
    """
    Plot power consumption vs digital code with publication-ready styling.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary for DTC_block (see DTC_block documentation)
    save_path : str or Path, optional
        Path to save the figure. If None, figure is displayed but not saved.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Get configuration
    if config is None:
        config = {}
    n = config.get('n', 8)
    
    # Generate all codes
    N = 2**n
    codes = np.arange(N)
    mid_code = 2**(n - 1)
    codes_shifted = np.arange(N - 1)
    
    # Calculate power for each code
    powers = np.zeros(len(codes))
    for i, code in enumerate(codes):
        _, powers[i] = DTC_block(code, config)
    
    # Convert to microwatts
    powers_uw = np.delete(powers, mid_code) * 1e6
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(codes_shifted, powers_uw, color='#1F77B4', linewidth=2.6, label='DTC Power')
    
    ax.set_title('DTC Power Consumption vs Digital Code', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_delay_comparison(configs_dict, save_path=None):
    """
    Plot delay characteristics for multiple configurations.
    
    Parameters:
    -----------
    configs_dict : dict
        Dictionary with labels as keys and config dictionaries as values
        Example: {'Ideal': {}, 'With CLM': {'enable_CLM': True}}
    save_path : str or Path, optional
        Path to save the figure. If None, figure is displayed but not saved.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Color palette
    colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD', '#8C564B']
    
    # Get first config to determine number of bits
    first_config = list(configs_dict.values())[0]
    n = first_config.get('n', 8)
    
    # Generate all codes
    N = 2**n
    codes = np.arange(N)
    mid_code = 2**(n - 1)
    codes_shifted = np.arange(N - 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for idx, (label, config) in enumerate(configs_dict.items()):
        # Calculate delay for each code
        delays = np.zeros(len(codes))
        for i, code in enumerate(codes):
            delays[i], _ = DTC_block(code, config)
        
        # Remove middle code and reindex to continuous shifted code axis
        delays_ns = np.delete(delays, mid_code) * 1e9
        
        # Plot
        color = colors[idx % len(colors)]
        linestyle = '-' if idx == 0 else '--'
        ax.plot(codes_shifted, delays_ns, color=color, linewidth=2.6, 
                linestyle=linestyle, label=label)
    
    ax.set_title('DTC Delay Characteristic Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def run_mc_mismatch_analysis(mc_runs=100, config=None, dnl_limit_lsb=0.5, save_path=None):
    """
    Run Monte Carlo mismatch simulation and plot DNL/INL clouds.

    This follows the approach used in DTC_simulation.py with capacitor mismatch
    applied to the binary capacitor array on every run.

    Parameters:
    -----------
    mc_runs : int
        Number of Monte Carlo realizations
    config : dict, optional
        Base DTC configuration; supports all DTC_block parameters
    dnl_limit_lsb : float
        DNL pass threshold in LSB for reporting pass probability
    save_path : str or Path, optional
        Output path for the Monte Carlo DNL/INL figure

    Returns:
    --------
    fig, (ax1, ax2), stats : tuple
        Figure, axes and dictionary with summary statistics
    """
    if config is None:
        config = {}

    default_cfg = {
        'n': 8,
        'Cu': 8e-15,
        'Vdd': 1.1,
        'f': 20e6,
        'Ich': 350e-9,
        'Cramp': 5e-15,
        'enable_CLM': False,
        'enable_nonlin': False,
        'C1': 0.323,
        'C2': -0.09,
        'I1': 0.184,
        'C0_scale': 1.5,
        'Ac': 5.218e-3,
        'A': 4.33,
    }
    cfg = {**default_cfg, **config}

    n = cfg['n']
    Cu = cfg['Cu']
    Ac = cfg['Ac']
    area = cfg['A']
    c0_scale = cfg['C0_scale']

    sigma_c = Ac / np.sqrt(area) * Cu
    codes = np.arange(2**n)
    mid_code = 2**(n - 1)
    codes_dnl = np.arange(1, len(codes) - 1)

    passing_runs = 0
    dnl_peaks = []
    lsb_last = None

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    for mc in range(mc_runs):
        c_array_mc = np.array([(2**j) * (Cu + np.random.randn() * sigma_c) for j in range(n - 1)])
        ca_mc = np.sum(c_array_mc)
        c0_mc = c0_scale * ca_mc

        run_cfg = dict(cfg)
        run_cfg['C_array'] = c_array_mc
        run_cfg['C0'] = c0_mc

        delays = np.zeros(len(codes))
        for i, code in enumerate(codes):
            delays[i], _ = DTC_block(code, run_cfg)

        delays = np.delete(delays, mid_code)

        lsb = (delays[-1] - delays[0]) / (len(delays) - 1)
        lsb_last = lsb
        dnl = np.diff(delays) / lsb - 1
        inl = np.cumsum(dnl)

        dnl_peak = np.max(dnl)
        dnl_peaks.append(dnl_peak)
        if dnl_peak < dnl_limit_lsb:
            passing_runs += 1

        alpha_value = 0.15 + (mc / max(mc_runs - 1, 1)) * 0.5
        ax1.plot(codes_dnl, dnl, alpha=alpha_value, linewidth=1.6, color='#D62728')
        ax2.plot(codes_dnl, inl, alpha=alpha_value, linewidth=1.6, color='#1F77B4')

    ax1.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
    ax1.set_title(
        f"Monte Carlo DNL and INL Analysis ({mc_runs} Realizations, LSB = {lsb_last:.2e})",
        fontsize=14,
        fontweight='bold',
        pad=12,
    )
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax1.axhline(y=dnl_limit_lsb, color='red', linestyle='--', linewidth=1.0, alpha=0.4,
                label=f'±{dnl_limit_lsb:.1f} LSB')
    ax1.axhline(y=-dnl_limit_lsb, color='red', linestyle='--', linewidth=1.0, alpha=0.4)
    ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax1.set_axisbelow(True)

    ax2.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax2.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    stats = {
        'mc_runs': mc_runs,
        'sigma_c': sigma_c,
        'lsb': lsb_last,
        'pass_probability_percent': (passing_runs / mc_runs) * 100,
        'max_dnl_peak': float(np.max(dnl_peaks)),
        'min_dnl_peak': float(np.min(dnl_peaks)),
        'mean_dnl_peak': float(np.mean(dnl_peaks)),
    }
    return fig, (ax1, ax2), stats
    
# Example usage and test
if __name__ == "__main__":
    print("=" * 60)
    print("DTC_block Function - Example Usage")
    print("=" * 60)

    f_cnt = 80e6
    C_load = 120e-15
    f_DTC = 20e6
    n_DTC = 8
    n_cnt = 8
    Vdd = 1.1

    DTC_CONFIG_MAIN = {
        'n': n_DTC,
        'Vdd': Vdd,
        'f': f_DTC,
        'Cu': 8e-15,
        'Ich': 350e-9,
        'Cramp': 5e-15,
        'enable_CLM': False,
        'enable_nonlin': False,
        'C1': 0.323,
        'C2': -0.09,
        'I1': 0.184,
    }

    test_codes = [0, 50, 100, 150, 200, 255]

    print("\n1. Basic usage with main configuration:")
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, DTC_CONFIG_MAIN)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")

    print("\n\n2. With CLM enabled:")
    config_clm = {**DTC_CONFIG_MAIN, 'enable_CLM': True}
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, config_clm)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")

    print("\n\n3. With both CLM and non-linearities enabled:")
    config_both = {**DTC_CONFIG_MAIN, 'enable_CLM': True, 'enable_nonlin': True}
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, config_both)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")

    print("\n\n4. Custom frequency (50 MHz):")
    config_freq = {**DTC_CONFIG_MAIN, 'f': 50e6}
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, config_freq)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")

    print("\n" + "=" * 60)

    print("\n\n" + "=" * 60)
    print("produce_delay Function - Example Usage")
    print("=" * 60)

    test_delays_ns = [5, 10, 15, 20, 50, 100, 150]

    print(f"\nParameters:")
    print(f"  Counter frequency: {f_cnt/1e6:.0f} MHz")
    print(f"  DTC frequency: {f_DTC/1e6:.0f} MHz")
    print(f"  Load capacitance: {C_load*1e15:.1f} fF")
    print(f"  Counter bits: {n_cnt}")
    print(f"  DTC bits: {n_DTC}")
    print(f"  Vdd: {Vdd} V")

    print(f"\n{'Target (ns)':<12} {'Actual (ns)':<12} {'Error (ps)':<12} {'Cnt Cycles':<12} {'DTC Code':<12} {'P_cnt (uW)':<12} {'P_DTC (uW)':<12}")
    print("-" * 100)

    for T_ns in test_delays_ns:
        T = T_ns * 1e-9
        result = produce_delay(T, f_cnt, C_load, f_DTC, n_DTC, n_cnt, Vdd, dtc_config=DTC_CONFIG_MAIN)
        if result is not None:
            P_cnt, power_DTC, total_delay, cnt_cycles, dtc_code = result
            error_ps = (total_delay - T) * 1e12
            print(f"{T_ns:<12.1f} {total_delay*1e9:<12.3f} {error_ps:<12.3f} {cnt_cycles:<12} {dtc_code:<12} {P_cnt*1e6:<12.3f} {power_DTC*1e6:<12.6f}")

    print("\n" + "=" * 60)

    print("\n\n" + "=" * 60)
    print("Generating Publication-Ready Plots")
    print("=" * 60)

    RUN_MC_SIMULATION = True
    MC_RUNS = 100

    output_dir = Path("C:\\Users\\zipar\\OneDrive - Delft University of Technology\\Second Year\\MEP\\plots_dtc")
    output_dir.mkdir(exist_ok=True)

    print("\n1. Plotting complete delay system (0-150 ns) with counter period markers...")
    plot_delay_system(delay_range_ns=150, f_cnt=f_cnt, C_load=C_load, f_DTC=f_DTC,
                      n_DTC=n_DTC, n_cnt=n_cnt, Vdd=Vdd, config=DTC_CONFIG_MAIN,
                      save_path=output_dir / "complete_delay_system.png")
    plt.close()

    print("\n2. Plotting complete delay system (0-30 ns zoom) with counter period markers...")
    plot_delay_system(delay_range_ns=30, f_cnt=f_cnt, C_load=C_load, f_DTC=f_DTC,
                      n_DTC=n_DTC, n_cnt=n_cnt, Vdd=Vdd, config=DTC_CONFIG_MAIN,
                      save_path=output_dir / "complete_delay_system_zoom.png")
    plt.close()

    print("\n3. Plotting DTC delay vs code (all 256 codes)...")
    plot_delay_vs_code(config=DTC_CONFIG_MAIN, save_path=output_dir / "dtc_delay_vs_code.png")
    plt.close()

    print("\n4. Plotting DTC power vs code...")
    plot_power_vs_code(config=DTC_CONFIG_MAIN, save_path=output_dir / "dtc_power_vs_code.png")
    plt.close()

    print("\n5. Plotting DTC delay comparison (Ideal vs CLM vs Both)...")
    configs_comparison = {
        'Ideal': dict(DTC_CONFIG_MAIN),
        'With CLM': {**DTC_CONFIG_MAIN, 'enable_CLM': True},
        'With Non-Linearities': {**DTC_CONFIG_MAIN, 'enable_nonlin': True},
        'CLM + Non-Linearities': {**DTC_CONFIG_MAIN, 'enable_CLM': True, 'enable_nonlin': True}
    }
    plot_delay_comparison(configs_comparison, save_path=output_dir / "dtc_delay_comparison.png")
    plt.close()

    if RUN_MC_SIMULATION:
        print("\n6. Running Monte Carlo mismatch simulation (DNL/INL cloud)...")
        mc_config = {
            **DTC_CONFIG_MAIN,
            'Ac': 5.218e-3,
            'A': 4.33,
            'C0_scale': 1.5,
        }
        _, _, mc_stats = run_mc_mismatch_analysis(
            mc_runs=MC_RUNS,
            config=mc_config,
            dnl_limit_lsb=0.5,
            save_path=output_dir / "dtc_mc_analysis.png",
        )
        plt.close()
        print(
            f"Probability of staying below 0.5 LSB = "
            f"{mc_stats['pass_probability_percent']:.2f}%"
        )

    

    


