import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# ===== PUBLICATION-READY PLOT STYLE =====
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

plt.rcParams.update({
    # Font sizes
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    
    # Line and marker styles
    "lines.linewidth": 2.6,
    "lines.markersize": 4,
    "lines.markeredgewidth": 1.0,
    
    # Grid
    "grid.alpha": 0.6,
    "grid.color": "#b7b7b7",
    "grid.linestyle": "--",
    "grid.linewidth": 1.2,
    
    # Figure
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    
    # Axes
    "axes.linewidth": 1.6,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "xtick.major.width": 1.4,
    "xtick.minor.width": 1.0,
    "ytick.major.width": 1.4,
    "ytick.minor.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
})


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
        'Cu': 30e-15,
        'Vdd': 1.1,
        'f': 20e6,
        'Ich': 300e-9,
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
    C0 = cfg.get('C0', Ca * 8/3)
    
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

def produce_delay(T, f_cnt, C_load, f_DTC, n_DTC, n_cnt, Vdd):
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
        
    Returns:
    --------
    P_cnt : float
        Counter power consumption in W
    power_DTC : float
        DTC power consumption in W
    total_delay : float
        Actual delay produced in seconds
    """
    # Counter period
    T_period = 1 / f_cnt
    
    # Get DTC delay range (all codes from 0 to 2^n_DTC - 1)
    t_min = DTC_block(0, {'n': n_DTC, 'Vdd': Vdd})[0]
    t_max = DTC_block(2**n_DTC - 1, {'n': n_DTC, 'Vdd': Vdd})[0]
    
    # Check if target is achievable
    if T < t_min:
        # Use DTC at minimum
        T_cnt_cycles = 0
        T_cnt = 0
        code = 0
    else:
        # Calculate how many counter cycles we need
        # We want: T_cnt + t_DTC = T, where t_min <= t_DTC <= t_max
        # To maximize counter usage: T_cnt = T - t_min (approximately)
        
        # Start with the counter cycles that gets us close
        T_cnt_cycles = int((T - t_min) // T_period)
        T_cnt = T_cnt_cycles * T_period
        
        # Check counter limit
        max_cnt_cycles = 2**n_cnt - 1
        if T_cnt_cycles >= max_cnt_cycles:
            print(f"Warning: T requires {T_cnt_cycles} counter cycles, exceeds maximum {max_cnt_cycles} for n={n_cnt}.")
            return None
        
        # Remaining delay for DTC
        T_DTC_target = T - T_cnt
        
        # If remaining delay is outside DTC range, adjust counter
        if T_DTC_target < t_min:
            # Need to reduce counter cycles
            if T_cnt_cycles > 0:
                T_cnt_cycles -= 1
                T_cnt = T_cnt_cycles * T_period
                T_DTC_target = T - T_cnt
            else:
                # Already at 0 counter cycles, use minimum DTC
                T_DTC_target = t_min
        elif T_DTC_target > t_max:
            # Need to increase counter cycles
            T_cnt_cycles += 1
            T_cnt = T_cnt_cycles * T_period
            T_DTC_target = T - T_cnt
            if T_DTC_target < t_min:
                print(f"Warning: Cannot achieve target delay {T*1e9:.3f} ns with given parameters.")
                return None
        
        # Calculate DTC code for target delay
        t_range = t_max - t_min
        resolution = t_range / (2**n_DTC - 1)  # LSB size (now 255 steps)
        
        # Map target delay to code
        T_DTC_normalized = T_DTC_target - t_min
        code = int(round(T_DTC_normalized / resolution))
        
        # Clamp code to valid range
        max_code = 2**n_DTC - 1
        code = max(0, min(code, max_code))
    
    # Get actual DTC delay and power for this code
    delay_DTC, power_DTC = DTC_block(code, {'n': n_DTC, 'Vdd': Vdd, 'f': f_DTC})
    
    # Total delay
    total_delay = T_cnt + delay_DTC
    
    # Counter power: P = C * V^2 * f
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
    # Generate target delays
    num_points = 500
    target_delays = np.linspace(0, delay_range_ns * 1e-9, num_points)
    actual_delays = []
    counter_transitions = []
    
    # Merge config if provided
    dtc_config = {'n': n_DTC, 'Vdd': Vdd, 'f': f_DTC}
    if config:
        dtc_config.update(config)
    
    # Track counter cycles for marking transitions
    prev_cnt_cycles = -1
    
    for T in target_delays:
        result = produce_delay(T, f_cnt, C_load, f_DTC, n_DTC, n_cnt, Vdd)
        if result is not None:
            P_cnt, power_DTC, total_delay, cnt_cycles, dtc_code = result
            actual_delays.append(total_delay * 1e9)  # Convert to ns
            
            # Mark counter transitions
            if cnt_cycles != prev_cnt_cycles and prev_cnt_cycles >= 0:
                counter_transitions.append(total_delay * 1e9)
            prev_cnt_cycles = cnt_cycles
    
    actual_delays = np.array(actual_delays)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(actual_delays, actual_delays, color='#D62728', linewidth=2.6, label='System Delay (Counter + DTC)')
    
    # Mark counter transitions with dots
    for transition in counter_transitions:
        ax.plot(transition, transition, 'o', color='#1F77B4', markersize=8, 
                markeredgewidth=1.5, markeredgecolor='white', zorder=5)
    
    # Add a dummy point for legend
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
    
    # Calculate delay for each code
    delays = np.zeros(len(codes))
    for i, code in enumerate(codes):
        delays[i], _ = DTC_block(code, config)
    
    # Convert to nanoseconds
    delays_ns = delays * 1e9
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(codes, delays_ns, color='#D62728', linewidth=2.6, label='DTC Delay')
    
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
    
    # Calculate power for each code
    powers = np.zeros(len(codes))
    for i, code in enumerate(codes):
        _, powers[i] = DTC_block(code, config)
    
    # Convert to microwatts
    powers_uw = powers * 1e6
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(codes, powers_uw, color='#1F77B4', linewidth=2.6, label='DTC Power')
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for idx, (label, config) in enumerate(configs_dict.items()):
        # Calculate delay for each code
        delays = np.zeros(len(codes))
        for i, code in enumerate(codes):
            delays[i], _ = DTC_block(code, config)
        
        # Convert to nanoseconds
        delays_ns = delays * 1e9
        
        # Plot
        color = colors[idx % len(colors)]
        linestyle = '-' if idx == 0 else '--'
        ax.plot(codes, delays_ns, color=color, linewidth=2.6, 
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
    
# Example usage and test
if __name__ == "__main__":
    print("=" * 60)
    print("DTC_block Function - Example Usage")
    print("=" * 60)
    
    # Test with a few codes
    test_codes = [0, 50, 100, 150, 200, 255]
    
    print("\n1. Basic usage with default parameters:")
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")
    
    print("\n\n2. With CLM enabled:")
    config_clm = {'enable_CLM': True}
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, config_clm)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")
    
    print("\n\n3. With both CLM and non-linearities enabled:")
    config_both = {'enable_CLM': True, 'enable_nonlin': True}
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, config_both)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")
    
    print("\n\n4. Custom frequency (50 MHz):")
    config_freq = {'f': 50e6}
    print(f"{'Code':<8} {'Delay (ns)':<15} {'Power (uW)':<15}")
    print("-" * 40)
    for code in test_codes:
        delay, power = DTC_block(code, config_freq)
        print(f"{code:<8} {delay*1e9:<15.3f} {power*1e6:<15.6f}")
    
    print("\n" + "=" * 60)
    
    # Test produce_delay function
    print("\n\n" + "=" * 60)
    print("produce_delay Function - Example Usage")
    print("=" * 60)
    
    # Example parameters
    f_cnt = 100e6  # 100 MHz counter frequency
    C_load = 10e-15  # 10 fF load capacitance
    f_DTC = 20e6  # 20 MHz DTC operating frequency
    n_DTC = 8  # 8-bit DTC
    n_cnt = 8  # 8-bit counter
    Vdd = 1.1  # 1.1V supply
    
    # Test with different target delays
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
        T = T_ns * 1e-9  # Convert to seconds
        result = produce_delay(T, f_cnt, C_load, f_DTC, n_DTC, n_cnt, Vdd)
        
        if result is not None:
            P_cnt, power_DTC, total_delay, cnt_cycles, dtc_code = result
            error_ps = (total_delay - T) * 1e12
            print(f"{T_ns:<12.1f} {total_delay*1e9:<12.3f} {error_ps:<12.3f} {cnt_cycles:<12} {dtc_code:<12} {P_cnt*1e6:<12.3f} {power_DTC*1e6:<12.6f}")
    
    print("\n" + "=" * 60)
    
    # Generate plots
    print("\n\n" + "=" * 60)
    print("Generating Publication-Ready Plots")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("plots_dtc")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Complete delay system characteristic (0-150 ns)
    print("\n1. Plotting complete delay system (0-150 ns) with counter period markers...")
    plot_delay_system(delay_range_ns=150, f_cnt=f_cnt, C_load=C_load, f_DTC=f_DTC,
                     n_DTC=n_DTC, n_cnt=n_cnt, Vdd=Vdd,
                     save_path=output_dir / "complete_delay_system.png")
    plt.close()
    
    # Plot 2: Complete delay system - zoomed to 30 ns for detail
    print("\n2. Plotting complete delay system (0-30 ns zoom) with counter period markers...")
    plot_delay_system(delay_range_ns=30, f_cnt=f_cnt, C_load=C_load, f_DTC=f_DTC,
                     n_DTC=n_DTC, n_cnt=n_cnt, Vdd=Vdd,
                     save_path=output_dir / "complete_delay_system_zoom.png")
    plt.close()
    
    # Plot 3: DTC only - delay vs code
    print("\n3. Plotting DTC delay vs code (all 256 codes)...")
    plot_delay_vs_code(save_path=output_dir / "dtc_delay_vs_code.png")
    plt.close()
    
    # Plot 4: DTC only - power vs code
    print("\n4. Plotting DTC power vs code...")
    plot_power_vs_code(save_path=output_dir / "dtc_power_vs_code.png")
    plt.close()
    
    # Plot 5: DTC delay comparison with different configurations
    print("\n5. Plotting DTC delay comparison (Ideal vs CLM vs Both)...")
    configs_comparison = {
        'Ideal': {},
        'With CLM': {'enable_CLM': True},
        'With Non-Linearities': {'enable_nonlin': True},
        'CLM + Non-Linearities': {'enable_CLM': True, 'enable_nonlin': True}
    }
    plot_delay_comparison(configs_comparison, save_path=output_dir / "dtc_delay_comparison.png")
    plt.close()
    
    # Plot 6: Power optimization analysis
    print("\n6. Analyzing optimal T_DTC for power minimization...")
    C_load_values = [5e-18, 10e-18, 20e-18, 50e-18, 100e-18]  # Various load capacitances
    plot_power_optimization(C_load_values=C_load_values, f_DTC=f_DTC, Vdd=Vdd, n_DTC=n_DTC,
                           save_path=output_dir / "power_optimization_T_DTC.png")
    plt.close()
    
    print(f"\n✓ All plots saved to: {output_dir.absolute()}")
    print("=" * 60)




