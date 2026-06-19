import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from plot_style import apply_science_style, _multi_panel_figsize


apply_science_style()

LABEL_SIZE = plt.rcParams.get("axes.labelsize", 10)
TITLE_SIZE = plt.rcParams.get("axes.titlesize", 9)
TICK_SIZE = plt.rcParams.get("xtick.labelsize", 8)
LEGEND_SIZE = plt.rcParams.get("legend.fontsize", 6)
ANNOTATION_SIZE = plt.rcParams.get("font.size", 6)

# ============================================================
# CORE FUNCTIONS
# ============================================================

def get_ca_cdac(n, Cu0):
    """Calculate capacitance values based on bit width."""
    Cu = Cu0 * 2**((n-1))
    if Cu < 0.5e-15:
        Cu = 0.5e-15  # Set a minimum unit capacitance to avoid unrealistic values
    Ca = Cu * (2**n - 1)  # Regular: full CDAC
    return Cu, Ca

def get_p_ana_coefficient(use_bit_extension=False):
    """Get analog power coefficient (divide factor)."""
    return 2 if use_bit_extension else 1

def calculate_current(shot_noise, td, J =1e-12, F = 0.05, q =1.6e-19, k = 1.38e-23, T = 100, gamma = 2/3, Vov = 200e-3 ):
        "Calculate current based on jitter requirements considering shot noise."
        if shot_noise:
            return F*q*td/J**2
        else:
            return (4*k*T*gamma*td)/(Vov*J**2)

def calculate_c_ramp(shot_noise, td, K_slope, **current_kwargs):
    """Calculate ramp capacitance from the channel current and K_slope."""
    return calculate_current(shot_noise=shot_noise, td=td, **current_kwargs) / K_slope

def power_method(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=False):
    """
    Calculate power for all bit configurations.
    
    Parameters:
    -----------
    use_bit_extension : bool
        If True, use bit extension mode (one fewer bit in CDAC)
    """
    P_dict = {}
    T_DTC_max = 2*VDD / K_slope if use_bit_extension else VDD / K_slope
    p_ana_coeff = get_p_ana_coefficient(use_bit_extension)
    n_start = 3 if use_bit_extension else 1
    f = 50e6  # 100 MHz 
    
    for n in range(n_start, NBITS + 1):
        n_actual = (n - 1) if use_bit_extension else n
        Cu, Ca = get_ca_cdac(n_actual, Cu0)
        
        t_tot = (2**n-2)*res
        I_ch = calculate_current(shot_noise=False, td=t_tot)
        C_ramp = calculate_c_ramp(shot_noise=False, td=t_tot, K_slope=K_slope)


        Co_res = (Cu * VDD) / (K_slope * res) - Ca - C_ramp 
        Co_min = max(1 * (Ca + C_ramp), Co_res)
        
        if C_load==0:
            if T_DTC_max >5e-9:
                T_list = np.linspace(5e-9, T_DTC_max, 200)
            else:
                return {}  # No valid T_DTC range for zero load
        else:
            T_list = np.linspace(res, T_DTC_max, 200)
        P_total = np.zeros(len(T_list))
        P_cnt = np.zeros(len(T_list))
        P_ana = np.zeros(len(T_list))
        P_dyn = np.zeros(len(T_list))
        I_ch_list = np.zeros(len(T_list))
        C_ramp_list = np.zeros(len(T_list))
        
        t_corner = 2*(Ca * VDD) / (K_slope * (Co_min + Ca + C_ramp)) if use_bit_extension else (Ca * VDD) / (K_slope * (Co_min + Ca + C_ramp))
        I_ch_corner = calculate_current(shot_noise=False, td=t_corner)
        C_ramp_corner = calculate_c_ramp(shot_noise=False, td=t_corner, K_slope=K_slope)
        p_corner_ana = Co_min * Ca/(Co_min + Ca+ C_ramp)*VDD**2*f/p_ana_coeff + 1.5125*C_ramp/2*f
        p_corner_cnt = (C_load * VDD**2) / t_corner 

        for i, t_DTC in enumerate(T_list):
            I_ch = calculate_current(shot_noise=False, td=t_DTC)
            C_ramp = calculate_c_ramp(shot_noise=False, td=t_DTC, K_slope=K_slope)
            I_ch_list[i] = I_ch
            C_ramp_list[i] = C_ramp
            C0_req = 2*(Ca * VDD) / (K_slope * t_DTC) - Ca -C_ramp if use_bit_extension else (Ca * VDD) / (K_slope * t_DTC) - Ca -C_ramp
            C0 = max(C0_req, Co_min)
            DeltaV = (Ca * VDD) / (C0 + Ca + C_ramp)
            P_cnt[i] = (C_load * VDD**2) / t_DTC
            P_ana[i] = C0 * Ca/(C0 + Ca+C_ramp)*VDD**2*f/p_ana_coeff 
            P_dyn[i] = C_ramp/2*(VDD**2/2+(VDD+DeltaV/2)*DeltaV/2)*f
            P_total[i] = P_cnt[i] + P_ana[i] + P_dyn[i]
        
        P_dict[n] = {
            "T_list": T_list, "P_list": P_total, "P_cnt": P_cnt, "P_ana": P_ana, "P_dyn": P_dyn,
            "I_ch": I_ch_list, "C_ramp": C_ramp_list,
            "I_ch_corner": I_ch_corner, "C_ramp_corner": C_ramp_corner,
            "p_corner": p_corner_cnt + p_corner_ana, "t_corner": t_corner
        }
    
    return P_dict

def get_corners_only(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=False):
    """
    Calculate corner points for all bit configurations.
    
    Parameters:
    -----------
    use_bit_extension : bool
        If True, use bit extension mode
    """
    corners = []
    previous_t = 0
    p_ana_coeff = get_p_ana_coefficient(use_bit_extension)
    n_start = 3 if use_bit_extension else 2
    n_start_check = 4 if use_bit_extension else 2

    f= 50e6  # 100 MHz

    for n in range(n_start, NBITS + 1):
        I_ch = calculate_current(shot_noise=False, td=(2**n-2)*res)
        C_ramp = I_ch/K_slope
        n_actual = (n - 1) if use_bit_extension else n
        Cu, Ca = get_ca_cdac(n_actual, Cu0)
        
        Co_res = (Cu * VDD) / (K_slope * res) - Ca -C_ramp
        DeltaV = (Ca * VDD) / (Co_res + Ca + C_ramp)
        Co_min = max(k_lim *(Ca + C_ramp), Co_res)
        t_corner = 2*(Ca * VDD) / (K_slope * (Co_min + Ca + C_ramp)) if use_bit_extension else (Ca * VDD) / (K_slope * (Co_min + Ca + C_ramp))
        
        # Redundancy check
        if n >= n_start_check and (t_corner <= previous_t * 1.001):
            break
        
        p_ana = Co_min * Ca/(Co_min + Ca+ C_ramp)*VDD**2*f/p_ana_coeff 
        p_dyn = C_ramp/2*(VDD**2/2+(VDD+DeltaV/2)*DeltaV/2)*f
        p_dig = (C_load * VDD**2) / t_corner
        
        corners.append({
            "n": n, "t_ns": t_corner * 1e9,
            "C0_fF": Co_min * 1e15, "Ca_fF": Ca * 1e15,
            "p_uw": (p_ana + p_dig + p_dyn) * 1e6,
            "p_ana_uw": p_ana * 1e6, "p_dig_uw": p_dig * 1e6, "p_dyn_uw": p_dyn * 1e6,
            "K_slope_MV": K_slope / 1e6, "k_lim": k_lim,
            "I_ch_corner": I_ch, "C_ramp_corner": C_ramp
        })
        previous_t = t_corner
    
    return corners

def sweep_k_slopes(K_slopes, VDD, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=False):
    """Sweep K_slope values and find optimal points."""
    min_points = []
    for k_s in K_slopes:
        corners = get_corners_only(VDD, k_s, C_load, k_lim, Cu0, res, NBITS, use_bit_extension)
        if corners:
            best_corner = min(corners, key=lambda x: x['p_uw'])
            min_points.append(best_corner)
    return min_points

def plot_power_breakdown(ax, data_dict, best_n, title_suffix="", resolution_s=None):
    """Plot power breakdown for best configuration."""
    best_data = data_dict[best_n]
    t_ns = best_data["T_list"] * 1e9
    corner_i_ch = best_data["I_ch_corner"]
    corner_c_ramp = best_data["C_ramp_corner"]
    
    ax.semilogy(t_ns, best_data["P_list"] * 1e6, 'k', linewidth=1, label='Total Power')
    ax.semilogy(t_ns, best_data["P_ana"] * 1e6, '--', color='#D62728', linewidth=1,
                label='$P_{DTC,max}$', alpha=1)
    ax.semilogy(t_ns, best_data["P_cnt"] * 1e6, '--', color='#1F77B4', linewidth=1,
                label='$P_{cnt}$ ', alpha=1)
    ax.semilogy(t_ns, best_data["P_dyn"] * 1e6, '--', color='#2CA02C', linewidth=1,
                label='$P_{DTC,dyn}$', alpha=1)
    ax.scatter(best_data["t_corner"]*1e9, best_data["p_corner"]*1e6, color='red', s=50,
               edgecolors='black', linewidths=0.2, zorder=10, marker='o')
    x_res_ns = best_data["t_corner"] * 1e9
    ax.axvspan(x_res_ns, np.max(t_ns), color='red', alpha=0.08, zorder=0)
    ax.axvline(x_res_ns, linestyle='--', color='#7F7F7F', linewidth=1.0,
               label='Resolution not met')
    ax.set_xlabel(r'$\Delta t$ [ns]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title_suffix, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.grid(True, which="both", linestyle='--', alpha=0.2, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, framealpha=0.96, edgecolor='black')

def plot_all_bits(ax, data_dict, title="", bits_to_plot=None):
    """Plot bit configurations. Optionally filter to a subset via `bits_to_plot` (iterable of ints)."""
    t_corners, p_corners = [], []
    for n, data in data_dict.items():
        if bits_to_plot is not None and n not in bits_to_plot:
            continue
        ax.semilogy(data["T_list"]*1e9, data["P_list"]*1e6, alpha=0.5, linewidth=1.0, label=f'{n} bits')
        t_corners.append(data["t_corner"] * 1e9)
        p_corners.append(data["p_corner"] * 1e6)
    
    ax.scatter(t_corners, p_corners, color='red', s=50, zorder=5, edgecolors='black',
               linewidths=0.2, label='Corner Points')
    
    ax.set_xlabel(r'$\Delta t$ [ns]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.grid(True, which="both", linestyle='--', alpha=0.2, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, framealpha=0.96, edgecolor='black', ncol=2)

def plot_frontier_with_all_k_slopes(ax, K_slopes, VDD, C_load, k_lim, Cu0, res, NBITS, 
                                     use_bit_extension=False, title="", colors=None,
                                     max_points_per_curve=4,
                                     preferred_bits=None):
    """Plot frontier curves for different K_slope values."""
    if colors is None:
        colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    if preferred_bits is None:
        preferred_bits = [3, 5, 8, 9, 10, 11, 12]
    annotated_bits = [3, 5, 9, 10, 11, 12]
    
    t_max = []
    p_max = []
    p_min = []
    
    for idx, k_s in enumerate(K_slopes):
        corners = get_corners_only(VDD, k_s, C_load, k_lim, Cu0, res, NBITS, use_bit_extension)
        if not corners:
            continue
        
        t_vals = [c['t_ns'] for c in corners]
        p_vals = [c['p_uw'] for c in corners]
        n_vals = [c['n'] for c in corners]

        selected_indices = [i for i, n in enumerate(n_vals) if n in preferred_bits]
        if len(selected_indices) < 2:
            if len(t_vals) > max_points_per_curve:
                selected_indices = np.unique(
                    np.round(np.linspace(0, len(t_vals) - 1, max_points_per_curve)).astype(int)
                ).tolist()
            else:
                selected_indices = list(range(len(t_vals)))

        t_plot = [t_vals[i] for i in selected_indices]
        p_plot = [p_vals[i] for i in selected_indices]
        n_plot = [n_vals[i] for i in selected_indices]
        
        color = colors[idx % len(colors)]
        ax.plot(t_plot, p_plot, 'o-', color=color, linewidth=1.0, markersize=3.5,
                markeredgewidth=0.6, markeredgecolor='white',
                label=f'$K_{{slope}}$ = {k_s/1e6:.0f} MV/s')
        
        for i, n in enumerate(n_plot):
            if n not in annotated_bits:
                continue
            ax.annotate(
                f"{n}b",
                (t_plot[i], p_plot[i]),
                textcoords="offset points",
                xytext=(0, 7 + 3 * (i % 2)),
                ha='center',
                fontsize=ANNOTATION_SIZE - 1,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7),
            )
        
        t_max.append(max(t_vals))
        if max(t_vals) > 2:
            p_max.append(p_vals[2])  # 5th point (n=5) as reference for max power
            p_min.append(min(p_vals))

    ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta t$ [ns]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.96, edgecolor='black')
    if t_max:
        ax.set_xlim(left=0.05, right=max(t_max) * 1.05)
    if p_max:
        y_top = min(max(p_max) * 10, 200)
        y_min = max(min(p_min) * 0.8, 20)
        ax.set_ylim(bottom=y_min, top=max(y_top, 1e-3))

def plot_optimal_frontier(ax, min_points, title="", mode_label=""):
    """Plot optimal frontier with star for global minimum."""
    t_mins = [m['t_ns'] for m in min_points]
    p_mins = [m['p_uw'] for m in min_points]
    k_labels = [m['K_slope_MV'] for m in min_points]
    n_labels = [m['n'] for m in min_points]
    
    ax.plot(t_mins, p_mins, 'o', markersize=8, linewidth=0.2, color='#2CA02C',
            markeredgewidth=2, markeredgecolor='white', label='Optimal Design Frontier')
    
    # Mark global minimum
    best_idx = min_points.index(min(min_points, key=lambda x: x['p_uw']))
    ax.scatter([t_mins[best_idx]], [p_mins[best_idx]], marker='*', s=100,
               color='gold', edgecolors='red', linewidths=0.1, zorder=20, label='Global Minimum')
    
    # Annotate points
    for i in range(len(min_points)):
        ax.annotate(
            f"{n_labels[i]}b\n{k_labels[i]:.0f} MV/s",
            (t_mins[i], p_mins[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=4,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
        )
    
    ax.set_yscale('log')
    ax.set_xlabel(r'Optimal $\Delta t$ [ns]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Minimum Power [µW]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.96, edgecolor='black')

# ============================================================
# CONFIGURATION SECTION
# ============================================================

SAVE_DIR = Path("C:\\Users\\zipar\\OneDrive - Delft University of Technology\\Second Year\\MEP\\plots_optimization")
VDD, K_slope, C_load, k_lim, Cu0, res, NBITS = 1.1, 380e6, 100e-15, 300e6, 2.44e-19, 1.3e-12, 12
K_slopes = np.linspace(200e6, 600e6, 5)

k_lim_val = 1.0

SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"📂 Saving plots and results to: {SAVE_DIR.absolute()}\n")

# ============================================================
# REGULAR MODE OPTIMIZATION
# ============================================================

print("="*120)
print("REGULAR MODE - FINDING OPTIMAL K_slope")
print("="*120 + "\n")

min_points = sweep_k_slopes(K_slopes, VDD, C_load, k_lim_val, Cu0, res, NBITS, use_bit_extension=False)
df_min = pd.DataFrame(min_points)
best_corner_global = min(min_points, key=lambda x: x['p_uw'])

print(f"Best K_slope: {best_corner_global['K_slope_MV']:.1f} MV/s, {int(best_corner_global['n'])} bits, "
      f"T_DTC={best_corner_global['t_ns']:.3f} ns, Power={best_corner_global['p_uw']:.2f} µW\n")
best_corner_i_ch = best_corner_global['I_ch_corner']
best_corner_c_ramp = best_corner_global['C_ramp_corner']
print(f"Best-case corner values: C_ramp={best_corner_c_ramp * 1e15:.3f} fF, I_ch={best_corner_i_ch * 1e6:.3f} µA\n")

# Generate results for best K_slope
K_slope_opt = best_corner_global['K_slope_MV'] * 1e6
results = power_method(VDD, K_slope_opt, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=False)
best_n = min(results.keys(), key=lambda n: results[n]['p_corner'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_multi_panel_figsize(1, 2))
plot_all_bits(ax1, results, "All Bit Configurations")
plot_power_breakdown(ax2, results, best_n, 
                     f'Power Breakdown: {best_n} bits, K_slope={best_corner_global["K_slope_MV"]:.0f} MV/s',
                     resolution_s=res)
plt.tight_layout()
fig.savefig(SAVE_DIR / "all_bits_and_best_case.png", dpi=300, bbox_inches='tight')
print(f"Saved: all_bits_and_best_case.png")

# Additional plot: K_slope = 300 MV/s, only bits 8-12
K_TARGET = 300e6
results_ks300 = power_method(VDD, K_TARGET, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=False)
if results_ks300:
    fig_ks300, ax_ks300 = plt.subplots()
    plot_all_bits(ax_ks300, results_ks300, f"K_slope = {K_TARGET/1e6:.0f} MV/s", bits_to_plot=range(8, 13))
    plt.tight_layout()
    fig_ks300.savefig(SAVE_DIR / f"all_bits_Ks_{int(K_TARGET/1e6)}_8to12.png", dpi=300, bbox_inches='tight')
    print(f"Saved: all_bits_Ks_{int(K_TARGET/1e6)}_8to12.png")

# Frontier curves for different K_slopes
fig2, ax = plt.subplots()
plot_frontier_with_all_k_slopes(ax, K_slopes, VDD, C_load, k_lim_val, Cu0, res, NBITS,
                                use_bit_extension=False, title="Corner Point Frontier (Regular Mode)")
fig2.savefig(SAVE_DIR / "frontier_curves_kslopes.png", dpi=300, bbox_inches='tight')
print(f"Saved: frontier_curves_kslopes.png")

# Optimal frontier
fig3, ax = plt.subplots()
plot_optimal_frontier(ax, min_points, title=f'Optimal Power Points')
fig3.savefig(SAVE_DIR / "optimal_design_frontier.png", dpi=300, bbox_inches='tight')
print(f"Saved: optimal_design_frontier.png")

# ============================================================
# BIT EXTENSION MODE OPTIMIZATION
# ============================================================

print("\n" + "="*120)
print("BIT EXTENSION MODE - FINDING OPTIMAL K_slope")
print("="*120 + "\n")

min_points_bext = sweep_k_slopes(K_slopes, VDD, C_load, k_lim_val, Cu0, res, NBITS, use_bit_extension=True)
df_min_bext = pd.DataFrame(min_points_bext)
best_corner_global_bext = min(min_points_bext, key=lambda x: x['p_uw'])

print(f"Best K_slope: {best_corner_global_bext['K_slope_MV']:.1f} MV/s, {int(best_corner_global_bext['n'])} bits, "
      f"T_DTC={best_corner_global_bext['t_ns']:.3f} ns, Power={best_corner_global_bext['p_uw']:.2f} µW\n")
best_corner_i_ch_bext = best_corner_global_bext['I_ch_corner']
best_corner_c_ramp_bext = best_corner_global_bext['C_ramp_corner']
print(f"Best-case corner values: C_ramp={best_corner_c_ramp_bext * 1e15:.3f} fF, I_ch={best_corner_i_ch_bext * 1e6:.3f} µA\n")

# Generate results for best K_slope (bit extension)
K_slope_opt_bext = best_corner_global_bext['K_slope_MV'] * 1e6
results_bext_opt = power_method(VDD, K_slope_opt_bext, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=True)
best_n_bext_opt = min(results_bext_opt.keys(), key=lambda n: results_bext_opt[n]['p_corner'])

# Split combined bit-extension figure into two separate standard-size plots
fig_bext_all, ax_bext_all = plt.subplots(1, 1, figsize=_multi_panel_figsize(1, 1))
plot_all_bits(
    ax_bext_all,
    results_bext_opt,
    f"Configuration: $K_{{slope}}$={best_corner_global_bext['K_slope_MV']:.0f} MV/s",
)
plt.tight_layout()
fig_bext_all.savefig(SAVE_DIR / "all_bits_bitextension_optimized.png", dpi=300, bbox_inches='tight')
print(f"Saved: all_bits_bitextension_optimized.png")

fig_bext_best, ax_bext_best = plt.subplots(1, 1, figsize=_multi_panel_figsize(1, 1))
plot_power_breakdown(ax_bext_best, results_bext_opt, best_n_bext_opt,
                     f'Power Breakdown: {best_n_bext_opt} bits',
                     resolution_s=res)
plt.tight_layout()
fig_bext_best.savefig(SAVE_DIR / "best_case_bitextension_optimized.png", dpi=300, bbox_inches='tight')
print(f"Saved: best_case_bitextension_optimized.png")

# Frontier curves for bit extension
fig_bext_frontier, ax_bext_frontier = plt.subplots()
plot_frontier_with_all_k_slopes(ax_bext_frontier, K_slopes, VDD, C_load, k_lim_val, Cu0, res, NBITS,
                                use_bit_extension=True, title=r"$K_{slope}$ optimization")
fig_bext_frontier.savefig(SAVE_DIR / "frontier_curves_kslopes_bitextension.png", dpi=300, bbox_inches='tight')
print(f"Saved: frontier_curves_kslopes_bitextension.png")

# Optimal frontier (bit extension)
fig_bext_opt_frontier, ax_bext_opt_frontier = plt.subplots()
plot_optimal_frontier(ax_bext_opt_frontier, min_points_bext,
                      title=f'Optimal Power Points')
fig_bext_opt_frontier.savefig(SAVE_DIR / "optimal_design_frontier_bitextension.png", dpi=300, bbox_inches='tight')
print(f"Saved: optimal_design_frontier_bitextension.png")

# ============================================================
# COMPARISON PLOT
# ============================================================

fig_comparison, ax_comp = plt.subplots()

t_mins_regular = [m['t_ns'] for m in min_points]
p_mins_regular = [m['p_uw'] for m in min_points]
t_mins_bext = [m['t_ns'] for m in min_points_bext]
p_mins_bext = [m['p_uw'] for m in min_points_bext]

ax_comp.plot(t_mins_regular, p_mins_regular, 'o-', markersize=8, linewidth=0.2,
             color='#1F77B4', markeredgewidth=2, markeredgecolor='white',
             label='Regular Mode', zorder=5)
ax_comp.plot(t_mins_bext, p_mins_bext, 's-', markersize=8, linewidth=0.2,
             color='#FF7F0E', markeredgewidth=2, markeredgecolor='white',
             label='Bit Extension Mode', zorder=5)

# Mark global minimums
best_idx_regular = min_points.index(best_corner_global)
best_idx_bext = min_points_bext.index(best_corner_global_bext)

ax_comp.scatter([t_mins_regular[best_idx_regular]], [p_mins_regular[best_idx_regular]],
                marker='*', s=100, color='gold', edgecolors='red', linewidths=0.2, zorder=25,
                label='Global Min (Regular)')
ax_comp.scatter([t_mins_bext[best_idx_bext]], [p_mins_bext[best_idx_bext]],
                marker='*', s=100, color='lime', edgecolors='darkgreen', linewidths=0.2, zorder=25,
                label='Global Min (Bit Extension)')

ax_comp.set_yscale('log')
ax_comp.set_xlabel(r'Optimal $\Delta t$ [ns]', fontsize=LABEL_SIZE, fontweight='bold')
ax_comp.set_ylabel('Minimum Power [µW]', fontsize=LABEL_SIZE, fontweight='bold')
ax_comp.set_title(
    f'Comparison: Regular vs Bit Extension Mode ($k_{{lim}}$={k_lim_val})',
    fontsize=TITLE_SIZE,
    fontweight='bold',
    pad=10,
)
ax_comp.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax_comp.set_axisbelow(True)
ax_comp.legend(fontsize=LEGEND_SIZE, framealpha=0.96, edgecolor='black', loc='best')

plt.tight_layout()
fig_comparison.savefig(SAVE_DIR / "comparison_regular_vs_bitextension.png", dpi=300, bbox_inches='tight')
print(f"Saved: comparison_regular_vs_bitextension.png")

# ============================================================
# SAVE SUMMARY FILE
# ============================================================

output_text = []
output_text.append("=" * 120)
output_text.append("OPTIMAL DESIGN POINTS SUMMARY")
output_text.append("=" * 120)
output_text.append("")

output_text.append("REGULAR MODE - GLOBAL MINIMUM")
output_text.append("-" * 120)
for key, val in [("K_slope", f"{best_corner_global['K_slope_MV']:.1f} MV/s"),
                 ("Bits", f"{int(best_corner_global['n'])}"),
                 ("T_DTC", f"{best_corner_global['t_ns']:.3f} ns"),
                 ("C_ramp", f"{best_corner_global['C_ramp_corner'] * 1e15:.3f} fF"),
                 ("I_ch", f"{best_corner_global['I_ch_corner'] * 1e6:.3f} µA"),
                 ("C0", f"{best_corner_global['C0_fF']:.2f} fF"),
                 ("Ca", f"{best_corner_global['Ca_fF']:.2f} fF"),
                 ("P_ana", f"{best_corner_global['p_ana_uw']:.3f} µW"),
                 ("P_dig", f"{best_corner_global['p_dig_uw']:.3f} µW"),
                 ("P_total", f"{best_corner_global['p_uw']:.3f} µW")]:
    output_text.append(f"{key:15} = {val}")

output_text.append("")
output_text.append("BIT EXTENSION MODE - GLOBAL MINIMUM")
output_text.append("-" * 120)
for key, val in [("K_slope", f"{best_corner_global_bext['K_slope_MV']:.1f} MV/s"),
                 ("Bits", f"{int(best_corner_global_bext['n'])}"),
                 ("T_DTC", f"{best_corner_global_bext['t_ns']:.3f} ns"),
                 ("C_ramp", f"{best_corner_global_bext['C_ramp_corner'] * 1e15:.3f} fF"),
                 ("I_ch", f"{best_corner_global_bext['I_ch_corner'] * 1e6:.3f} µA"),
                 ("C0", f"{best_corner_global_bext['C0_fF']:.2f} fF"),
                 ("Ca", f"{best_corner_global_bext['Ca_fF']:.2f} fF"),
                 ("P_ana", f"{best_corner_global_bext['p_ana_uw']:.3f} µW"),
                 ("P_dig", f"{best_corner_global_bext['p_dig_uw']:.3f} µW"),
                 ("P_total", f"{best_corner_global_bext['p_uw']:.3f} µW")]:
    output_text.append(f"{key:15} = {val}")

output_text.append("")
output_text.append("=" * 120)
output_text.append("CIRCUIT PARAMETERS")
output_text.append("=" * 120)
for key, val in [("VDD", f"{VDD} V"), ("C_load", f"{C_load*1e15} fF"),
                 ("Cu0", f"{Cu0*1e15} fF"), ("Resolution", f"{res*1e12} ps"),
                 ("k_lim", f"{k_lim_val}")]:
    output_text.append(f"{key:15} = {val}")
output_text.append("=" * 120)

summary_path = SAVE_DIR / "optimal_points_summary.txt"
with open(summary_path, 'w') as f:
    f.write('\n'.join(output_text))

print(f"\nSaved: optimal_points_summary.txt")
print(f"\n✓ All results saved to: {SAVE_DIR.absolute()}")
