import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ===== PUBLICATION-READY PLOT STYLE =====
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.figsize": (10, 6), "lines.linewidth": 2.6, "lines.markersize": 4,
    "lines.markeredgewidth": 1.0, "grid.alpha": 0.6, "grid.color": "#b7b7b7",
    "grid.linestyle": "--", "grid.linewidth": 1.2, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.linewidth": 1.6, "axes.edgecolor": "black", "axes.facecolor": "white",
    "xtick.major.width": 1.4, "xtick.minor.width": 1.0,
    "ytick.major.width": 1.4, "ytick.minor.width": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": True, "legend.framealpha": 0.96,
    "legend.edgecolor": "black", "legend.fancybox": False,
})

# ============================================================
# CORE FUNCTIONS
# ============================================================

def get_ca_cdac(n, Cu0):
    """Calculate capacitance values based on bit width."""
    Cu = Cu0 * 2**((n-1)/2)
    Ca = Cu * (2**n - 1)  # Regular: full CDAC
    return Cu, Ca

def get_p_ana_coefficient(use_bit_extension=False):
    """Get analog power coefficient (divide factor)."""
    return 2 if use_bit_extension else 1

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
    f = 100e6  # 100 MHz 
    
    for n in range(n_start, NBITS + 1):
        n_actual = (n - 1) if use_bit_extension else n
        Cu, Ca = get_ca_cdac(n_actual, Cu0)
        
        Co_res = (Cu * VDD) / (K_slope * res) - Ca 
        Co_min = max(1 * Ca, Co_res)
        
        if C_load==0:
            if T_DTC_max >10e-9:
                T_list = np.linspace(10e-9, T_DTC_max, 200)
            else:
                return {}  # No valid T_DTC range for zero load
        else:
            T_list = np.linspace(res, T_DTC_max, 200)
        P_total = np.zeros(len(T_list))
        P_cnt = np.zeros(len(T_list))
        P_ana = np.zeros(len(T_list))
        
        t_corner = 2*(Ca * VDD) / (K_slope * (Co_min + Ca)) if use_bit_extension else (Ca * VDD) / (K_slope * (Co_min + Ca))
        p_corner_ana = Co_min * Ca/(Co_min + Ca)*VDD**2*f/p_ana_coeff 
        p_corner_cnt = (C_load * VDD**2) / t_corner 

        for i, t_DTC in enumerate(T_list):
            C0_req = 2*(Ca * VDD) / (K_slope * t_DTC) - Ca if use_bit_extension else (Ca * VDD) / (K_slope * t_DTC) - Ca
            C0 = max(C0_req, Co_min)
            P_cnt[i] = (C_load * VDD**2) / t_DTC
            P_ana[i] = C0 * Ca/(C0 + Ca)*VDD**2*f/p_ana_coeff 
            P_total[i] = P_cnt[i] + P_ana[i]
        
        P_dict[n] = {
            "T_list": T_list, "P_list": P_total, "P_cnt": P_cnt, "P_ana": P_ana,
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

    f= 100e6  # 100 MHz
    
    for n in range(n_start, NBITS + 1):
        n_actual = (n - 1) if use_bit_extension else n
        Cu, Ca = get_ca_cdac(n_actual, Cu0)
        
        Co_res = (Cu * VDD) / (K_slope * res) - Ca
        Co_min = max(k_lim * Ca, Co_res)
        t_corner = 2*(Ca * VDD) / (K_slope * (Co_min + Ca)) if use_bit_extension else (Ca * VDD) / (K_slope * (Co_min + Ca))
        
        # Redundancy check
        if n >= n_start_check and (t_corner <= previous_t * 1.001):
            break
        
        p_ana = Co_min * Ca/(Co_min + Ca)*VDD**2*f/p_ana_coeff
        p_dig = (C_load * VDD**2) / t_corner
        
        corners.append({
            "n": n, "t_ns": t_corner * 1e9,
            "C0_fF": Co_min * 1e15, "Ca_fF": Ca * 1e15,
            "p_uw": (p_ana + p_dig) * 1e6,
            "p_ana_uw": p_ana * 1e6, "p_dig_uw": p_dig * 1e6,
            "K_slope_MV": K_slope / 1e6, "k_lim": k_lim
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
    
    ax.semilogy(t_ns, best_data["P_list"] * 1e6, 'k', linewidth=2.6, label='Total Power')
    ax.semilogy(t_ns, best_data["P_ana"] * 1e6, '--', color='#D62728', linewidth=2.2,
                label='$P_{ana}$ (Analog/DTC)', alpha=0.8)
    ax.semilogy(t_ns, best_data["P_cnt"] * 1e6, '--', color='#1F77B4', linewidth=2.2,
                label='$P_{cnt}$ (Digital/Counter)', alpha=0.8)
    
    ax.scatter(best_data["t_corner"]*1e9, best_data["p_corner"]*1e6, color='red', s=120,
               edgecolors='black', linewidths=2, zorder=10, marker='o')
    x_res_ns = best_data["t_corner"] * 1e9
    ax.axvspan(x_res_ns, np.max(t_ns), color='red', alpha=0.08, zorder=0)
    ax.axvline(x_res_ns, linestyle='--', color='#7F7F7F', linewidth=2.0,
               label='Resolution not met')
    ax.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
    ax.set_title(title_suffix, fontsize=14, fontweight='bold', pad=12)    
    ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')

def plot_all_bits(ax, data_dict, title=""):
    """Plot all bit configurations."""
    t_corners, p_corners = [], []
    for n, data in data_dict.items():
        ax.semilogy(data["T_list"]*1e9, data["P_list"]*1e6, alpha=0.5, linewidth=2.2, label=f'{n} bits')
        t_corners.append(data["t_corner"] * 1e9)
        p_corners.append(data["p_corner"] * 1e6)
    
    ax.scatter(t_corners, p_corners, color='red', s=60, zorder=5, edgecolors='black',
               linewidths=1.5, label='Corner Points')
    
    ax.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.96, edgecolor='black', ncol=2)

def plot_frontier_with_all_k_slopes(ax, K_slopes, VDD, C_load, k_lim, Cu0, res, NBITS, 
                                     use_bit_extension=False, title="", colors=None):
    """Plot frontier curves for different K_slope values."""
    if colors is None:
        colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    
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
        
        color = colors[idx % len(colors)]
        ax.plot(t_vals, p_vals, 'o-', color=color, linewidth=2.4, markersize=8,
                markeredgewidth=1.5, markeredgecolor='white',
                label=f'$K_{{slope}}$ = {k_s/1e6:.0f} MV/s')
        
        for i, n in enumerate(n_vals):
            ax.annotate(f"{n}b", (t_vals[i], p_vals[i]), textcoords="offset points",
                        xytext=(0,12), ha='center', fontsize=9, fontweight='bold')
        
        t_max.append(max(t_vals))
        if max(t_vals) > 2:
            p_max.append(p_vals[2])  # 5th point (n=5) as reference for max power
            p_min.append(min(p_vals))

    ax.set_yscale('log')
    ax.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
    if t_max:
        ax.set_xlim(left=1.0, right=max(t_max) * 1.05)
    if p_max:
        y_top = min(max(p_max) * 1.2, 200)
        y_min = max(min(p_min) * 0.8, 20)
        ax.set_ylim(bottom=y_min, top=max(y_top, 1e-3))

def plot_optimal_frontier(ax, min_points, title="", mode_label=""):
    """Plot optimal frontier with star for global minimum."""
    t_mins = [m['t_ns'] for m in min_points]
    p_mins = [m['p_uw'] for m in min_points]
    k_labels = [m['K_slope_MV'] for m in min_points]
    n_labels = [m['n'] for m in min_points]
    
    ax.plot(t_mins, p_mins, 'o', markersize=10, linewidth=2.6, color='#2CA02C',
            markeredgewidth=2, markeredgecolor='white', label='Optimal Design Frontier')
    
    # Mark global minimum
    best_idx = min_points.index(min(min_points, key=lambda x: x['p_uw']))
    ax.scatter([t_mins[best_idx]], [p_mins[best_idx]], marker='*', s=800,
               color='gold', edgecolors='red', linewidths=2, zorder=20, label='Global Minimum')
    
    # Annotate points
    for i in range(len(min_points)):
        ax.annotate(f"{n_labels[i]}b\n{k_labels[i]:.0f} MV/s",
                    (t_mins[i], p_mins[i]), textcoords="offset points", xytext=(0,15),
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_yscale('log')
    ax.set_xlabel('Optimal $T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Minimum Power [µW]', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, framealpha=0.96, edgecolor='black')

# ============================================================
# CONFIGURATION SECTION
# ============================================================

SAVE_DIR = Path("C:\\Users\\zipar\\OneDrive - Delft University of Technology\\Second Year\\MEP\\plots_optimization")
VDD, K_slope, C_load, k_lim, Cu0, res, NBITS = 1.1, 380e6, 120e-15, 300e6, 44e-18, 5e-12, 11
K_slopes = np.linspace(100e6, 300e6, 10)

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

# Generate results for best K_slope
K_slope_opt = best_corner_global['K_slope_MV'] * 1e6
results = power_method(VDD, K_slope_opt, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=False)
best_n = min(results.keys(), key=lambda n: results[n]['p_corner'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_all_bits(ax1, results, "All Bit Configurations")
plot_power_breakdown(ax2, results, best_n, 
                     f'Power Breakdown: {best_n} bits, K_slope={best_corner_global["K_slope_MV"]:.0f} MV/s',
                     resolution_s=res)
plt.tight_layout()
fig.savefig(SAVE_DIR / "all_bits_and_best_case.png", dpi=300, bbox_inches='tight')
print(f"Saved: all_bits_and_best_case.png")

# Frontier curves for different K_slopes
fig2, ax = plt.subplots(figsize=(10, 7))
plot_frontier_with_all_k_slopes(ax, K_slopes, VDD, C_load, k_lim_val, Cu0, res, NBITS,
                                use_bit_extension=False, title="Corner Point Frontier (Regular Mode)")
fig2.savefig(SAVE_DIR / "frontier_curves_kslopes.png", dpi=300, bbox_inches='tight')
print(f"Saved: frontier_curves_kslopes.png")

# Optimal frontier
fig3, ax = plt.subplots(figsize=(10, 7))
plot_optimal_frontier(ax, min_points, title=f'Optimal Power Points (Regular Mode, $k_{{lim}}$={k_lim_val})')
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

# Generate results for best K_slope (bit extension)
K_slope_opt_bext = best_corner_global_bext['K_slope_MV'] * 1e6
results_bext_opt = power_method(VDD, K_slope_opt_bext, C_load, k_lim, Cu0, res, NBITS, use_bit_extension=True)
best_n_bext_opt = min(results_bext_opt.keys(), key=lambda n: results_bext_opt[n]['p_corner'])

fig_bext_opt, (ax1_be, ax2_be) = plt.subplots(1, 2, figsize=(14, 6))
plot_all_bits(ax1_be, results_bext_opt, "All Bit Configurations (Bit Extension - Optimized)")
plot_power_breakdown(ax2_be, results_bext_opt, best_n_bext_opt,
                     f'Power Breakdown (Bit Extension): {best_n_bext_opt} bits, K_slope={best_corner_global_bext["K_slope_MV"]:.0f} MV/s',
                     resolution_s=res)
plt.tight_layout()
fig_bext_opt.savefig(SAVE_DIR / "all_bits_and_best_case_bitextension_optimized.png", dpi=300, bbox_inches='tight')
print(f"Saved: all_bits_and_best_case_bitextension_optimized.png")

# Frontier curves for bit extension
fig_bext_frontier, ax_bext_frontier = plt.subplots(figsize=(10, 7))
plot_frontier_with_all_k_slopes(ax_bext_frontier, K_slopes, VDD, C_load, k_lim_val, Cu0, res, NBITS,
                                use_bit_extension=True, title="Corner Point Frontier (Bit Extension Mode)")
fig_bext_frontier.savefig(SAVE_DIR / "frontier_curves_kslopes_bitextension.png", dpi=300, bbox_inches='tight')
print(f"Saved: frontier_curves_kslopes_bitextension.png")

# Optimal frontier (bit extension)
fig_bext_opt_frontier, ax_bext_opt_frontier = plt.subplots(figsize=(10, 7))
plot_optimal_frontier(ax_bext_opt_frontier, min_points_bext,
                      title=f'Optimal Power Points (Bit Extension Mode, $k_{{lim}}$={k_lim_val})')
fig_bext_opt_frontier.savefig(SAVE_DIR / "optimal_design_frontier_bitextension.png", dpi=300, bbox_inches='tight')
print(f"Saved: optimal_design_frontier_bitextension.png")

# ============================================================
# COMPARISON PLOT
# ============================================================

fig_comparison, ax_comp = plt.subplots(figsize=(12, 7))

t_mins_regular = [m['t_ns'] for m in min_points]
p_mins_regular = [m['p_uw'] for m in min_points]
t_mins_bext = [m['t_ns'] for m in min_points_bext]
p_mins_bext = [m['p_uw'] for m in min_points_bext]

ax_comp.plot(t_mins_regular, p_mins_regular, 'o-', markersize=11, linewidth=2.8,
             color='#1F77B4', markeredgewidth=2, markeredgecolor='white',
             label='Regular Mode', zorder=5)
ax_comp.plot(t_mins_bext, p_mins_bext, 's-', markersize=11, linewidth=2.8,
             color='#FF7F0E', markeredgewidth=2, markeredgecolor='white',
             label='Bit Extension Mode', zorder=5)

# Mark global minimums
best_idx_regular = min_points.index(best_corner_global)
best_idx_bext = min_points_bext.index(best_corner_global_bext)

ax_comp.scatter([t_mins_regular[best_idx_regular]], [p_mins_regular[best_idx_regular]],
                marker='*', s=1000, color='gold', edgecolors='red', linewidths=2.5, zorder=25,
                label='Global Min (Regular)')
ax_comp.scatter([t_mins_bext[best_idx_bext]], [p_mins_bext[best_idx_bext]],
                marker='*', s=1000, color='lime', edgecolors='darkgreen', linewidths=2.5, zorder=25,
                label='Global Min (Bit Extension)')

ax_comp.set_yscale('log')
ax_comp.set_xlabel('Optimal $T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax_comp.set_ylabel('Minimum Power [µW]', fontsize=12, fontweight='bold')
ax_comp.set_title(f'Comparison: Regular vs Bit Extension Mode ($k_{{lim}}$={k_lim_val})',
                  fontsize=14, fontweight='bold', pad=12)
ax_comp.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax_comp.set_axisbelow(True)
ax_comp.legend(fontsize=11, framealpha=0.96, edgecolor='black', loc='best')

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
