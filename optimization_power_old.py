import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
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

def power_method(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS):
    P_dict = {}
    T_DTC_max = VDD / K_slope
    
    for n in range(1, NBITS + 1):
        # Mismatch scaling
        Cu = Cu0 * (2**((n-1)/2))
        Ca = Cu * (2**n - 1) 
        
        # Resolution Floor: The C0 needed to hit exactly 'res'
        # Derived from: res = (Cu * VDD) / ((C0 + Ca) * K_slope)
        Co_res = (Cu * VDD) / (K_slope * res) - Ca
        Co_min = max(k_lim*Ca, Co_res)
        
        T_list = np.linspace(100e-12, T_DTC_max, 200)

        P_total = np.zeros(len(T_list))
        P_cnt = np.zeros(len(T_list))
        P_ana = np.zeros(len(T_list))
        
        # Calculate the theoretical t_corner
        # It's the time where C0_req matches Co_min
        t_corner = (Ca * VDD) / (K_slope * (Co_min + Ca))
        p_corner = (C_load * VDD**2) / t_corner +  Co_min * K_slope * VDD/4

        for i, t_DTC in enumerate(T_list):
            C0_req = (Ca * VDD) / (K_slope * t_DTC) - Ca
            C0 = max(C0_req, Co_min)
            
            P_cnt[i] = (C_load * VDD**2) / t_DTC
            P_ana[i] =  C0 * K_slope * VDD / 4  # Updated formula for P_ana
            P_total[i] = P_cnt[i] + P_ana[i]

        P_dict[n] = {
            "T_list": T_list, 
            "P_list": P_total, 
            "P_cnt": P_cnt, 
            "P_ana": P_ana, 
            "p_corner": p_corner, 
            "t_corner": t_corner
        }
    
    return P_dict

def power_method_bitextension(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS):
    P_dict = {}
    T_DTC_max = 2*VDD / K_slope  # Allow for longer T_DTC due to bit extension
    
    for n in range(3, NBITS + 1):
        n = n - 1 #bit extension, so the CDAC is in reality wiht one bit less
        # Mismatch scaling
        Cu = Cu0 * (2**((n-1)/2))
        Ca = Cu * (2**n - 2)  # One bit less in the CDAC, so we subtract 2 instead of 1
        
        # Resolution Floor: The C0 needed to hit exactly 'res'
        # Derived from: res = (Cu * VDD) / ((C0 + Ca) * K_slope)
        Co_res = 2*Ca*VDD / (K_slope * res*(2**(n+1)-1)) - Ca
        Co_min = max(k_lim*Ca, Co_res)
        
        T_list = np.linspace(100e-12, T_DTC_max, 200)

        P_total = np.zeros(len(T_list))
        P_cnt = np.zeros(len(T_list))
        P_ana = np.zeros(len(T_list))
        
        # Calculate the theoretical t_corner
        # It's the time where C0_req matches Co_min
        t_corner = (2 * Ca * VDD) / (K_slope * (Co_min + Ca))
        p_corner = (C_load * VDD**2) / t_corner +  Co_min * K_slope * VDD/8

        for i, t_DTC in enumerate(T_list):
            C0_req = 2*(Ca * VDD) / (K_slope * t_DTC) - Ca
            C0 = max(C0_req, Co_min)
            
            P_cnt[i] = (C_load * VDD**2) / t_DTC
            P_ana[i] =  C0 * K_slope * VDD / 8  # Updated formula for P_ana (bit extension)     
            P_total[i] = P_cnt[i] + P_ana[i]

        P_dict[n] = {
            "T_list": T_list, 
            "P_list": P_total, 
            "P_cnt": P_cnt, 
            "P_ana": P_ana, 
            "p_corner": p_corner, 
            "t_corner": t_corner
        }
    
    return P_dict

def get_corners_only(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS):
    """Calculates corner points and stops if bits become redundant."""
    corners = []
    previous_t = 0
    
    for n in range(2, NBITS + 1):
        Cu = Cu0 * (2**((n-1)/2))
        Ca = Cu * (2**n - 1)  # One bit less in the CDAC, so we subtract 2 instead of 1
        
        Co_res = (Cu * VDD) / (K_slope * res) - Ca
        Co_min = max(k_lim * Ca, Co_res)
        
        t_corner = (Ca * VDD) / (K_slope * (Co_min + Ca))
        
        # --- REDUNDANCY CHECK ---
        # If t_corner is almost the same as previous (saturated) 
        # or if the power jump is massive, we stop.
        if n > 1 and (t_corner <= previous_t * 1.001):
            break
            
        p_ana = Co_min * K_slope * VDD / 4
        p_dig = (C_load * VDD**2) / t_corner
        
        corners.append({
            "n": n,
            "t_ns": t_corner * 1e9,
            "C0_fF": Co_min * 1e15,
            "Ca_fF": Ca * 1e15,
            "p_uw": (p_ana + p_dig) * 1e6,
            "p_ana_uw": p_ana * 1e6,
            "p_dig_uw": p_dig * 1e6,
            "K_slope_MV": K_slope / 1e6,
            "k_lim": k_lim
        })
        previous_t = t_corner
        
    return corners

def get_corners_only_bitextension(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS):
    """Calculates corner points for bit extension and stops if bits become redundant."""
    corners = []
    previous_t = 0
    
    for n in range(3, NBITS + 1):
        n_actual = n - 1  # Bit extension: CDAC has one bit less
        Cu = Cu0 * (2**((n_actual-1)/2))
        Ca = Cu * (2**n_actual - 2)  # One bit less for bit extension
        
        Co_res = 2*Ca*VDD / (K_slope * res*(2**(n+1)-1)) - Ca
        Co_min = max(k_lim * Ca, Co_res)
        
        t_corner = 2*(Ca * VDD) / (K_slope * (Co_min + Ca))
        
        # --- REDUNDANCY CHECK ---
        if n > 3 and (t_corner <= previous_t * 1.001):
            break
            
        p_ana = Co_min * K_slope * VDD / 8  # Different power formula for bit extension
        p_dig = (C_load * VDD**2) / t_corner
        
        corners.append({
            "n": n_actual,
            "t_ns": t_corner * 1e9,
            "C0_fF": Co_min * 1e15,
            "Ca_fF": Ca * 1e15,
            "p_uw": (p_ana + p_dig) * 1e6,
            "p_ana_uw": p_ana * 1e6,
            "p_dig_uw": p_dig * 1e6,
            "K_slope_MV": K_slope / 1e6,
            "k_lim": k_lim
        })
        previous_t = t_corner
        
    return corners

# ============================================================
# ⚙️ CONFIGURATION SECTION - SET YOUR PARAMETERS HERE
# ============================================================

# 📁 Save Directory - Change this to your desired output folder
SAVE_DIR = Path("C:\\Users\\zipar\\OneDrive - Delft University of Technology\\Second Year\\MEP\\plots_optimization")  # CUSTOMIZE THIS PATH

# Circuit Parameters
VDD, K_slope, C_load, k_lim, Cu0, res, NBITS = 1.1, 380e6, 10e-15, 1.0, 1e-15, 50e-12, 10

# K_slope sweep space for optimization (in V/s)
K_slopes = [50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6, 450e6]
#K_slopes = [340e6, 360e6, 380e6]
k_lim_val = 1.0

# ============================================================

# Create save directory if it doesn't exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"📂 Saving plots and results to: {SAVE_DIR.absolute()}\n")

# --- SEARCH FOR OPTIMAL K_slope (GLOBAL MINIMUM POWER) ---
min_points = []
for k_s in K_slopes:
    all_corners = get_corners_only(VDD, k_s, C_load, k_lim_val, Cu0, res, NBITS)
    best_corner = min(all_corners, key=lambda x: x['p_uw'])
    min_points.append(best_corner)

df_min = pd.DataFrame(min_points)

best_corner_global = min(min_points, key=lambda x: x['p_uw'])
best_k_slope = best_corner_global['K_slope_MV'] * 1e6

print("Best K_slope search:")
print(f"  K_slope = {best_corner_global['K_slope_MV']:.1f} MV/s")
print(f"  Best n = {best_corner_global['n']}")
print(f"  Best T_DTC = {best_corner_global['t_ns']:.3f} ns")
print(f"  Min Power = {best_corner_global['p_uw']:.2f} µW\n")

# Use the optimal K_slope for the power-method plots
K_slope = best_k_slope
results = power_method(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS)

# --- FIND THE BEST N (for optimal K_slope) ---
best_n = min(results.keys(), key=lambda n: results[n]['p_corner'])
best_data = results[best_n]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Original All-Bits Plot
t_corners = []
p_corners = []
for n, data in results.items():
    ax1.semilogy(data["T_list"]*1e9, data["P_list"]*1e6, alpha=0.5, linewidth=2.2, label=f'{n} bits')
    t_corners.append(data["t_corner"] * 1e9)
    p_corners.append(data["p_corner"] * 1e6)

ax1.scatter(t_corners, p_corners, color='red', s=60, zorder=5, edgecolors='black', linewidths=1.5, label='Corner Points')
ax1.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax1.set_title('All Bit Configurations', fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax1.set_axisbelow(True)
ax1.legend(fontsize=9, framealpha=0.96, edgecolor='black', ncol=2)

# Plot 2: Breakdown for the BEST Case (best_n)
t_ns = best_data["T_list"] * 1e9
ax2.semilogy(t_ns, best_data["P_list"] * 1e6, 'k', linewidth=2.6, label='Total Power')
ax2.semilogy(t_ns, best_data["P_ana"] * 1e6, '--', color='#D62728', linewidth=2.2, label='$P_{ana}$ (Analog/DTC)', alpha=0.8)
ax2.semilogy(t_ns, best_data["P_cnt"] * 1e6, '--', color='#1F77B4', linewidth=2.2, label='$P_{cnt}$ (Digital/Counter)', alpha=0.8)

# Mark the corner point for the best case
ax2.scatter(best_data["t_corner"]*1e9, best_data["p_corner"]*1e6, color='red', s=120, 
           edgecolors='black', linewidths=2, zorder=10, marker='o')

ax2.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax2.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax2.set_title(
    f'Power Breakdown for Best Case: {best_n} bits, K_slope={best_corner_global["K_slope_MV"]:.0f} MV/s',
    fontsize=14,
    fontweight='bold',
    pad=12
)
ax2.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax2.set_axisbelow(True)
ax2.legend(fontsize=10, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path1 = SAVE_DIR / "all_bits_and_best_case.png"
fig.savefig(save_path1, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path1}")

print(f"\nOptimal Configuration: {best_n} bits")
print(f"Minimum Power at Corner: {best_data['p_corner']*1e6:.2f} µW")
print(f"Optimal T_DTC: {best_data['t_corner']*1e9:.3f} ns")

# --- BIT EXTENSION TEST: power_method_bitextension ---
results_bitext = power_method_bitextension(VDD, K_slope, C_load, k_lim, Cu0, res, NBITS)
best_n_bitext = min(results_bitext.keys(), key=lambda n: results_bitext[n]['p_corner'])
best_data_bitext = results_bitext[best_n_bitext]

fig_bitext, (ax1_be, ax2_be) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All-bits plot (bit extension)
t_corners_be = []
p_corners_be = []
for n, data in results_bitext.items():
    ax1_be.semilogy(data["T_list"]*1e9, data["P_list"]*1e6, alpha=0.5, linewidth=2.2, label=f'{n} bits')
    t_corners_be.append(data["t_corner"] * 1e9)
    p_corners_be.append(data["p_corner"] * 1e6)

ax1_be.scatter(t_corners_be, p_corners_be, color='red', s=60, zorder=5, edgecolors='black', linewidths=1.5, label='Corner Points')
ax1_be.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax1_be.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax1_be.set_title('All Bit Configurations (Bit Extension)', fontsize=14, fontweight='bold', pad=12)
ax1_be.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax1_be.set_axisbelow(True)
ax1_be.legend(fontsize=9, framealpha=0.96, edgecolor='black', ncol=2)

# Plot 2: Breakdown for the best case (bit extension)
t_ns_be = best_data_bitext["T_list"] * 1e9
ax2_be.semilogy(t_ns_be, best_data_bitext["P_list"] * 1e6, 'k', linewidth=2.6, label='Total Power')
ax2_be.semilogy(t_ns_be, best_data_bitext["P_ana"] * 1e6, '--', color='#D62728', linewidth=2.2, label='$P_{ana}$ (Analog/DTC)', alpha=0.8)
ax2_be.semilogy(t_ns_be, best_data_bitext["P_cnt"] * 1e6, '--', color='#1F77B4', linewidth=2.2, label='$P_{cnt}$ (Digital/Counter)', alpha=0.8)

ax2_be.scatter(best_data_bitext["t_corner"]*1e9, best_data_bitext["p_corner"]*1e6, color='red', s=120,
               edgecolors='black', linewidths=2, zorder=10, marker='o')

ax2_be.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax2_be.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax2_be.set_title(
    f'Power Breakdown (Bit Extension): {best_n_bitext} bits, K_slope={best_corner_global["K_slope_MV"]:.0f} MV/s',
    fontsize=14,
    fontweight='bold',
    pad=12
)
ax2_be.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax2_be.set_axisbelow(True)
ax2_be.legend(fontsize=10, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path_bitext = SAVE_DIR / "all_bits_and_best_case_bitextension.png"
fig_bitext.savefig(save_path_bitext, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_bitext}")

print(f"\nBit Extension Optimal Configuration: {best_n_bitext} bits")
print(f"Bit Extension Min Power at Corner: {best_data_bitext['p_corner']*1e6:.2f} µW")
print(f"Bit Extension Optimal T_DTC: {best_data_bitext['t_corner']*1e9:.3f} ns")

# ============================================================
# ⚙️ OPTIMIZATION FOR BIT EXTENSION - FIND BEST K_slope
# ============================================================
print("\n" + "="*120)
print("OPTIMIZATION FOR BIT EXTENSION - FINDING OPTIMAL K_slope")
print("="*120 + "\n")

# --- SEARCH FOR OPTIMAL K_slope (GLOBAL MINIMUM POWER) FOR BIT EXTENSION ---
min_points_bext = []
for k_s in K_slopes:
    all_corners_bext = get_corners_only_bitextension(VDD, k_s, C_load, k_lim_val, Cu0, res, NBITS)
    if all_corners_bext:  # Only if we have some corners
        best_corner_bext = min(all_corners_bext, key=lambda x: x['p_uw'])
        min_points_bext.append(best_corner_bext)

df_min_bext = pd.DataFrame(min_points_bext)

best_corner_global_bext = min(min_points_bext, key=lambda x: x['p_uw'])
best_k_slope_bext = best_corner_global_bext['K_slope_MV'] * 1e6

print("Best K_slope search (Bit Extension):")
print(f"  K_slope = {best_corner_global_bext['K_slope_MV']:.1f} MV/s")
print(f"  Best n = {best_corner_global_bext['n']}")
print(f"  Best T_DTC = {best_corner_global_bext['t_ns']:.3f} ns")
print(f"  Min Power = {best_corner_global_bext['p_uw']:.2f} µW\n")

# Use the optimal K_slope for the power-method plots for bit extension
K_slope_bext = best_k_slope_bext
results_bext_opt = power_method_bitextension(VDD, K_slope_bext, C_load, k_lim, Cu0, res, NBITS)

# --- FIND THE BEST N (for optimal K_slope in bit extension) ---
best_n_bext_opt = min(results_bext_opt.keys(), key=lambda n: results_bext_opt[n]['p_corner'])
best_data_bext_opt = results_bext_opt[best_n_bext_opt]

fig_bext_opt, (ax1_bext_opt, ax2_bext_opt) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All-bits plot (bit extension optimized)
t_corners_bext_opt = []
p_corners_bext_opt = []
for n, data in results_bext_opt.items():
    ax1_bext_opt.semilogy(data["T_list"]*1e9, data["P_list"]*1e6, alpha=0.5, linewidth=2.2, label=f'{n} bits')
    t_corners_bext_opt.append(data["t_corner"] * 1e9)
    p_corners_bext_opt.append(data["p_corner"] * 1e6)

ax1_bext_opt.scatter(t_corners_bext_opt, p_corners_bext_opt, color='red', s=60, zorder=5, edgecolors='black', linewidths=1.5, label='Corner Points')
ax1_bext_opt.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax1_bext_opt.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax1_bext_opt.set_title('All Bit Configurations (Bit Extension - Optimized)', fontsize=14, fontweight='bold', pad=12)
ax1_bext_opt.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax1_bext_opt.set_axisbelow(True)
ax1_bext_opt.legend(fontsize=9, framealpha=0.96, edgecolor='black', ncol=2)

# Plot 2: Breakdown for the best case (bit extension optimized)
t_ns_bext_opt = best_data_bext_opt["T_list"] * 1e9
ax2_bext_opt.semilogy(t_ns_bext_opt, best_data_bext_opt["P_list"] * 1e6, 'k', linewidth=2.6, label='Total Power')
ax2_bext_opt.semilogy(t_ns_bext_opt, best_data_bext_opt["P_ana"] * 1e6, '--', color='#D62728', linewidth=2.2, label='$P_{ana}$ (Analog/DTC)', alpha=0.8)
ax2_bext_opt.semilogy(t_ns_bext_opt, best_data_bext_opt["P_cnt"] * 1e6, '--', color='#1F77B4', linewidth=2.2, label='$P_{cnt}$ (Digital/Counter)', alpha=0.8)

ax2_bext_opt.scatter(best_data_bext_opt["t_corner"]*1e9, best_data_bext_opt["p_corner"]*1e6, color='red', s=120,
                     edgecolors='black', linewidths=2, zorder=10, marker='o')

ax2_bext_opt.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax2_bext_opt.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax2_bext_opt.set_title(
    f'Power Breakdown (Bit Extension - Optimized): {best_n_bext_opt} bits, K_slope={best_corner_global_bext["K_slope_MV"]:.0f} MV/s',
    fontsize=14,
    fontweight='bold',
    pad=12
)
ax2_bext_opt.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax2_bext_opt.set_axisbelow(True)
ax2_bext_opt.legend(fontsize=10, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path_bext_opt = SAVE_DIR / "all_bits_and_best_case_bitextension_optimized.png"
fig_bext_opt.savefig(save_path_bext_opt, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_bext_opt}")

print(f"\nBit Extension Optimal Configuration (Optimized K_slope): {best_n_bext_opt} bits")
print(f"Bit Extension Min Power at Corner: {best_data_bext_opt['p_corner']*1e6:.2f} µW")
print(f"Bit Extension Optimal T_DTC: {best_data_bext_opt['t_corner']*1e9:.3f} ns")

# --- FRONTIER CURVE FOR BIT EXTENSION ---
fig_bext_frontier, ax_bext_frontier = plt.subplots(figsize=(10, 7))

colors_bext = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']

for idx, k_s in enumerate(K_slopes):
    corners_bext = get_corners_only_bitextension(VDD, k_s, C_load, k_lim_val, Cu0, res, NBITS)
    if not corners_bext:
        continue
    
    # Extract data for plotting
    t_vals_bext = [c['t_ns'] for c in corners_bext]
    p_vals_bext = [c['p_uw'] for c in corners_bext]
    n_vals_bext = [c['n'] for c in corners_bext]
    
    # Plot the Frontier Curve
    color = colors_bext[idx % len(colors_bext)]
    line, = ax_bext_frontier.plot(t_vals_bext, p_vals_bext, 'o-', color=color, linewidth=2.4, markersize=8, 
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=f'$K_{{slope}}$ = {k_s/1e6:.0f} MV/s')
    
    # Annotate bit numbers on the curve
    for i, n in enumerate(n_vals_bext):
        ax_bext_frontier.annotate(f"{n}b", (t_vals_bext[i], p_vals_bext[i]), textcoords="offset points", 
                         xytext=(0,12), ha='center', fontsize=9, fontweight='bold')

    # Print Table for this specific K_slope (Bit Extension)
    df_bext = pd.DataFrame(corners_bext)
    print(f"\n--- Results for K_slope = {k_s/1e6} MV/s (Bit Extension) ---")
    print(df_bext[['n', 't_ns', 'p_ana_uw', 'p_dig_uw', 'p_uw']].to_string(index=False))

ax_bext_frontier.set_yscale('log')
ax_bext_frontier.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax_bext_frontier.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax_bext_frontier.set_title('Corner Point Frontier for Different $K_{slope}$ (Bit Extension)', fontsize=14, fontweight='bold', pad=12)
ax_bext_frontier.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax_bext_frontier.set_axisbelow(True)
ax_bext_frontier.legend(fontsize=10, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path_bext_frontier = SAVE_DIR / "frontier_curves_kslopes_bitextension.png"
fig_bext_frontier.savefig(save_path_bext_frontier, dpi=300, bbox_inches='tight')
print(f"\nSaved: {save_path_bext_frontier}")

# --- OPTIMAL DESIGN FRONTIER FOR BIT EXTENSION ---
fig_bext_opt_frontier, ax_bext_opt_frontier = plt.subplots(figsize=(10, 7))

t_mins_bext = [m['t_ns'] for m in min_points_bext]
p_mins_bext = [m['p_uw'] for m in min_points_bext]
k_labels_bext = [m['K_slope_MV'] for m in min_points_bext]
n_labels_bext = [m['n'] for m in min_points_bext]

# Plot the curve connecting the minimums
ax_bext_opt_frontier.plot(t_mins_bext, p_mins_bext, 'o', markersize=10, linewidth=2.6, color='#2CA02C',
        markeredgewidth=2, markeredgecolor='white', label='Optimal Design Frontier')

# Mark the global minimum with a star
best_idx_bext = min_points_bext.index(best_corner_global_bext)
ax_bext_opt_frontier.scatter([t_mins_bext[best_idx_bext]], [p_mins_bext[best_idx_bext]], 
                             marker='*', s=800, color='gold', edgecolors='red', linewidths=2, 
                             zorder=20, label='Global Minimum')

# Annotate each point with Bits and K_slope
for i in range(len(min_points_bext)):
    ax_bext_opt_frontier.annotate(f"{n_labels_bext[i]}b\n{k_labels_bext[i]:.0f} MV/s", 
                 (t_mins_bext[i], p_mins_bext[i]), 
                 textcoords="offset points", 
                 xytext=(0,15), ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

ax_bext_opt_frontier.set_yscale('log')
ax_bext_opt_frontier.set_xlabel('Optimal $T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax_bext_opt_frontier.set_ylabel('Minimum Power [µW]', fontsize=12, fontweight='bold')
ax_bext_opt_frontier.set_title(f'Optimal Power Points for Different $K_{{slope}}$ - Bit Extension ($k_{{lim}}$={k_lim_val})', 
             fontsize=14, fontweight='bold', pad=12)
ax_bext_opt_frontier.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax_bext_opt_frontier.set_axisbelow(True)
ax_bext_opt_frontier.legend(fontsize=11, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path_bext_opt_frontier = SAVE_DIR / "optimal_design_frontier_bitextension.png"
fig_bext_opt_frontier.savefig(save_path_bext_opt_frontier, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_bext_opt_frontier}")

# Scenarios to sweep (You can change K_slope or k_lim here)

fig2, ax = plt.subplots(figsize=(10, 7))

colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']

for idx, k_s in enumerate(K_slopes):
    corners = get_corners_only(VDD, k_s, C_load, k_lim_val, Cu0, res, NBITS)
    
    # Extract data for plotting
    t_vals = [c['t_ns'] for c in corners]
    p_vals = [c['p_uw'] for c in corners]
    n_vals = [c['n'] for c in corners]
    
    # Plot the Frontier Curve
    color = colors[idx % len(colors)]
    line, = ax.plot(t_vals, p_vals, 'o-', color=color, linewidth=2.4, markersize=8, 
                    markeredgewidth=1.5, markeredgecolor='white',
                    label=f'$K_{{slope}}$ = {k_s/1e6:.0f} MV/s')
    
    # Annotate bit numbers on the curve
    for i, n in enumerate(n_vals):
        ax.annotate(f"{n}b", (t_vals[i], p_vals[i]), textcoords="offset points", 
                     xytext=(0,12), ha='center', fontsize=9, fontweight='bold')

    # Print Table for this specific K_slope
    df = pd.DataFrame(corners)
    print(f"\n--- Results for K_slope = {k_s/1e6} MV/s, k_lim = {k_lim_val} ---")
    print(df[['n', 't_ns', 'p_ana_uw', 'p_dig_uw', 'p_uw']].to_string(index=False))

ax.set_yscale('log')
ax.set_xlabel('$T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax.set_ylabel('Power [µW]', fontsize=12, fontweight='bold')
ax.set_title('Corner Point Frontier for Different $K_{slope}$', fontsize=14, fontweight='bold', pad=12)
ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path2 = SAVE_DIR / "frontier_curves_kslopes.png"
fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"\nSaved: {save_path2}")

# --- PLOTTING ---
fig3, ax = plt.subplots(figsize=(10, 7))

t_mins = [m['t_ns'] for m in min_points]
p_mins = [m['p_uw'] for m in min_points]
k_labels = [m['K_slope_MV'] for m in min_points]
n_labels = [m['n'] for m in min_points]

# Plot the curve connecting the minimums
ax.plot(t_mins, p_mins, 'o', markersize=10, linewidth=2.6, color='#2CA02C',
        markeredgewidth=2, markeredgecolor='white', label='Optimal Design Frontier')

# Mark the global minimum with a star
best_idx = min_points.index(best_corner_global)
ax.scatter([t_mins[best_idx]], [p_mins[best_idx]], 
           marker='*', s=800, color='gold', edgecolors='red', linewidths=2, 
           zorder=20, label='Global Minimum')

# Annotate each point with Bits and K_slope
for i in range(len(min_points)):
    ax.annotate(f"{n_labels[i]}b\n{k_labels[i]:.0f} MV/s", 
                 (t_mins[i], p_mins[i]), 
                 textcoords="offset points", 
                 xytext=(0,15), ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_yscale('log')
ax.set_xlabel('Optimal $T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax.set_ylabel('Minimum Power [µW]', fontsize=12, fontweight='bold')
ax.set_title(f'Optimal Power Points for Different $K_{{slope}}$ ($k_{{lim}}$={k_lim_val})', 
             fontsize=14, fontweight='bold', pad=12)
ax.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.96, edgecolor='black')

plt.tight_layout()
save_path3 = SAVE_DIR / "optimal_design_frontier.png"
fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path3}")

# --- Save Final Table to Text File ---
output_text = []
output_text.append("=" * 120)
output_text.append("OPTIMAL DESIGN POINTS SUMMARY")
output_text.append("=" * 120)
output_text.append("")

# Regular Mode
output_text.append("REGULAR MODE - GLOBAL MINIMUM")
output_text.append("-" * 120)
output_text.append(f"K_slope = {best_corner_global['K_slope_MV']:.1f} MV/s")
output_text.append(f"Bits = {int(best_corner_global['n'])}")
output_text.append(f"T_DTC = {best_corner_global['t_ns']:.3f} ns")
output_text.append(f"C0 = {best_corner_global['C0_fF']:.2f} fF")
output_text.append(f"Ca = {best_corner_global['Ca_fF']:.2f} fF")
output_text.append(f"P_ana = {best_corner_global['p_ana_uw']:.3f} µW")
output_text.append(f"P_dig = {best_corner_global['p_dig_uw']:.3f} µW")
output_text.append(f"P_total = {best_corner_global['p_uw']:.3f} µW")
output_text.append("")

# Bit Extension Mode
output_text.append("BIT EXTENSION MODE - GLOBAL MINIMUM")
output_text.append("-" * 120)
output_text.append(f"K_slope = {best_corner_global_bext['K_slope_MV']:.1f} MV/s")
output_text.append(f"Bits = {int(best_corner_global_bext['n'])}")
output_text.append(f"T_DTC = {best_corner_global_bext['t_ns']:.3f} ns")
output_text.append(f"C0 = {best_corner_global_bext['C0_fF']:.2f} fF")
output_text.append(f"Ca = {best_corner_global_bext['Ca_fF']:.2f} fF")
output_text.append(f"P_ana = {best_corner_global_bext['p_ana_uw']:.3f} µW")
output_text.append(f"P_dig = {best_corner_global_bext['p_dig_uw']:.3f} µW")
output_text.append(f"P_total = {best_corner_global_bext['p_uw']:.3f} µW")
output_text.append("")

output_text.append("=" * 120)
output_text.append("CIRCUIT PARAMETERS")
output_text.append("=" * 120)
output_text.append(f"VDD = {VDD} V")
output_text.append(f"C_load = {C_load*1e15} fF")
output_text.append(f"Cu0 = {Cu0*1e15} fF")
output_text.append(f"Resolution = {res*1e12} ps")
output_text.append(f"k_lim = {k_lim_val}")
output_text.append("=" * 120)

# Save to file
summary_path = SAVE_DIR / "optimal_points_summary.txt"
with open(summary_path, 'w') as f:
    f.write('\n'.join(output_text))

print(f"\nSaved: {summary_path}")

# --- Print to console ---
print("\n--- Optimal Design Points Summary ---")
print("\nREGULAR MODE - GLOBAL MINIMUM")
print(f"  K_slope = {best_corner_global['K_slope_MV']:.1f} MV/s")
print(f"  Bits = {int(best_corner_global['n'])}")
print(f"  T_DTC = {best_corner_global['t_ns']:.3f} ns")
print(f"  P_total = {best_corner_global['p_uw']:.3f} µW")
print("\nBIT EXTENSION MODE - GLOBAL MINIMUM")
print(f"  K_slope = {best_corner_global_bext['K_slope_MV']:.1f} MV/s")
print(f"  Bits = {int(best_corner_global_bext['n'])}")
print(f"  T_DTC = {best_corner_global_bext['t_ns']:.3f} ns")
print(f"  P_total = {best_corner_global_bext['p_uw']:.3f} µW")
# ============================================================
# COMPARISON PLOT: Regular vs Bit Extension
# ============================================================
fig_comparison, ax_comp = plt.subplots(figsize=(12, 7))

# Plot regular mode optimal points
t_mins_regular = [m['t_ns'] for m in min_points]
p_mins_regular = [m['p_uw'] for m in min_points]
k_labels_regular = [m['K_slope_MV'] for m in min_points]
n_labels_regular = [m['n'] for m in min_points]

# Plot bit extension mode optimal points
t_mins_bext_comp = [m['t_ns'] for m in min_points_bext]
p_mins_bext_comp = [m['p_uw'] for m in min_points_bext]
k_labels_bext_comp = [m['K_slope_MV'] for m in min_points_bext]
n_labels_bext_comp = [m['n'] for m in min_points_bext]

# Plot both frontiers
line_regular, = ax_comp.plot(t_mins_regular, p_mins_regular, 'o', markersize=11, linewidth=2.8, 
                              color='#1F77B4', markeredgewidth=2, markeredgecolor='white',
                              label='Regular Mode', zorder=5)
line_bext, = ax_comp.plot(t_mins_bext_comp, p_mins_bext_comp, 's', markersize=11, linewidth=2.8, 
                           color='#FF7F0E', markeredgewidth=2, markeredgecolor='white',
                           label='Bit Extension Mode', zorder=5)

# Mark global minimums with stars
best_idx_regular = min_points.index(best_corner_global)
ax_comp.scatter([t_mins_regular[best_idx_regular]], [p_mins_regular[best_idx_regular]], 
                marker='*', s=1000, color='gold', edgecolors='red', linewidths=2.5, 
                zorder=25, label='Global Minimum (Regular)')

best_idx_bext_comp = min_points_bext.index(best_corner_global_bext)
ax_comp.scatter([t_mins_bext_comp[best_idx_bext_comp]], [p_mins_bext_comp[best_idx_bext_comp]], 
                marker='*', s=1000, color='lime', edgecolors='darkgreen', linewidths=2.5, 
                zorder=25, label='Global Minimum (Bit Extension)')

# Annotate regular mode points
for i in range(len(min_points)):
    ax_comp.annotate(f"{n_labels_regular[i]}b\n{k_labels_regular[i]:.0f}", 
                     (t_mins_regular[i], p_mins_regular[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='#1F77B4', 
                               edgecolor='black', alpha=0.6, linewidth=0.8))

# Annotate bit extension mode points
for i in range(len(min_points_bext)):
    ax_comp.annotate(f"{n_labels_bext_comp[i]}b\n{k_labels_bext_comp[i]:.0f}", 
                     (t_mins_bext_comp[i], p_mins_bext_comp[i]), 
                     textcoords="offset points", 
                     xytext=(0,-18), ha='center', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='#FF7F0E', 
                               edgecolor='black', alpha=0.6, linewidth=0.8))

ax_comp.set_yscale('log')
ax_comp.set_xlabel('Optimal $T_{DTC}$ [ns]', fontsize=12, fontweight='bold')
ax_comp.set_ylabel('Minimum Power [µW]', fontsize=12, fontweight='bold')
ax_comp.set_title(f'Comparison: Regular vs Bit Extension Mode ($k_{{lim}}$={k_lim_val})', 
                  fontsize=14, fontweight='bold', pad=12)
ax_comp.grid(True, which="both", linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
ax_comp.set_axisbelow(True)
ax_comp.legend(fontsize=11, framealpha=0.96, edgecolor='black', loc='best')

plt.tight_layout()
save_path_comparison = SAVE_DIR / "comparison_regular_vs_bitextension.png"
fig_comparison.savefig(save_path_comparison, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path_comparison}")
print(f"\n✓ All results saved to: {SAVE_DIR.absolute()}")