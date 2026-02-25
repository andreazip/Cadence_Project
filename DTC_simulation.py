import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path

# ===== PUBLICATION-READY PLOT STYLE (matching plot_delay.py) =====
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
    
    # Grid (match reference style)
    "grid.alpha": 0.6,
    "grid.color": "#b7b7b7",
    "grid.linestyle": "--",
    "grid.linewidth": 1.2,
    
    # Figure
    "figure.dpi": 100,
    "savefig.dpi": 300,  # High resolution for saving
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

# ===== CONFIGURATION =====
# Set SAVE_DIR to the parent directory where you want to save plots
# The script will create a "plot_python" folder inside SAVE_DIR automatically
SAVE_DIR = Path(r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP")
PLOT_FOLDER = "plot_python"
SAVE_PATH = SAVE_DIR / PLOT_FOLDER
# Create the plot directory if it doesn't exist
SAVE_PATH.mkdir(parents=True, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to SAVE_PATH with given filename."""
    filepath = SAVE_PATH / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")

def E_msb_0(C_k, C0, Ca, Vdd):
    return  (C0+(Ca-C_k))*(C_k/(C0+Ca))*Vdd**2

def E_msb_1(Ca, C_k, Vdd):
    return (Ca-C_k)/(C0+Ca)*(C0+C_k)*Vdd**2

RUN = {
    "DAC_mismatch": True,
    "CLM": False,
    "Non-linearities-capacitor": False
}

#define delay in the DTC by using 
n = 8 #number of bits
N = 2**n-1 #max ndigital number
Cu = 30e-15 #unit capacitance size

# Generate mismatched binary capacitors
C_array = np.zeros(n-1)

#sigmaC/C = Ac/sqrt(A) Pelgrom's law, where A is the area of the capacitor and Ac is a process-dependent constant. Assuming C = Cu, we can express the mismatch as sigmaC/C = Ac/sqrt(Cu). For a given mismatch factor (e.g., 0.003), we can derive Ac as follows:
Ac = 5.218e-3 #nm, which is a typical value for modern processes
A = 15 #um^2, which is a typical area for a 30fF capacitor

sigma_c = Ac / np.sqrt(A)*Cu  # Calculate sigmaC based on Pelgrom's law

for j in range(n-1):
        ideal_value = (2**j) * Cu
        if RUN["DAC_mismatch"] :
            mismatch =  np.random.randn() * sigma_c
        else:
            mismatch = 0
        C_array[j] = ideal_value + mismatch

Ca = np.sum(C_array)
C0= Ca*8/3 #fF, the reference capacitor, which can also be mismatched

Vdd = 1.1 #V
f = 20e6 #Hz, operating frequency

Vth = Vdd/2 #V
Ich = 300e-9 #A
Cramp = 5e-15 #F

# C1 = 5.300001e-07 #1/V
# C2 = 2.030000e-06 #1/V^2
# I1 = 1.35e-6 #A/V
#from paper, but reaally large non-linearities
C1 = 0.323
C2 = -0.09
I1 = 0.184

# Store initial values for CLM and Non-linearities calculations
Ich_init = Ich
Cramp_init = Cramp

k = -Ich_init/Cramp_init

Vst_array = np.zeros(N)
E_array =np.zeros(N)

half = (N-1)//2
first = True
for i in range(N):
    #first iteration for MSB = 0
    C_k = 0
    
    for j in range(n-1):
        bit = (i >> j) & 1
        C_k += (1 - bit) * C_array[j]
    if i > half:
        Vst_array[i] = (1+(Ca-C_k)/(C0+Ca))*Vdd
        E_array[i] = E_msb_1(Ca, C_k, Vdd)
    else:
        Vst_array[i] = (1-C_k/(C0+Ca))*Vdd
        E_array[i] = E_msb_0(C_k, C0, Ca, Vdd)


#calculate the delay
delay = np.zeros(N)
Iarray = np.zeros(N)
C_array = np.zeros(N)

Vstmin = Vst_array[0]
Vstmax = Vst_array[-1]

for i, Vst in enumerate(Vst_array):
    # Initialize with base values
    Ich_eff = Ich_init
    Cramp_eff = Cramp_init
    
    # Apply CLM if enabled
    if RUN["CLM"]:
        Ich_eff = Ich_init * (1 + I1*(Vst-0.4))
        Iarray[i] = Ich_eff
    
    # Apply Non-linearities-capacitor if enabled
    if RUN["Non-linearities-capacitor"]:
        Cramp_eff = Cramp_init * (1 + C1*Vst + C2*Vst**2)
        C_array[i] = Cramp_eff

    # Calculate delay using updated values
    k_eff = -Ich_eff/Cramp_eff
    delay[i] = (Vth - Vst) / k_eff

codes = np.arange(N-1)

delay=np.delete(delay, half)
E_array=np.delete(E_array, half)
E_array = E_array * f

# Plotting
fig_delay = plt.figure(figsize=(10, 6))
ax = fig_delay.add_subplot(111)
ax.plot(codes, delay * 1e9, color='#D62728', linewidth=2.6, label='DTC Delay')
ax.set_title('DTC Delay vs Digital Code', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
save_figure(fig_delay, 'dtc_delay_vs_code.png')
plt.close(fig_delay)

fig_energy = plt.figure(figsize=(10, 6))
ax = fig_energy.add_subplot(111)
ax.plot(codes, E_array*1e6, color='#1F77B4', linewidth=2.6, label='Energy')
ax.set_title('DTC Power Consumption vs Digital Code', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy [μW]', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
save_figure(fig_energy, 'dtc_power_vs_code.png')
plt.close(fig_energy)

# Plot I_array if CLM is enabled
if RUN["CLM"]:
    Iarray_nonzero = Iarray[Iarray != 0]
    if len(Iarray_nonzero) > 0:
        fig_current = plt.figure(figsize=(10, 6))
        ax = fig_current.add_subplot(111)
        ax.plot(codes, Iarray[:-1] * 1e9, color='#2CA02C', linewidth=2.6, label='CLM Current')
        ax.set_title('Effective Current vs Digital Code (CLM Enabled)', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Current [nA]', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
        save_figure(fig_current, 'dtc_clm_current.png')
        plt.close(fig_current)

# Plot C_array if Non-linearities-capacitor is enabled
if RUN["Non-linearities-capacitor"]:
    C_array_nonzero = C_array[C_array != 0]
    if len(C_array_nonzero) > 0:
        fig_capacitance = plt.figure(figsize=(10, 6))
        ax = fig_capacitance.add_subplot(111)
        ax.plot(codes, C_array[:-1] * 1e15, color='#FF7F0E', linewidth=2.6, label='Ramp Capacitance')
        ax.set_title('Ramp Capacitance vs Digital Code (Non-Linearities Enabled)', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capacitance [fF]', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black')
        save_figure(fig_capacitance, 'dtc_capacitance.png')
        plt.close(fig_capacitance)

#calculate DNL and INL for different scenarios
# Start from code 1
codes_dnl = np.arange(1, N-1)

# Store delay for ideal case (no CLM, no Non-linearities)
delay_ideal = np.zeros(N)
for i, Vst in enumerate(Vst_array):
    k_eff = -Ich_init/Cramp_init
    delay_ideal[i] = (Vth - Vst) / k_eff
delay_ideal = np.delete(delay_ideal, half)

# Calculate with CLM only
delay_clm = np.zeros(N)
for i, Vst in enumerate(Vst_array):
    Ich_eff = Ich_init * (1 + I1*(Vst-0.4))
    k_eff = -Ich_eff/Cramp_init
    delay_clm[i] = (Vth - Vst) / k_eff
delay_clm = np.delete(delay_clm, half)

# Calculate with Non-linearities only
delay_nonlin = np.zeros(N)
for i, Vst in enumerate(Vst_array):
    Cramp_eff = Cramp_init * (1 + C1*Vst + C2*Vst**2)
    k_eff = -Ich_init/Cramp_eff
    delay_nonlin[i] = (Vth - Vst) / k_eff
delay_nonlin = np.delete(delay_nonlin, half)

# Calculate with both CLM and Non-linearities
delay_both = np.zeros(N)
for i, Vst in enumerate(Vst_array):
    Ich_eff = Ich_init * (1 + I1*(Vst-0.4))
    Cramp_eff = Cramp_init * (1 + C1*Vst + C2*Vst**2)
    k_eff = -Ich_eff/Cramp_eff
    delay_both[i] = (Vth - Vst) / k_eff
delay_both = np.delete(delay_both, half)

# Calculate DNL and INL for CLM only
lsb_clm = (delay_clm[-1] - delay_clm[0]) / (len(delay_clm) - 1)
dnl_clm = np.diff(delay_clm) / lsb_clm - 1
inl_clm = np.cumsum(dnl_clm)

# Calculate DNL and INL for Non-linearities only
lsb_nonlin = (delay_nonlin[-1] - delay_nonlin[0]) / (len(delay_nonlin) - 1)
dnl_nonlin = np.diff(delay_nonlin) / lsb_nonlin - 1
inl_nonlin = np.cumsum(dnl_nonlin)

# Calculate DNL and INL for both
lsb_both = (delay_both[-1] - delay_both[0]) / (len(delay_both) - 1)
dnl_both = np.diff(delay_both) / lsb_both - 1
inl_both = np.cumsum(dnl_both)

# Plot delay characteristic comparison: Ideal vs Both (CLM + Non-linearities)
fig_delay_comp = plt.figure(figsize=(11, 6))
ax = fig_delay_comp.add_subplot(111)
ax.plot(codes, delay_ideal * 1e9, color='#D62728', linewidth=2.6, linestyle='-', 
        label='Ideal (No CLM, No Non-Linearities)', marker=None)
ax.plot(codes, delay_both * 1e9, color='#1F77B4', linewidth=2.6, linestyle='--', 
        label='With CLM + Non-Linearities', marker=None)
ax.set_title('DTC Delay Characteristic: Ideal vs Non-Ideal Effects', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
save_figure(fig_delay_comp, 'dtc_delay_characteristic_comparison.png')
plt.close(fig_delay_comp)

# Plot delay characteristic comparison: Ideal vs Capacitor Non-Linearities only
fig_delay_comp_cap = plt.figure(figsize=(11, 6))
ax = fig_delay_comp_cap.add_subplot(111)
ax.plot(codes, delay_ideal * 1e9, color='#D62728', linewidth=2.6, linestyle='-', 
        label='Ideal (No Effects)', marker=None)
ax.plot(codes, delay_nonlin * 1e9, color='#FF7F0E', linewidth=2.6, linestyle='--', 
        label='Capacitor Non-Linearities Only', marker=None)
ax.set_title('DTC Delay Characteristic: Ideal vs Capacitor Non-Linearities', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
save_figure(fig_delay_comp_cap, 'dtc_delay_characteristic_capacitor_nonlin.png')
plt.close(fig_delay_comp_cap)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

# DNL plot
ax1.plot(codes_dnl, dnl_clm, label='CLM Only', linewidth=2.6, alpha=0.85, color='#D62728')
ax1.plot(codes_dnl, dnl_nonlin, label='Non-Linearities Only', linewidth=2.6, alpha=0.85, color='#1F77B4')
ax1.plot(codes_dnl, dnl_both, label='CLM + Non-Linearities', linewidth=2.6, alpha=0.85, color='#2CA02C')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
ax1.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
ax1.set_title("Differential Non-Linearity (DNL) and Integral Non-Linearity (INL) Comparison", 
              fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
ax1.set_axisbelow(True)

# INL plot
ax2.plot(codes_dnl, inl_clm, label='CLM Only', linewidth=2.6, alpha=0.85, color='#D62728')
ax2.plot(codes_dnl, inl_nonlin, label='Non-Linearities Only', linewidth=2.6, alpha=0.85, color='#1F77B4')
ax2.plot(codes_dnl, inl_both, label='CLM + Non-Linearities', linewidth=2.6, alpha=0.85, color='#2CA02C')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
ax2.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
ax2.set_axisbelow(True)

plt.tight_layout()
save_figure(fig, 'dtc_dnl_inl_comparison.png')
plt.close(fig)

ideal_offset = delay_ideal[0] * 1e9
ideal_range = (delay_ideal[-1] - delay_ideal[0]) * 1e9
ideal_res = (delay_ideal[-1] - delay_ideal[0]) * 1e12 / len(delay_ideal)

# Calculate power from energy at minimum and maximum codes
P_min = E_array[0] * 1e6  # Convert to uW
P_max = E_array[-1] * 1e6  # Convert to uW

print(
    f"Delta V = {(Vst_array[-1]-Vst_array[0]) * 1e3} mV \n"
    f"T_offset (ideal) = {ideal_offset} ns \n"
    f"T_range (ideal) = {ideal_range} ns \n"
    f"Resolution (ideal) = {ideal_res} ps \n"
    f"\n"
    f"Power @ Code 0 (f={f/1e6:.0f} MHz) = {P_min:.3e} uW \n"
    f"Power @ Code 255 (f={f/1e6:.0f} MHz) = {P_max:.3e} uW"
)
print(f"CLM LSB = {lsb_clm:.2e}, Non-lin LSB = {lsb_nonlin:.2e}, Both LSB = {lsb_both:.2e}")
print(f"\nAll plots saved to: {SAVE_PATH}")

if RUN["DAC_mismatch"]:
    MC_runs = 100  # number of mismatch realizations

    min_list = []

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    for mc in range(MC_runs):

        # ----- regenerate mismatch every run -----


        # binary capacitor array
        C_array = np.zeros(n-1)
        for j in range(n-1):
            C_array[j] = (2**j)*(Cu + np.random.randn()*sigma_c)

        Ca = np.sum(C_array)

        # ----- compute Vst -----
        Vst_array = np.zeros(N)

        for i in range(N):
            C_k = 0
            for j in range(n-1):
                bit = (i >> j) & 1
                C_k += (1 - bit) * C_array[j]

            if i > half:
                Vst_array[i] = (1+(Ca-C_k)/(C0+Ca))*Vdd
            else:
                Vst_array[i] = (1-C_k/(C0+Ca))*Vdd

        # ----- delay with CLM and Non-linearities -----
        delay = np.zeros(N)
        for i, Vst in enumerate(Vst_array):
            # Initialize with base values
            Ich_eff = Ich_init
            Cramp_eff = Cramp_init
            
            # Apply CLM if enabled
            if RUN["CLM"]:
                Ich_eff = Ich_init*(1+ I1*(Vst-0.4))
            
            # Apply Non-linearities-capacitor if enabled
            if RUN["Non-linearities-capacitor"]:
                Cramp_eff = Cramp_init*(1 + C1*Vst + C2*Vst**2)

            # Calculate delay using updated values
            k_eff = -Ich_eff/Cramp_eff
            delay[i] = (Vth - Vst) / k_eff
        
        delay = np.delete(delay, half)

        # ----- DNL / INL -----
        lsb_ideal = (delay[-1] - delay[0]) / (len(delay)-1)

        dnl = np.diff(delay)/lsb_ideal - 1
        inl = np.cumsum(dnl)

        if max(dnl) < 0.5:
            min_list.append(max(dnl))

        codes_dnl = np.arange(1, N-1)
        
        # Vary alpha based on iteration for different shades
        alpha_value = 0.15 + (mc / MC_runs) * 0.5  # Alpha ranges from 0.15 to 0.65
        ax1.plot(codes_dnl, dnl, alpha=alpha_value, linewidth=1.6, color='#D62728', label=None)
        ax2.plot(codes_dnl, inl, alpha=alpha_value, linewidth=1.6, color='#1F77B4', label=None)

    ax1.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Monte Carlo DNL and INL Analysis ({MC_runs} Realizations, LSB = {lsb_ideal:.2e})", 
                  fontsize=14, fontweight='bold', pad=12)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.0, alpha=0.4, label='±0.5 LSB')
    ax1.axhline(y=-0.5, color='red', linestyle='--', linewidth=1.0, alpha=0.4)
    ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax1.set_axisbelow(True)
    
    ax2.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, 'dtc_mc_analysis.png')
    plt.close(fig)

    print(f"Probability of staying below 0.5 LSB = {len(min_list)/MC_runs*100} %")

# ===== PARAMETRIC SWEEP: DNL/INL sensitivity to I_ch and C_ramp =====
def calculate_dnl_inl(ich_val, cramp_val, vst_array_in):
    """Calculate DNL and INL for given I_ch and C_ramp values."""
    delay_clm_sweep = np.zeros(N)
    delay_nonlin_sweep = np.zeros(N)
    delay_both_sweep = np.zeros(N)
    
    for i, Vst in enumerate(vst_array_in):
        # CLM only
        ich_eff = ich_val * (1 + I1*(Vst-0.4))
        k_eff = -ich_eff / cramp_val
        delay_clm_sweep[i] = (Vth - Vst) / k_eff
        
        # Non-linearities only
        cramp_eff = cramp_val * (1 + C1*Vst + C2*Vst**2)
        k_eff = -ich_val / cramp_eff
        delay_nonlin_sweep[i] = (Vth - Vst) / k_eff
        
        # Both
        ich_eff = ich_val * (1 + I1*(Vst-0.4))
        cramp_eff = cramp_val * (1 + C1*Vst + C2*Vst**2)
        k_eff = -ich_eff / cramp_eff
        delay_both_sweep[i] = (Vth - Vst) / k_eff
    
    # Delete middle element
    delay_clm_sweep = np.delete(delay_clm_sweep, half)
    delay_nonlin_sweep = np.delete(delay_nonlin_sweep, half)
    delay_both_sweep = np.delete(delay_both_sweep, half)
    
    # Calculate DNL and INL for each scenario
    lsb_clm_s = (delay_clm_sweep[-1] - delay_clm_sweep[0]) / (len(delay_clm_sweep) - 1)
    dnl_clm_s = np.diff(delay_clm_sweep) / lsb_clm_s - 1
    inl_clm_s = np.cumsum(dnl_clm_s)
    
    lsb_nonlin_s = (delay_nonlin_sweep[-1] - delay_nonlin_sweep[0]) / (len(delay_nonlin_sweep) - 1)
    dnl_nonlin_s = np.diff(delay_nonlin_sweep) / lsb_nonlin_s - 1
    inl_nonlin_s = np.cumsum(dnl_nonlin_s)
    
    lsb_both_s = (delay_both_sweep[-1] - delay_both_sweep[0]) / (len(delay_both_sweep) - 1)
    dnl_both_s = np.diff(delay_both_sweep) / lsb_both_s - 1
    inl_both_s = np.cumsum(dnl_both_s)
    
    return dnl_clm_s, inl_clm_s, dnl_nonlin_s, inl_nonlin_s, dnl_both_s, inl_both_s

# Parameter combinations
param_sets_ich = [
    {"Ich": 1e-9, "Cramp": 5e-15, "label": "I_ch=100nA, C_ramp=5fF"},
    {"Ich": 300e-9, "Cramp": 5e-15, "label": "I_ch=300nA, C_ramp=5fF"},
    {"Ich": 1000e-9, "Cramp": 5e-15, "label": "I_ch=500nA, C_ramp=5fF"},
]

param_sets_cramp = [
    {"Ich": 300e-9, "Cramp": 1e-18, "label": "I_ch=300nA, C_ramp=1fF"},
    {"Ich": 300e-9, "Cramp": 5e-15, "label": "I_ch=300nA, C_ramp=5fF"},
    {"Ich": 300e-9, "Cramp": 1e-12, "label": "I_ch=300nA, C_ramp=10fF"},
]

# Figure 1: I_ch sweep (C_ramp = 5 fF fixed)
fig_ich, axes_ich = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
for idx, param in enumerate(param_sets_ich):
    ich_p = param["Ich"]
    cramp_p = param["Cramp"]
    label_p = param["label"]
    
    dnl_clm, inl_clm, dnl_nl, inl_nl, dnl_both, inl_both = calculate_dnl_inl(ich_p, cramp_p, Vst_array)
    codes_dnl = np.arange(1, N-1)
    
    # DNL subplot (top row)
    axes_ich[0, idx].plot(codes_dnl, dnl_clm, label='CLM Only', linewidth=2.2, alpha=0.85, color='#D62728')
    axes_ich[0, idx].plot(codes_dnl, dnl_nl, label='Non-Linearities Only', linewidth=2.2, alpha=0.85, color='#1F77B4')
    axes_ich[0, idx].plot(codes_dnl, dnl_both, label='CLM + Non-Linearities', linewidth=2.2, alpha=0.85, color='#2CA02C')
    axes_ich[0, idx].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    axes_ich[0, idx].set_ylabel("DNL (LSB)", fontsize=11, fontweight='bold')
    axes_ich[0, idx].set_title(label_p, fontsize=12, fontweight='bold', pad=10)
    axes_ich[0, idx].grid(True, linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    axes_ich[0, idx].set_axisbelow(True)
    if idx == 0:
        axes_ich[0, idx].legend(fontsize=9, framealpha=0.95, edgecolor='black', loc='best')
    
    # INL subplot (bottom row)
    axes_ich[1, idx].plot(codes_dnl, inl_clm, label='CLM Only', linewidth=2.2, alpha=0.85, color='#D62728')
    axes_ich[1, idx].plot(codes_dnl, inl_nl, label='Non-Linearities Only', linewidth=2.2, alpha=0.85, color='#1F77B4')
    axes_ich[1, idx].plot(codes_dnl, inl_both, label='CLM + Non-Linearities', linewidth=2.2, alpha=0.85, color='#2CA02C')
    axes_ich[1, idx].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    axes_ich[1, idx].set_ylabel("INL (LSB)", fontsize=11, fontweight='bold')
    axes_ich[1, idx].set_xlabel("Digital Code", fontsize=11, fontweight='bold')
    axes_ich[1, idx].grid(True, linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    axes_ich[1, idx].set_axisbelow(True)

fig_ich.suptitle('DNL and INL Sensitivity to Source Current (I_ch) - C_ramp = 5 fF', 
                  fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig_ich, 'dtc_dnl_inl_ich_sweep.png')
plt.close(fig_ich)

# Figure 2: C_ramp sweep (I_ch = 300 nA fixed)
fig_cramp, axes_cramp = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
for idx, param in enumerate(param_sets_cramp):
    ich_p = param["Ich"]
    cramp_p = param["Cramp"]
    label_p = param["label"]
    
    dnl_clm, inl_clm, dnl_nl, inl_nl, dnl_both, inl_both = calculate_dnl_inl(ich_p, cramp_p, Vst_array)
    codes_dnl = np.arange(1, N-1)
    
    # DNL subplot (top row)
    axes_cramp[0, idx].plot(codes_dnl, dnl_clm, label='CLM Only', linewidth=2.2, alpha=0.85, color='#D62728')
    axes_cramp[0, idx].plot(codes_dnl, dnl_nl, label='Non-Linearities Only', linewidth=2.2, alpha=0.85, color='#1F77B4')
    axes_cramp[0, idx].plot(codes_dnl, dnl_both, label='CLM + Non-Linearities', linewidth=2.2, alpha=0.85, color='#2CA02C')
    axes_cramp[0, idx].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    axes_cramp[0, idx].set_ylabel("DNL (LSB)", fontsize=11, fontweight='bold')
    axes_cramp[0, idx].set_title(label_p, fontsize=12, fontweight='bold', pad=10)
    axes_cramp[0, idx].grid(True, linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    axes_cramp[0, idx].set_axisbelow(True)
    if idx == 0:
        axes_cramp[0, idx].legend(fontsize=9, framealpha=0.95, edgecolor='black', loc='best')
    
    # INL subplot (bottom row)
    axes_cramp[1, idx].plot(codes_dnl, inl_clm, label='CLM Only', linewidth=2.2, alpha=0.85, color='#D62728')
    axes_cramp[1, idx].plot(codes_dnl, inl_nl, label='Non-Linearities Only', linewidth=2.2, alpha=0.85, color='#1F77B4')
    axes_cramp[1, idx].plot(codes_dnl, inl_both, label='CLM + Non-Linearities', linewidth=2.2, alpha=0.85, color='#2CA02C')
    axes_cramp[1, idx].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    axes_cramp[1, idx].set_ylabel("INL (LSB)", fontsize=11, fontweight='bold')
    axes_cramp[1, idx].set_xlabel("Digital Code", fontsize=11, fontweight='bold')
    axes_cramp[1, idx].grid(True, linestyle='--', alpha=0.6, linewidth=1.0, color="#b7b7b7")
    axes_cramp[1, idx].set_axisbelow(True)

fig_cramp.suptitle('DNL and INL Sensitivity to Ramp Capacitance (C_ramp) - I_ch = 300 nA', 
                    fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
save_figure(fig_cramp, 'dtc_dnl_inl_cramp_sweep.png')
plt.close(fig_cramp)

print(f"Parametric sweep plots saved successfully!")