import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# 1. Load the file
df = pd.read_csv('results_cadence\cap_data.csv')

voltages = []
capacitances = []

# 2. Extract Voltage and Capacitance from multi-column headers
# Regex finds the number after 'Vbias='
pattern = re.compile(r"Vbias=([+-]?\d+\.?\d*e?[+-]?\d*)")

for col in df.columns:
    if col.endswith(' Y'): # Focus on capacitance columns
        match = pattern.search(col)
        if match:
            v_val = float(match.group(1))
            c_val = df[col].iloc[0] # Taking the single row of data
            voltages.append(v_val)
            capacitances.append(c_val)

# 3. Sort data by voltage
voltages = np.array(voltages)
capacitances = np.array(capacitances)
sort_idx = np.argsort(voltages)
voltages = voltages[sort_idx]
capacitances = capacitances[sort_idx]

# 4. Polynomial Fit: C(V) = p2*V^2 + p1*V + p0
# This maps to C0 * (1 + VC1*V + VC2*V^2)
coeffs = np.polyfit(voltages, capacitances, 2)
p2, p1, p0 = coeffs

C0 = p0
VC1 = p1 / C0
VC2 = p2 / C0

print(f"C0:  {C0:.6e} F")
print(f"VC1: {VC1:.6e} V^-1")
print(f"VC2: {VC2:.6e} V^-2")

# 5. Visual Verification
plt.scatter(voltages, capacitances, label='Data', color='red')
v_fit = np.linspace(min(voltages), max(voltages), 100)
c_fit = np.polyval(coeffs, v_fit)
plt.plot(v_fit, c_fit, label='Fit', linestyle='--')
plt.xlabel('Bias Voltage (V)')
plt.ylabel('Capacitance (F)')
plt.legend()
plt.show()

