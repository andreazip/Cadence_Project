import numpy as np
import matplotlib.pyplot as plt

def E_msb_0(C_k, C0, Ca, Vdd):
    return  C0*(C_k/(C0+Ca))*Vdd

def E_msb_1(Ca, C_k, Vdd):
    return (Ca-C_k)*Vdd

RUN = {
    "DAC_mismatch": True,
    "CLM":False
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
#C0= 338*Cu #fF, the reference capacitor, which can also be mismatched
C0= Ca*8/3 #fF, the reference capacitor, which can also be mismatched
# for j in range(337):
#     ideal_value = Cu
#     if RUN["DAC_mismatch"] :
#         mismatch =  np.random.randn() * sigma_c
#     else:
#         mismatch = 0
#     C0 += ideal_value + mismatch

Vdd = 1.1 #V

Vth = Vdd/2 #V
Ich = 300e-9 #A
Cramp = 5e-15 #F

k = -Ich/Cramp

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
for i, Vst in enumerate(Vst_array):
    # Effective current including CLM if enabled

    # Recalculate slope k
    k_eff = -Ich/Cramp

    # Delay
    delay[i] = (Vth - Vst) / k_eff
    delay[i] = (Vth-Vst)/k

codes = np.arange(N-1)

delay=np.delete(delay, half)
E_array=np.delete(E_array, half)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(codes, delay * 1e9)
plt.title('DTC Delay vs Digital Code (10-bit)')
plt.xlabel('Digital Code')
plt.ylabel('Delay [ns]')
plt.grid(True, linestyle='--', alpha=0.7)

plt.figure(figsize=(10, 6))
plt.plot(codes, E_array)
plt.title('DTC Energy consumption vs Digital Code (10-bit)')
plt.xlabel('Digital Code')
plt.ylabel('Energy [J]')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#calculate DNL and INL
lsb_ideal = (delay[-1] - delay[0]) / (len(delay) - 1) if len(delay) > 1 else 0

dnl = np.insert(np.diff(delay) / lsb_ideal - 1, 0, 0)
inl = np.cumsum(dnl)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
ax1.plot(codes, dnl, color='blue', markersize=4)
ax1.set_ylabel("DNL (LSB)")
ax1.set_title(f"DNL and INL (LSB={lsb_ideal:.2e})")
        
ax2.plot(codes, inl, color='crimson', markersize=4)
ax2.set_ylabel("INL (LSB)")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()

plt.show()

print(f"Delta V = {(Vst_array[-1]-Vst_array[0] )*1e3} mV \nT_offset = {delay[0]*1e9} ns \nT_range = {(delay[-1] -delay[0])*1e9} ns \nResolution = {(delay[-1] -delay[0])*1e12/len(delay)} ps ")

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

        # ----- delay -----
        delay = (Vth - Vst_array)/k
        delay = np.delete(delay, half)

        # ----- DNL / INL -----
        lsb_ideal = (delay[-1] - delay[0]) / (len(delay)-1)

        dnl = np.insert(np.diff(delay)/lsb_ideal - 1, 0, 0)
        inl = np.cumsum(dnl)

        if max(dnl) < 0.5:
            min_list.append(max(dnl))


        codes = np.arange(N-1)

        ax1.plot(codes, dnl, alpha=0.4)   # transparency helps visualization
        ax2.plot(codes, inl, alpha =0.4)

    ax1.set_ylabel("DNL (LSB)")
    ax1.set_title(f"DNL and INL (LSB={lsb_ideal:.2e})")
    ax2.set_ylabel("INL (LSB)")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    plt.show()

    print(f"Probability of staying below 0.5 LSB = {len(min_list)/MC_runs*100} %")