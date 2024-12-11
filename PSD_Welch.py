import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Parameters
WL = 532e-9  # Wavelength
v_max = 0.1056
v_min = 0.0129
window_size = 22  # Adjust based on your data

# Read the CSV file
file_path = "scope.csv"
data = pd.read_csv(file_path)

# Extract time and voltage columns (ensure column names match your CSV file)
time = data['second']
voltage = data['Volt1']

# Sampling frequency
fs = 1 / (time[1] - time[0])
print(f"Sampling Frequency: {fs} Hz")

# Convert voltage to phase shift
dphi = np.arcsin(((voltage - v_min) / ((v_max - v_min) / 2)) - 1)

# Compute PSD using Welch's method
frequencies, psd = welch(dphi, fs, nfft=len(dphi))

# Plotting the PSD
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, psd, label="Phase Shift PSD", linewidth=2)
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD of Phase Shift (rad²/Hz)")
plt.title("Power Spectral Density of Phase Shift Signal")
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("PSD_plot_phase_shift_signal.png")
plt.show()