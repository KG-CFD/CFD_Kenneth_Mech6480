import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read the Excel file
file_path = "scope.csv"
data = pd.read_csv(file_path)
# Display the first few rows
print(data.head())

# Extract time and voltage
time = data['second']
voltage = data['Volt1']


v_max = 0.1056
v_min =0.0129

phase_shift = np.arcsin(((voltage - v_min ) / ((v_max - v_min) / 2)-1))

# Print the phase shift values
print("First few phase shift values:", phase_shift.head())




sampling_frequency = 1 / (time[1] - time[0])
print(sampling_frequency)
# Perform FFT on the phase shift data
fft_phase_shift = np.fft.fft(phase_shift)

# Calculate the magnitude of the FFT (this gives the amplitude spectrum)
fft_magnitude = np.abs(fft_phase_shift)

# Calculate the Power Spectral Density
psd = (fft_magnitude ** 2) / len(phase_shift)

# Frequency
frequencies = np.fft.fftfreq(len(phase_shift), d=(time[1] - time[0]))


positive_frequencies = frequencies[:len(frequencies) // 2]
psd_positive = psd[:len(frequencies) // 2]

# Plot the Power Spectral Density

plt.figure(figsize=(10, 6))
plt.semilogy(positive_frequencies / 10000, psd_positive)
plt.title("Power Spectral Density of Phase Shift Signal")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Power Spectral Density")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('FFT_Heat-gun_channel1.png')
plt.show()
