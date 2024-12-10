import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Read the Excel file
file_path = "scope.csv"  # Replace with your file name
data = pd.read_csv(file_path)  # Reads the entire Excel sheet

# Display the first few rows
print(data.head())

# Extract time and voltage
time = data['second']
voltage = data['Volt1']
print("First few time values:", time.head())  # Time data
print("First few voltage values:", voltage.head())  # Voltage data

v_max = 0.1056
v_min =0.0129

phase_shift = np.arcsin(((voltage - v_min ) / ((v_max - v_min) / 2)-1))

# Print the phase shift values
print("First few phase shift values:", phase_shift.head())



# Sampling frequency (assuming the time data is evenly spaced)
sampling_frequency = 1 / (time[1] - time[0])

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
plt.semilogy(positive_frequencies, psd_positive)
plt.title('Power Spectral Density of Phase Shift Signal')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.savefig('FFT_Heat-gun_channel1.png')
plt.show()
