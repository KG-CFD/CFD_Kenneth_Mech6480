import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
p_inf = 100.0

# Load the first file (cone 1)
file1 = 'Task3B-blk-2-cell-42.dat.0'
data1 = pd.read_csv(file1,delim_whitespace=True)

# Load the second file (cone 2)
file2 = 'Task3B-blk-6-cell-479.dat.0'
data2 = pd.read_csv(file2,delim_whitespace=True)
Dynamic_pressure =4154.547
# Extract x and y data
x1 = data1.iloc[:, 0]
y1 = data1.iloc[:, 9]
Cp1 = (y1 - p_inf) / Dynamic_pressure

x2 = data2.iloc[:, 0]
y2 = data2.iloc[:, 9]
Cp2 = (y2 - p_inf) / Dynamic_pressure



spline1 = interpolate.make_interp_spline(x1, Cp1)
x1_smooth = np.linspace(x1.min(), x1.max(), 500)
Cp1_smooth = spline1(x1_smooth)

# Create spline interpolation for Cone 2
spline2 = interpolate.make_interp_spline(x2, Cp2)
x2_smooth = np.linspace(x2.min(), x2.max(), 500)
Cp2_smooth = spline2(x2_smooth)

# Plotting the  data for Cone 1
plt.plot(x1_smooth, Cp1_smooth, label='Cone 1', color ='blue')
plt.plot(x1, Cp1, 'bo')
# Plotting data for Cone 2
plt.plot(x2_smooth, Cp2_smooth, label='Cone 2 ', color='green')  # Smooth line
plt.plot(x2, Cp2, 'gs')  # Actual data points

# Add labels and title
plt.xlabel('Time step')
plt.ylabel('Cp')
plt.title('Coefficient of Pressure at 2 cone surface Mid-points for Task3 caseB')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()