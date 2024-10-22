import pandas as pd
import numpy as np

# Load the surface data CSV file
surface_data = pd.read_csv('Line2Task2.csv')

# Free-stream conditions
p_inf = 100.0  # Free-stream pressure in Pa


# Extract x and pressure values
x = surface_data.iloc[:, 1]  # First column after time is 'x'
pressure = surface_data.iloc[:, 2]  # Second column after time is 'pressure'

# Cone geometry
L = 0.1  # Length of the cone in meters
angle1 = 55  # Cone half-angle in degrees
half_angle1 = np.radians(angle1)  # Convert angle to radians

# Function to calculate the local radius r at each point based on x
def calculate_r(x_val):
    return (x_val / L) * np.tan(half_angle1)

# Calculate the local radius at each x position
surface_data['r'] = x.apply(calculate_r)

# Compute dL, the difference between consecutive x positions
surface_data['dL'] = x.diff().fillna(0)  # First value will be zero

# Subtract free-stream pressure from the surface pressure
surface_data['delta_p'] = pressure - p_inf

# Compute differential drag force components (dF_drag)
surface_data['dF_drag'] = surface_data['delta_p'] * 2 * np.pi * surface_data['r'] * surface_data['dL']

# Compute the total drag force
total_drag = surface_data['dF_drag'].sum()

# Output the total drag force
print(f'Total Drag Force: {total_drag} N')