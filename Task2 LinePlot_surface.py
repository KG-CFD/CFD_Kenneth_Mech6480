import pandas as pd
import matplotlib.pyplot as plt

p_inf = 100 # Pascal
velocity =2720 # ms-1
T =300 #K
R = 296.8
Nitrogen_rho = p_inf/(T *R)
dynamic_pressure = 0.5 *(Nitrogen_rho * velocity**2)


file1 = 'Task2_Line1.csv'
data1 = pd.read_csv(file1)


file2 = 'Task2_Line2.csv'
data2 = pd.read_csv(file2)


x1 = data1.iloc[:, 0]
y1 = data1.iloc[:, 1]
Cp1 = (y1 - p_inf) / dynamic_pressure

x2 = data2.iloc[:, 1]
y2 = data2.iloc[:, 2]
Cp2 = (y2-p_inf)/ dynamic_pressure
# Plotting
plt.figure(figsize=(8, 6))

# Plot for cone 1
plt.plot(x1, Cp1, label='Cone 1', marker='o')

# Plot for cone 2
plt.plot(x2, Cp2, label='Cone 2', marker='s')

# Add labels and title
plt.xlabel('x')
plt.ylabel('Cp')
plt.title('Co-efficient of Pressure over 2 cone surfaces')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
