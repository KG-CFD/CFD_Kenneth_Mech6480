import pandas as pd
import matplotlib.pyplot as plt

# Load the first CSV file (cone 1)
file1 = 'Line1.csv'
data1 = pd.read_csv(file1)

# Load the second CSV file (cone 2)
file2 = 'Line2.csv'
data2 = pd.read_csv(file2)

# Assuming the first two columns are 'x' and 'y'
x1 = data1.iloc[:, 1]
y1 = data1.iloc[:, 2]

x2 = data2.iloc[:, 1]
y2 = data2.iloc[:, 2]

# Plotting the data
plt.figure(figsize=(8, 6))

# Plot the data for cone 1
plt.plot(x1, y1, label='Cone 1', marker='o')

# Plot the data for cone 2
plt.plot(x2, y2, label='Cone 2', marker='s')

# Add labels and title
plt.xlabel('x')
plt.ylabel('Cp')
plt.title('Co-efficient of Pressure over 2 cone surfaces')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig('Task2_Cp')
plt.show()
