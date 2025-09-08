import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file1 = 'CFRP.txt'
data1 = pd.read_csv(file1, delim_whitespace=True)

y1 = data1.iloc[:, 0]  # Stress
x1 = data1.iloc[:, 1]  # Strain

# Create mask for the desired linear range
mask = (x1 >= 0.00281) & (x1 <= 0.00946)
filtered_x = x1[mask]
filtered_y = y1[mask]

# Convert to NumPy arrays
X = filtered_x.values
Y = filtered_y.values

# Perform linear regression using NumPy's polyfit
slope, intercept = np.polyfit(X, Y, 1)  # 1 = linear fit

# Calculate R-squared value
y_pred = slope * X + intercept
ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Generate points for the regression line
x_line = np.array([filtered_x.min(), filtered_x.max()])
y_line = slope * x_line + intercept

# Apply 0.2% offset method - FIXED APPROACH
# offset_strain = 0.002  # 0.2%

# Create the offset line: parallel to elastic slope but shifted right by 0.002 strain
 #offset_line = slope * (x_line - offset_strain) + intercept

# Find intersection between offset line and actual stress-strain curve
# We'll find where the actual curve first rises above the offset line
#found_yield = False
#for i in range(len(x1)):
    # Calculate what the offset line predicts at this strain
    #offset_stress_at_point = slope * (x1.iloc[i] - offset_strain) + intercept

    # Check if actual stress exceeds the offset line prediction


# If no intersection found (shouldn't happen for steel), use a fallback
#if not found_yield:
   # print("Warning: No yield point found using 0.2% offset method")
    # Fallback: use the maximum stress in the linear region
    #yield_strength = filtered_y.max()
    #yield_strain = filtered_x[filtered_y.idxmax()]

# Plotting
plt.figure(figsize=(10, 6))
# Plot all data in light gray
plt.plot(x1, y1, color='lightgray', marker='o', markersize=3, label='All Data')
# Plot filtered data in red
plt.plot(filtered_x, filtered_y, color='red', marker='o', linewidth=2, label='Linear Region')
# Plot regression line
plt.plot(x_line, y_line, 'b--', linewidth=2, label='Linear Fit')
# Plot offset line
#plt.plot(x_line, offset_line, 'g--', linewidth=2, label='0.2% Offset Line')
# Mark yield point


# Add Young's Modulus as a labeled gradient on the plot
mid_x = np.mean(x_line)
mid_y = np.mean(y_line)
# Create gradient label text
gradient_label = f'E = {slope/1000:.1f} GPa'
# Add text with arrow pointing to the regression line
plt.annotate(gradient_label, xy=(mid_x, mid_y), xytext=(mid_x + 0.001, mid_y - 1),
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
             fontsize=8.5, fontweight='bold', color='blue',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.xlabel('Strain',fontsize=14)
plt.ylabel('Stress (MPa)',fontsize=14)
plt.title('Stress vs Strain for CFRP',fontweight='bold',fontsize=18,pad=20)
plt.legend()
plt.grid(True)

# Add text box with results
textstr = f'Young\'s Modulus (E) = {slope/1000:.1f} GPa\n'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

textstr =f'Yield Strength = 514.4 Mpa'
props =dict(boxstyle='round', facecolor ='wheat', alpha =0.8)
plt.text(0.05,0.85,textstr,transform =plt.gca().transAxes , fontsize=10,
         verticalalignment='top', bbox=props)
plt.show()

