#Plotting pressure along wall normalised against reservoir

import matplotlib.pyplot as plt
import pandas as pd
file1 = 'Pressure_symmetryWall.txt'
data1 = pd.read_csv(file1,delim_whitespace=True)

# Extract x and y data
x1 = data1.iloc[:, 0]
y1 = data1.iloc[:, 8]
p_r = 500
p1 =y1/500
plt.plot(x1,p1,'b')
plt.title('Pressure variation along wall')
plt.xlabel('distance(m)')
plt.ylabel('P/Pr (Pa)')
plt.show()