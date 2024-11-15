# plot of nozzle pressure at entrance and exit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file1 = 'nozzle_flow-blk-0-cell-2205.dat.0'
data1 = pd.read_csv(file1,delim_whitespace=True)

# Extract x and y data
x1 = data1.iloc[:, 0]
y1 = data1.iloc[:, 9]


plt.plot(x1,y1,'b')
plt.title('Pressure variation at inlet over time')
plt.xlabel('Time')
plt.ylabel('Pressure (Pa)')

plt.show()