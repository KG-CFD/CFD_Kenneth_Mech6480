import numpy as np
from matplotlib import pyplot as plt
data = np.genfromtxt('curvedata.txt')
y =data[:,1]
x = data[:,0]
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of data from Data file')
plt.savefig('Figure 1.png')
plt.show()