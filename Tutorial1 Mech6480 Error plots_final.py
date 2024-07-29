import numpy as np
from matplotlib import pyplot as plt
data = np.genfromtxt('curvedata.txt')
y =data[:,1]
x = data[:,0]
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of data from Data file')
plt.show()

# Numerical scheme for forward difference
dy_forward = []
dy_central =[]
dx_forward =[]
dx_central =[]
y_exact = -np.sin(x)
Error_central = []
Error_forward = []
Error_forward1 = []
n = len(x)
for  i in range(n-1):  # can't calculate derivative at last data point with forward difference scheme
    dy_forward.append((y[i+1]-y[i])/((x[i+1] - x[i])))
    dx_forward.append(x[i])

for i in range(1,n-1):
    dy_central.append((y[i+1] -y[i-1])/((x[i+1] - x[i-1])))
    dx_central.append(x[i])

for i in range(n-2):
    Error_central.append(( -y_exact[i+1] + dy_central[i]))
    
for i in range(n-1):    
    Error_forward.append((dy_forward[i] - y_exact[i]))
    
for i in range(n-1):    # Trying to get percentage errror
    if y_exact[i] != 0:
        Error_forward1.append((dy_forward[i] - y_exact[i])/(y_exact[i]) * 100)
    else:
        y_exact[i] == dy_forward[i]
        Error_forward1.append((dy_forward[i] - y_exact[i])/(y_exact[i]) * 100)
        
    
    
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(dx_forward,dy_forward, label = "Forward difference approx")
ax1.plot(dx_central,dy_central, label ="Central difference approx")
ax1.plot(x,y_exact, label ="Exact solution")
ax2.plot(dx_central,Error_central, label ="Central difference Error")
ax2.plot(dx_forward,Error_forward, label ="Forward differencce Error")         
ax1.legend()
ax2.legend()
plt.show()

plt.plot(dx_forward, Error_forward1)
plt.title('Forward difference Percentage error for cosine(x)')
plt.ylabel('Percentage error')
plt.show()