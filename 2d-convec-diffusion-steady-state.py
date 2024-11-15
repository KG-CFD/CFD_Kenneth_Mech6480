import numpy as np
import matplotlib.pyplot as plt
Lx =1
Ly =1
Nx = 40
Ny = 40
h = np.float64(Lx/ (Nx-1))

gamma =0.1
u = 4
pe =u*h/gamma
print('pec number is', pe)

A =0.001

T = np.zeros((Ny,Nx))
T[0,:] = 1
T[:,0] = 1


T_new =np.zeros((Ny,Nx))
T_new[0,:] =1
T_new[:,0] =1

iterations =0
epsilon =1e-4
numerical_error =1


while numerical_error > epsilon:
    for i in range(1,(Ny-1)):
        for j in range(1,Nx-1):
            a_E = np.float64(gamma/h - u/2)
            a_W =np.float64(gamma/h + u/2)
            a_N =np.float64(gamma/h )
            a_S =np.float64(gamma / h)
            a_P =a_E +a_W + a_S +a_N
            T_new[i,j] =(a_E*T[i,j+1] +a_W*T[i,j-1]+a_N*T[i-1,j]+a_S*T[i+1,j])/a_P

    numerical_error =0
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            numerical_error= numerical_error + abs(T[i,j] - T_new[i,j] )

    iterations = iterations + 1
    T =T_new.copy()

    if iterations%200 ==0:
        plt.figure(10)
        plt.semilogy(iterations,numerical_error,'ko')
        plt.pause(0.01)

x_dom =np.arange(Nx)*h
y_dom = Ly-np.arange(Ny)*h
[X,Y] = np.meshgrid(x_dom,y_dom)

plt.figure(11)
plt.contourf(X,Y,T,12)
plt.colorbar(orientation ='vertical')
plt.title('Temperature in convected flow')
plt.show()