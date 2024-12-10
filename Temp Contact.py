import matplotlib.pyplot as plt
import numpy as np
import time


LX =10
LY = 10
Alpha =4
R =2 ;r2 = R**2
CX =5
CY=5


NX =100
NY =100
dx =LX/NX
dy =LY/NY

x=np.linspace(dx/2.,LX -dx/2.0,NX) # C.V centres
y=np.linspace(dy/2.,LY -dy/2.0,NY)
xx,yy=np.meshgrid(x,y,indexing='ij')


#Time
sim_time =0.
iteration = 0.

T_end =62.5e-3
DT =0.0000625
Plot_Every = 10

# I.C
T_cool =300.0
T_Hot = 700.0
T_wall =300.0
T= np.ones((NX,NY))*T_cool

for i in range(NX):
    for j in range(NY):
        if ((xx[i,j]-CX)**2 +(yy[i,j]-CY)**2)< r2:
            T[i,j] = T_Hot
            
           
# allocate memory for fluxes
x_flux = np.zeros((NX+1,NY))
y_flux =np.zeros((NX,NY+1))

#Plot set up
fig = plt.figure(figsize =(8,8))
gs = fig.add_gridspec(1,1)
ax0=fig.add_subplot(gs[0,0])

tic= time.time()
while sim_time < T_end:
    #Fluxes interior
    x_flux[1:-1,:] = Alpha*(T[1:,:] -T[:-1,:])/dx
    y_flux[:,1:-1] = Alpha*(T[:,1:]-T[:,:-1])/dy
    #Fluxes Boundaries
    x_flux[0,:] =Alpha*(T[0,:] -T_wall)/(dx/2.)
    x_flux[-1,:] = Alpha *(T_wall -T[-1,:])/(dx/2.0)
    y_flux[:,0] = Alpha * (T[:,0] -T_wall)/(dy/2.)
    y_flux[:,-1] = Alpha *(T_wall -T[:,-1])/(dy/2.0)
    
    
    T = T + DT *(dy*(x_flux[1:,:]-x_flux[:-1,:])\
                 +dx*(y_flux[:,1:] -y_flux[:,:-1]))/(dx*dy)
   

#plotting
    if np.mod(iteration,Plot_Every)==0:
        ax0.cla()
        ax0.contourf(xx,yy,T,Vmin =T_cool, Vmax= T_Hot)
        ax0.set_xlabel('x'); ax0.set_ylabel('y')
        ax0.set_title('Temperature')
        ax0.set_aspect('equal')
        plt.pause(0.1)

    sim_time +=DT
    iteration+=1
    
#plot final T-distribution
    
    
    
