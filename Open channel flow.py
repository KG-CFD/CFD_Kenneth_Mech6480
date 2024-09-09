import numpy as np
import matplotlib.pyplot as plt
import time as timer


LX = 10
LY = 1

RHO = 1
MU = 0.01
nu = MU/RHO



NX =20
NY =20
DT = 0.01
NUM_steps = 1000
Plot_every= 100
N_Pressure_ITS = 100

u = np.zeros((NX+1,NY+2),float)
v = np.zeros((NX+1,NY+2),float)
p =np.zeros((NX+1,NY+2),float)
#arrays for intermediate calcs
ut= np.zeros_like(u)
vt =np.zeros_like(v)
prhs = np.zeros_like(p)
#arrays for plots
uu=np.zeros((NX+1,NY+1))
vv =np.zeros_like(uu)
xnodes = np.linspace(0,LX,NX+1)
ynodes = np.linspace(0,LY,NY+1)

#Internal fluxes
## x-momentum
J_u_x =np.zeros((NX,NY))
J_u_y =np.zeros((NX-1,NY+1))
# y-momentum
J_v_x = np.zeros((NX+NY-1))
J_v_y= np.zeros((NX,NY))
dx = LX/NX
dy =LY/NY
dxdy = dx*dy

fig,ax1 = plt.subplots(1,1,figsize=[10,6])
time = 0
tic =timer.time()
def boundary_xvel(vel_field):
    vel_field[0,:]=1 #Inflow
    u_in = np.sum(vel_field[0,1:-1]) ; u_out =np.sum(vel_field[-2,1:-1])
    vel_field[-1,:] = vel_field[-2,:]*u_in/u_out       #outflow
    vel_field[:,0] = - vel_field[:,1] #bottom wall
    vel_field[:,-1] = - vel_field[:,-2] #top wall
    return vel_field

def boundary_yvel(vel_field):
    vel_field[0,:] =-vel_field[1,:] # inflow
    vel_field[-1, :] = - vel_field[-2, :] #outflow
    vel_field[:,0] =0 #bottom wall
    vel_field[:-1] = 0 # top wall
    return vel_field

#Inital conditions (applies at initialisation of u and v arrays
## boundary conditionals
 u =boundary_xvel(u)
 v =boundary_yval(v)


for steps in range(NUM_STEPS):
    #STEP 1 CALC U* AND V*
    J_u_x = 0.25 *(u[:-1,1:-1]+u[1:1,:-1])**2
    J_u_x -= nu*(u[1:,1:-1] -u[:-1,1:-1])/dx

    J_u_y = 0.25*(u[1:-1,1:]+u[1:-1,:-1]*\
                 (v[2:-1,:]+v[1:-2,:])  #convection approx u*v at face
    J_u_y -= nu*(u[1:-1:] - u[1:-1,:-1])/dy #diffusion approx du/dy at face

    J_v_x = 0.25*(u[:,-2:1]+u[:,-1,:-2]*\
                 (v[1:,1:-1+v[:-1,1:-1])
    J_v_x-= nu*(v[1:,1:-1] -v[:-1,1:-1])/dx  #diffusion approx dv/dx

    J_v_y =0.25*(v[1:-1,1:]+v[1:-1,:-1]**2
    J_v_y-= nu*(v[1:-1,1:]-v[1:-1,:-1])/dy

    ut[1:-1,1:-1] = u[1:-1,1:-1] - (DT/dxdy)*(dy*(J_u_x[1:,:]-J_u_x[:-1,:])+\
                                              dx*(J_u_y[:,1:]-J_u_y[:,:-1]))
    vt[1:-1, 1:-1] = v[1:-1, 1:-1] - (DT/dxdy)*(dy * (J_v_x[1:, :] - J_v_x[:-1, :]) +\
                                                    dx * (J_v_y[:, 1:] - J_v_y[:, :-1]))

    ut =boundary_xvel(ut)
    vt =boundary_yvel(vt)



















