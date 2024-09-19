import numpy as np
import matplotlib.pyplot as plt
import time as timer

LX = 0.6 # Horizontal length in Metres
LY = 0.03 # Vertical Outlet height in metres
H =0.03   # vertical Outlet height in metres
h_inlet = 0.02   # inlet Height
A_r = h_inlet/H   # Ratio of Areas
h_step = 0.01  # Step height
w_step = 0.22  # Width of Step
RHO = 998.  # Density in Kg per m^3
MU = 0.001
Re =230
nu = MU / RHO
U_inlet = Re*MU/(RHO*h_inlet) #0.011523  #ms-1
print(U_inlet)
NX = 900
NY = 45
Step_cell_y =  int((h_step/LY)*NY)
Step_cell_x = int((w_step/LX)*NX)
DT = 0.001          #time step
NUM_STEPS = 20000
PLOT_EVERY = 20000
N_PRESSURE_ITS = 100

u = np.zeros((NX+1, NY+2), float)
v = np.zeros((NX+2, NY+1), float)
p = np.zeros((NX+2, NY+2), float)
ut = np.zeros_like(u)
vt = np.zeros_like(v)
prhs = np.zeros_like(p)

uu = np.zeros((NX+1, NY+1))
vv = np.zeros_like(uu)
xnodes = np.linspace(0, LX, NX+1)
ynodes = np.linspace(0, LY, NY+1)

J_u_x = np.zeros((NX, NY))
J_u_y = np.zeros((NX-1, NY+1))
J_v_x = np.zeros((NX+1, NY-1))
J_v_y = np.zeros((NX, NY))

dx = LX / NX     #grid spacing
dy = LY / NY     #grid spacing
dxdy = dx * dy

fig, ax1 = plt.subplots(1, 1, figsize=[15, 300])


def boundary_xvel(vel_field):   # function that implements Boundary conditions
    vel_field[0,Step_cell_y +1:-1] = U_inlet  # Inflow conditions for y>0.01m
    vel_field[0, 0:(Step_cell_y+1)] = 0.0                          # Inflow I.c for y<0.01m
    u_in = np.sum(vel_field[0, (Step_cell_y +1):-1])  ; u_out = np.sum(vel_field[-2,1:-1])
    epsilon =1e-10
    if u_out == 0:
        u_out = epsilon
    vel_field[-1, :] = vel_field[-2, :] *A_r* u_in/u_out  # Outflow Balancing of mass With different areas
    vel_field[(Step_cell_x +1):-1 ,0] = - vel_field[(Step_cell_x + 1):-1, 1]  # Bottom wall
    vel_field[:, -1] = - vel_field[:, -2]  # Top wall
    vel_field[1:Step_cell_x,Step_cell_y] = -vel_field[1:Step_cell_x, (Step_cell_y + 1)]         # Horizontal step Edge
    vel_field[(Step_cell_x),1:(Step_cell_y+1)]  = 0.0           #Vertical step Edge 1
    vel_field[:Step_cell_x,:Step_cell_y] =0.0  # setting all Xvel values in step to zero                                                #Values inside of step excluding ghost cells
    return vel_field

def boundary_yvel(vel_field):
    vel_field[0, :] = -vel_field[1, :]  # Inflow
    vel_field[0,(Step_cell_y+1) :-1] = -vel_field[1,(Step_cell_y +1):-1]
    vel_field[-1, :] = - vel_field[-2, :]  # Outflow
    vel_field[(Step_cell_x+1):, 0] = 0  # Bottom wall
    vel_field[:, -1] = 0  # Top wall
    vel_field[1:(Step_cell_x+1),Step_cell_y] = 0 #Horizontal step Edge Boundary
    vel_field[1:Step_cell_x,1:(Step_cell_y+1)]= -vel_field[(Step_cell_x+1),1:Step_cell_y+1]   # vertical step edge Boundary
    vel_field[:Step_cell_x,Step_cell_y]= 0.0    #setting all velocities to zero in step area
    return vel_field


# Initial boundary conditions
u = boundary_xvel(u)
v = boundary_yvel(v)
time = 0
tic = timer.time()
for steps in range(NUM_STEPS):
    # Step 1: Calculate intermediate velocities (u*, v*)
    J_u_x = 0.25 * (u[:-1, 1:-1] + u[1:, 1:-1]) ** 2
    J_u_x -= nu * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx

    J_u_y = 0.25 * ((u[1:-1, 1:] + u[1:-1, :-1]) * (v[2:-1, :] + v[1:-2,:]))  # convection approx u*v at face
    J_u_y -= nu * (u[1:-1, 1:] - u[1:-1, :-1]) / dy  # diffusion approx du/dy at face

    J_v_x = 0.25 * (u[:,2:-1] + u[:, 1:-2]) * (v[1:, 1:-1] + v[:-1, 1:-1])
    J_v_x -= nu * (v[1:, 1:-1] - v[:-1, 1:-1])/dx  # diffusion approx dv/dx

    J_v_y = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1])** 2
    J_v_y -= nu * (v[1:-1, 1:] - v[1:-1, :-1])/ dy

    ut[1:-1, 1:-1] = u[1:-1, 1:-1] - (DT / dxdy) * (dy * (J_u_x[1:, :] - J_u_x[:-1, :]) + dx * (J_u_y[:, 1:] - J_u_y[:, :-1])) # Tentative velocity div free
    vt[1:-1, 1:-1] = v[1:-1, 1:-1] - (DT / dxdy) * (dy * (J_v_x[1:, :] - J_v_x[:-1, :]) + dx * (J_v_y[:, 1:] - J_v_y[:, :-1]))  #tentative velocity

    ut = boundary_xvel(ut)
    vt = boundary_yvel(vt)
    divergence = (ut[1:, 1:-1] - ut[:-1, 1:-1])/dx + (vt[1:-1,1:] - vt[1:-1, :-1])/dy
    # Step 2: Solve Poisson equation for pressure correction
    prhs =divergence* RHO/DT
    p_next =np.zeros_like(p)
    for _ in range(N_PRESSURE_ITS):
        p_next[1:-1, 1:-1] =  (-prhs*dxdy**2 +dy**2*(p[2:, 1:-1] + p[:-2, 1:-1]) + dx**2 *(p[1:-1, 2:] + p[1:-1, :-2]))\
                              /(2*dx**2 +2*dy**2)
        p_next[0,(Step_cell_y+1):-1] =p_next[1,(Step_cell_y+ 1 ):-1]    #inflow B.C
        p_next[-1, :] = p_next[-2,:]     #outflow B.C   # using dirichlet B.C
        p_next[(Step_cell_x+1):-1, 0] = p_next[(Step_cell_x+1):-1,1]     # Bottom wall B.C
        p_next[:,-1] = p_next[:,-2]   #Top wall B.C
        p_next[1:(Step_cell_x+1),Step_cell_y] =p_next[1:(Step_cell_x+1),(Step_cell_y+1)]       #Top of step B.C
        p_next[Step_cell_x,1:(Step_cell_y+1)] = p_next[(Step_cell_x+1),1:(Step_cell_y+1)]   #Right edge of step
        p_next[:Step_cell_x,:Step_cell_y] = 0.0                          #Setting all pressure val in step to 0
        p = p_next.copy()
    # Step 3: Correct velocities with pressure gradient
    u[1:-1, 1:-1] = ut[1:-1, 1:-1] - DT*(1/dx) * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / RHO
    v[1:-1, 1:-1] = vt[1:-1, 1:-1] - DT*(1/dy) * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / RHO
    #apply B.C
    u = boundary_xvel(u)
    v = boundary_yvel(v)


    time =time +DT
    if ((steps+1) %PLOT_EVERY ==0):
        divu = (u[1:, 1:-1] - u[:-1, 1:-1])/dx + (v[1:-1, 1:] - v[1:-1, :-1])/dy
        toc =timer.time()
        print(f"Step {steps}, normof div(u):{np.linalg.norm(divu):.4e}. \n Sec per it ={(toc-tic)/(steps+1):.4e}")
        # interpolate velocity field to consistent locations
        uu =0.5*(u[0:NX+1,1:NY+2] + u[0:NX+1, 0:NY+1])
        vv = 0.5*(v[1:NX+2, 0:NY+1] +v[0:NX+1, 0:NY+1])
        uu[:(Step_cell_x +1),:(Step_cell_y+1)] = 0.0   #setting all values in step to zero
        vv[:(Step_cell_x +1),:(Step_cell_y+1)] =0.0    #setting all values in step to zero
        xx,yy =np.meshgrid(xnodes,ynodes,indexing='ij')
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))  # Adjust width and height as needed
        ax1.contourf(xx, yy, np.sqrt(uu ** 2 + vv ** 2), levels=30, cmap='viridis')
        ax1.set_title(f'Flow Contour at Time (Time = {time:.3f} s)')
        ax1.set_aspect('equal', 'box')

        # Add colorbar
        cbar = fig.colorbar(ax1.contourf(xx, yy, np.sqrt(uu ** 2 + vv ** 2), levels=30, cmap='viridis'), ax=ax1)
        cbar.set_label('Velocity Magnitude (m/s)')




plt.show()
x_nodes = np.linspace(0, LX, NX + 1)
y_nodes = np.linspace(0, LY, NY + 1)
# Now, U, V have the same shape as xnodes, ynodes (NX+1, NY+1)
plt.figure(figsize=(8, 6))
plt.streamplot(x_nodes, y_nodes, uu.T, vv.T, density=1.5, linewidth=1, arrowsize=1, cmap='cool')
plt.title(f'Streamlines of Velocity Field at Time (Time = {time:.3f} s)')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, LX)
plt.ylim(0, LY)
plt.gca().set_aspect('equal', 'box')
plt.show()
x_pos_1 = Step_cell_x - 1  # One control volume before the end of the step
x_pos_2 = int((w_step + 0.06) / LX * NX)  # Roughly 6 cm after the step

# Extract  velocity for both positions
u_slice_1 = uu[x_pos_1, :]
u_slice_2 = uu[x_pos_2, :]

# Create plots
plt.figure(figsize=(10, 6))

# Plot horizontal velocity one control volume before the end of the step
plt.subplot(2, 1, 1)
plt.plot(ynodes, u_slice_1, label=f'X = {xnodes[x_pos_1]:.2f} m', color='blue')
plt.title('Horizontal Velocity (u) 1 Control Volume Before the End of the Step')
plt.xlabel('Y Position (m)')
plt.ylabel('Horizontal Velocity (u) (m/s)')
plt.grid(True)
plt.legend()

# Plot horizontal velocity 6 cm after the step
plt.subplot(2, 1, 2)
plt.plot(ynodes, u_slice_2, label=f'X = {xnodes[x_pos_2]:.2f} m', color='green')
plt.title('Horizontal Velocity (u) 6 cm After the Step')
plt.xlabel('Y Position (m)')
plt.ylabel('Horizontal Velocity (u) (m/s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()