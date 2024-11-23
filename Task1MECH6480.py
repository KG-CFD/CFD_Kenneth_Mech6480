import numpy as np
import matplotlib.pyplot as plt

# Number of grid points in x and y directions
Nx = 401
Ny = 41

# Domain size in x and y directions
Lx = 0.1
Ly = 0.01

# Corresponding grid spacing
hx = Lx / (Nx-1)
hy = Ly / (Ny-1)

# Time parameters
t_final = 15 # Final simulation time
dt = 0.005      # Time step size
max_temperature = []  # List to store the max temperature at each time step
time_steps = []  # List to store the time at each step

# Velocity profile
U_max = 0.031262525 #m/s

# Parabolic velocity profile
y_dom = np.linspace(0, Ly, Ny)
u = U_max * (1 - ((y_dom - Ly/2) / (Ly/2))**2)


# Physical properties
rho = 1000        # Density kg/m^3
cp= 4186          # specific heat
k= 0.6            # W/mk
gamma = k /(rho*cp )     # Diffusivity
q_top = 5000   # W/m^2
q_bottom =5000 #W/m^2
# Initializing time variable
t = 0.0

# Initializing temperature field (Nx by Ny grid)
T = np.ones((Nx, Ny))*300.0
T[0, :] = 300  # Dirichlet BC on the left boundary
T[-1, :] = T[-2,:] # Neumann BC on the right boundary
T[:, 0] = T[:, 1] + 2*(q_bottom * hx / k)  # heat flux B.C
T[:, -1] =  T[:, -2] + 2*(q_bottom * hx / k)  #Heat flux B.C

# Initialize new temperature field
T_new = np.ones_like(T)*300
T_new[0, :] = 300.0
T_new[-1, :] =  T_new[-2,:]
T_new[:, -1] = T_new[:, -2] + (q_top * hx / k)    
# Heat flux boundary condition for the bottom boundary
T_new[:, 0] = T_new[:, 1] + (q_bottom * hx / k)

# CFL condition 
CFL_x = u * dt / hx
alpha_x = gamma * dt / (hx**2)
alpha_y = gamma * dt / (hy**2)
print('CFL',CFL_x)
print('Alpha',alpha_x)
# Plot setup


# Time-stepping loop
while t < t_final:
    # Update the interior points
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # Diffusion terms (in x and y directions)
            diffusion_x = alpha_x * (T[i+1, j] - 2*T[i, j] + T[i-1, j])
            diffusion_y = alpha_y * (T[i, j+1] - 2*T[i, j] + T[i, j-1])
            
            # Convection terms (in x and y directions)
            convection_x = -u[j] * dt / hx * (T[i, j] - T[i-1, j])
            
            
            # Update temperature
            T_new[i, j] = T[i, j] + diffusion_x + diffusion_y + convection_x 
    
    T_new[0, :] = 300.0
    #update for Dirichlet B.C
    T_new[-1, :] = T_new[-2, :]
    # Heat flux boundary condition for the top boundary
    T_new[:, -1] = T[:, -2] + 2*(q_top * hx / k)
    
    # Heat flux boundary condition for the bottom boundary
    T_new[:, 0] = T[:, 1] + 2*(q_bottom * hx / k)
    # Update the temperature field
    T = T_new.copy()
    
    max_temperature.append(np.max(T))
    time_steps.append(t)
    # Advance time
    t += dt

    
# Final plot

plt.figure(figsize=(16, 4))  # Adjust the figure size to reflect the aspect ratio of Lx/Ly
plt.contourf(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny), T.T, cmap="jet", levels=30)
plt.colorbar(label="Temperature")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Final Temperature Distribution T(x, y) at t = {t_final:.2f} s")
plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio matches the physical dimensions
plt.show()

plt.figure(2)
# Plot temperature profiles at different x locations
plt.plot(y_dom, T[(Nx-1)//2, :], label="x = 0.5L")  
plt.plot(y_dom, T[(Nx-1)//4, :], label="x = 0.25L")  
plt.plot(y_dom, T[(Nx-1)//8, :], label="x = 0.125L")  

plt.title('Temperature Profiles at Different x Locations')
plt.xlabel('Y-Domain')
plt.ylabel('Temperature(K)')
plt.legend(loc="best")  
plt.grid(True)

plt.show()
plt.figure(3)
plt.plot(time_steps, max_temperature, label='Max Temperature', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Maximum Temperature (K)')
plt.title('Maximum Temperature in Domain Over Time')
plt.grid(True)
plt.legend()
plt.show()

