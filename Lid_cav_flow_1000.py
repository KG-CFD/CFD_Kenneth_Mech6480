import numpy as np
import matplotlib.pyplot as plt
import time as timer

LX = 1
LY = 1
RHO = 1.
Re = 1000
U_North = 1.0
U_South = 0.
V_East = 0.
V_West = 0.
MU = RHO* U_North *LX / Re
nu=MU/RHO
NX = 40
NY = 40
DT = 0.001
NUM_STEPS = 40000
PLOT_EVERY = 10000
N_PRESSURE_ITS = 150


data1 = np.loadtxt('Lid_Cav_Yvelocities_1000.txt', delimiter=' ', usecols=(0, 3))
data2 = np.loadtxt('Lid_Cav_Xvelocities_1000.txt', delimiter=' ', usecols=(0, 3))

u = np.zeros((NX + 1, NY + 2), float)
v = np.zeros((NX + 2, NY + 1), float)
p = np.zeros((NX + 2, NY + 2), float)
u_prev = np.copy(u)
v_prev = np.copy(v)
ut = np.zeros_like(u)
vt = np.zeros_like(v)
prhs = np.zeros_like(p)

uu = np.zeros((NX + 1, NY + 1))
vv = np.zeros_like(uu)
xnodes = np.linspace(0, LX, NX + 1)
ynodes = np.linspace(0, LY, NY + 1)

J_u_x = np.zeros((NX, NY))
J_u_y = np.zeros((NX - 1, NY + 1))
J_v_x = np.zeros((NX + 1, NY - 1))
J_v_y = np.zeros((NX, NY))

dx = LX / NX
dy = LY / NY
dxdy = dx * dy

fig, ax1 = plt.subplots(1, 1)


def boundary_xvel(vel_field):
    vel_field[:, 0] = 2 * U_South - vel_field[:, 1]  # Bottom wall
    vel_field[:, -1] = 2 * U_North - vel_field[:, -2]  # Top wall
    return vel_field


def boundary_yvel(vel_field):
    vel_field[0, :] = 2 * V_East - vel_field[1, :]  # Left  wall
    vel_field[-1, :] = 2 * V_West - vel_field[-2, :]  # Right wall
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

    J_u_y = 0.25 * ((u[1:-1, 1:] + u[1:-1, :-1]) * (v[2:-1, :] + v[1:-2, :]))  # convection approx u*v at face
    J_u_y -= nu * (u[1:-1, 1:] - u[1:-1, :-1]) / dy  # diffusion approx du/dy at face

    J_v_x = 0.25 * (u[:, 2:-1] + u[:, 1:-2]) * (v[1:, 1:-1] + v[:-1, 1:-1])
    J_v_x -= nu * (v[1:, 1:-1] - v[:-1, 1:-1]) / dx  # diffusion approx dv/dx

    J_v_y = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1]) ** 2
    J_v_y -= nu * (v[1:-1, 1:] - v[1:-1, :-1]) / dy

    ut[1:-1, 1:-1] = u[1:-1, 1:-1] - (DT / dxdy) * (
            dy * (J_u_x[1:, :] - J_u_x[:-1, :]) + dx * (J_u_y[:, 1:] - J_u_y[:, :-1]))
    vt[1:-1, 1:-1] = v[1:-1, 1:-1] - (DT / dxdy) * (
            dy * (J_v_x[1:, :] - J_v_x[:-1, :]) + dx * (J_v_y[:, 1:] - J_v_y[:, :-1]))

    ut = boundary_xvel(ut)
    vt = boundary_yvel(vt)
    divergence = (ut[1:, 1:-1] - ut[:-1, 1:-1]) / dx + (vt[1:-1, 1:] - vt[1:-1, :-1]) / dy
    # Step 2: Solve Poisson equation for pressure correction
    prhs = divergence * RHO / DT
    p_next = np.zeros_like(p)
    for _ in range(N_PRESSURE_ITS):
        p_next[1:-1, 1:-1] = (-prhs * dxdy ** 2 + dy ** 2 * (p[2:, 1:-1] + p[:-2, 1:-1]) + dx ** 2 * (
                p[1:-1, 2:] + p[1:-1, :-2])) \
                             / (2 * dx ** 2 + 2 * dy ** 2)
        p_next[0, :] = p_next[1, :]
        p_next[-1, :] = -p_next[-2, :]
        p_next[:, 0] = p_next[:, 1]
        p_next[:, -1] = p_next[:, -2]
        p = p_next.copy()
    # Step 3: Correct velocities with pressure gradient
    u[1:-1, 1:-1] = ut[1:-1, 1:-1] - DT * (1. / dx) * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / RHO
    v[1:-1, 1:-1] = vt[1:-1, 1:-1] - DT * (1. / dy) * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / RHO
    # apply B.C
    u = boundary_xvel(u)
    v = boundary_yvel(v)

    time = time + DT
    if ((steps + 1) % PLOT_EVERY == 0):
        divu = (u[1:, 1:-1] - u[:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
        toc = timer.time()
        print(f"Step {steps}, norm of div(u): {np.linalg.norm(divu)}. \n Sec per it = {(toc - tic) / (steps + 1):.4e}")
        # Compute residuals for u and v
        u_residual = np.sum(np.abs(u - u_prev)) / np.sum(np.abs(u))
        v_residual = np.sum(np.abs(v - v_prev)) / np.sum(np.abs(v))

        # Update previous velocity fields
        u_prev = np.copy(u)
        v_prev = np.copy(v)

        # Print residuals at each plotting step
        print(f"Step {steps + 1}, u residual: {u_residual:.4e}, v residual: {v_residual:.4e}")
        # Interpolate velocity field to the correct locations (as you have NX + 1, NY + 1 points)
        uu = 0.5 * (u[0:NX + 1, 1:NY + 2] + u[0:NX + 1, 0:NY + 1])  # u velocity at cell centers
        vv = 0.5 * (v[1:NX + 2, 0:NY + 1] + v[0:NX + 1, 0:NY + 1])  # v velocity at cell centers

        xx, yy = np.meshgrid(xnodes, ynodes, indexing='ij')

        magnitude = np.sqrt(uu ** 2 + vv ** 2)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

        # Plot velocity magnitude contour
        c1 = ax1.contourf(xx, yy, magnitude, levels=50, cmap='viridis')
        fig.colorbar(c1, ax=ax1)
        ax1.set_title(f'Velocity Magnitude (Time = {time:.3f} s)')
        ax1.quiver(xx, yy, uu, vv)
        # Set aspect ratio to be equal for square grid
        ax1.set_aspect('equal', 'box')

        # Plot vertical velocity at y = LY / 2
        x_values = data1[:, 0]
        y_values = data1[:, 1]
        ax2.plot(x_values, y_values, marker='o')
        y_index = np.argmin(np.abs(ynodes - LY / 2))
        v_at_y = vv[:, y_index]  # Extracting velocity along x
        x_mid = xnodes[:]  # Corresponding x-coordinates
        ax2.plot(x_mid, v_at_y, label='v at y = LY/2', color='b')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Vertical Velocity (v)')
        ax2.set_title(f'Vertical Velocity at y = LY/2 (Time = {time:.3f} s)')

        # Plot horizontal velocity at x = LX / 2
        x_values = data2[:, 0]
        y_values = data2[:, 1]
        ax3.plot(y_values, x_values, marker='o')
        x_index = np.argmin(np.abs(xnodes - LX / 2))
        u_at_x = uu[x_index, :]  # Extracting velocity along y
        y_mid = ynodes[:]  # Corresponding y-coordinates
        ax3.plot(u_at_x, y_mid, label='u at x = LX/2', color='r')
        ax3.set_xlabel('Horizontal Velocity (u)')
        ax3.set_ylabel('y')
        ax3.set_title(f'Horizontal Velocity at x = LX/2 (Time = {time:.3f} s)')

        plt.tight_layout()
plt.show()


x_nodes = np.linspace(0, LX, NX + 1)
y_nodes = np.linspace(0, LY, NY + 1)
# Now, U, V have the same shape as xnodes, ynodes (NX+1, NY+1)
plt.figure(figsize=(8, 6))
plt.streamplot(x_nodes, y_nodes, uu.T, vv.T, density=1.5, linewidth=1, arrowsize=1, cmap='cool')
plt.title(f'Streamlines of Velocity Field at Final Time (Time = {time:.3f} s)')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, LX)
plt.ylim(0, LY)
plt.gca().set_aspect('equal', 'box')
plt.show()
