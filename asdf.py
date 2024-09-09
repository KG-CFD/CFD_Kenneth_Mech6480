import numpy as np
import matplotlib.pyplot as plt
import time as timer

# Constants
LX = 10
LY = 1
RHO = 1
MU = 0.01
nu = MU / RHO

NX = 20
NY = 20
DT = 0.01
NUM_STEPS = 1000
Plot_every = 100
N_Pressure_ITS = 100

# Initialize fields
u = np.zeros((NX + 1, NY + 2), float)
v = np.zeros((NX + 2, NY + 1), float)
p = np.zeros((NX + 2, NY + 2), float)
ut = np.zeros_like(u)
vt = np.zeros_like(v)
prhs = np.zeros_like(p)

uu = np.zeros((NX + 1, NY + 1))
vv = np.zeros_like(uu)
xnodes = np.linspace(0, LX, NX + 1)
ynodes = np.linspace(0, LY, NY + 1)

# Internal fluxes
J_u_x = np.zeros((NX, NY))
J_u_y = np.zeros((NX, NY))
J_v_x = np.zeros((NX, NY))
J_v_y = np.zeros((NX, NY))

dx = LX / NX
dy = LY / NY
dxdy = dx * dy

fig, ax1 = plt.subplots(1, 1, figsize=[10, 6])
time = 0
tic = timer.time()


def boundary_xvel(vel_field):
    vel_field[0, :] = 1  # Inflow
    u_in = np.sum(vel_field[0, 1:-1])
    u_out = np.sum(vel_field[-2, 1:-1])

    if u_out == 0:
        u_out = 1e-6  # Avoid division by zero with a small number

    vel_field[-1, :] = vel_field[-2, :] * u_in / u_out  # Outflow
    vel_field[:, 0] = - vel_field[:, 1]  # Bottom wall
    vel_field[:, -1] = - vel_field[:, -2]  # Top wall
    return vel_field


def boundary_yvel(vel_field):
    vel_field[0, :] = -vel_field[1, :]  # Inflow
    vel_field[-1, :] = - vel_field[-2, :]  # Outflow
    vel_field[:, 0] = 0  # Bottom wall
    vel_field[:, -1] = 0  # Top wall
    return vel_field


def pressure_poisson(p, prhs, dx, dy, N_its):
    for _ in range(N_its):
        p[1:-1, 1:-1] = 0.25 * (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - prhs[1:-1, 1:-1])
        p = boundary_pressure(p)
    return p


def boundary_pressure(p):
    p[:, 0] = p[:, 1]  # Bottom wall
    p[:, -1] = p[:, -2]  # Top wall
    p[0, :] = p[1, :]  # Inflow
    p[-1, :] = p[-2, :]  # Outflow
    return p


def divergence(u, v, dx, dy):
    div = np.zeros((NX, NY))
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / dx + (v[2:-1, 1:-1] - v[:-2, 1:-1]) / dy
    return div


# Initial boundary conditions
u = boundary_xvel(u)
v = boundary_yvel(v)

for step in range(NUM_STEPS):
    # Step 1: Calculate intermediate velocities (u*, v*)
    J_u_x = 0.25 * (u[:-1, 1:-1] + u[1:, 1:-1]) ** 2
    J_u_x -= nu * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx

    J_u_y = 0.25 * ((u[1:-1, 1:] + u[1:-1, :-1]) * (v[2:-1, 1:] + v[1:-2, 1:]))
    J_u_y -= nu * (u[1:-1, 1:] - u[1:-1, :-1]) / dy

    J_v_x = 0.25 * (u[:-1, 1:-1] + u[:-1, 2:]) * (v[1:, 1:-1] + v[:-1, 1:-1])
    J_v_x -= nu * (v[1:, 1:-1] - v[:-1, 1:-1]) / dx

    J_v_y = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1]) ** 2
    J_v_y -= nu * (v[1:-1, 1:] - v[1:-1, :-1]) / dy

    ut[1:-1, 1:-1] = u[1:-1, 1:-1] - (DT / dxdy) * (
                dy * (J_u_x[1:, :] - J_u_x[:-1, :]) + dx * (J_u_y[:, 1:] - J_u_y[:, :-1]))
    vt[1:-1, 1:-1] = v[1:-1, 1:-1] - (DT / dxdy) * (
                dy * (J_v_x[1:, :] - J_v_x[:-1, :]) + dx * (J_v_y[:, 1:] - J_v_y[:, :-1]))

    ut = boundary_xvel(ut)
    vt = boundary_yvel(vt)

    # Step 2: Solve Poisson equation for pressure correction
    prhs[1:-1, 1:-1] = (ut[1:, 1:-1] - ut[:-1, 1:-1]) / dx + (vt[1:-1, 1:] - vt[1:-1, :-1]) / dy
    p = pressure_poisson(p, prhs, dx, dy, N_Pressure_ITS)

    # Step 3: Correct velocities with pressure gradient
    u[1:-1, 1:-1] = ut[1:-1, 1:-1] - DT * (p[2:, 1:-1] - p[1:-1, 1:-1]) / dx
    v[1:-1, 1:-1] = vt[1:-1, 1:-1] - DT * (p[1:-1, 2:] - p[1:-1, 1:-1]) / dy

    # Step 4: Check divergence
    div = divergence(u, v, dx, dy)
    max_div = np.max(np.abs(div))
    if max_div > 1e-6:
        print(f"Warning: Divergence not zero at step {step}, max divergence: {max_div}")

    # Plotting
    if step % Plot_every == 0:
        uu[:, :] = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
        vv[:, :] = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
        plt.cla()
        plt.quiver(xnodes, ynodes, uu.T, vv.T)
        plt.pause(0.01)

plt.show()