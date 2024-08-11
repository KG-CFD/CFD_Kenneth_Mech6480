
import matplotlib.pyplot as plt
import numpy as np



def generate_matrix_dirichlet(n, dx, convective_flux, diffusive_conductance, boundary_left, boundary_right):
    F = convective_flux
    D = diffusive_conductance

    a_WW = -1 / 8 * F
    a_W = D + 7 / 8 * F
    a_E = D - 3 / 8 * F
    a_P = a_W + a_E + a_WW

    # Constructing the matrix
    matrix_A = np.diag(np.repeat(-a_WW, n - 2), -2) + \
               np.diag(np.repeat(-a_W, n - 1), -1) + \
               np.diag(np.repeat(a_P, n), 0) + \
               np.diag(np.repeat(-a_E, n - 1), 1)

    S_p_left = (8 / 3 * D + 10 / 8 * F)
    S_p_right = (8 / 3 * D - F)

    # Adjust the boundary conditions
    matrix_A[0, 0] = F / 3 + a_E + S_p_left
    matrix_A[0, 1] -= F / 3

    # One CV in from left boundary CV
    matrix_A[1, 0] = -(D + F)
    matrix_A[1, 2] = -(D - 3 / 8 * F)
    matrix_A[1, 1] = -matrix_A[1, 0] - matrix_A[1, 2] - F / 4

    # Right boundary CV
    matrix_A[-1, -2] = -(4 / 3 * D + 7 / 8 * F)
    matrix_A[-1, -1] = a_WW - matrix_A[-1, -2] + S_p_right

    # Vector b (source term)
    vector_b = np.zeros(n)
    vector_b[0] = S_p_left * boundary_left
    vector_b[1] = -F / 4
    vector_b[-1] = S_p_right * boundary_right

    return matrix_A, vector_b


# SYSTEM PARAMETERS
Gamma = 0.1  # kg/m.s
u = 0.2  # m/s
rho = 1.0  # kg/m^3
length = 1  # m

# BOUNDARIES:
phi_0 = 1
phi_L = 0

# GRID GENERATION
num_cells = 5  # [-]
dx = length / (num_cells)  # [m]
x_locations = np.linspace(0.5 * dx, (num_cells - 0.5) * dx, num_cells)

# CONVECTIVE AND DIFFUSIVE TERMS
F = rho * u
D = Gamma / dx

# SYSTEM OF EQUATIONS
solution_matrix, source_matrix = generate_matrix_dirichlet(num_cells, dx, F, D, phi_0, phi_L)
concentration = np.linalg.solve(solution_matrix, source_matrix)


# PLOTTING
def analytic_solution(x):
    return phi_0 + (phi_L - phi_0) * (np.exp(rho * u * x / Gamma) - 1) / (np.exp(rho * u * length / Gamma) - 1)


x = np.linspace(0, length, 100)
solution = analytic_solution(x)
fig, ax = plt.subplots()
ax.plot(x, solution, 'r--', x_locations, concentration, 'b-o')

plt.title('QUICK scheme vs Analytical solution for 1D Conv-diffusion')
plt.xlabel('Distance along hallway (m)')
plt.ylabel('Concentration')
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)



plt.show()

