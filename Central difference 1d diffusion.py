

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from git import Repo


def Matrix_A(n, dx, diffusive_conductance, Tboundary_left, Tboundary_right):
    D = diffusive_conductance

    # Internal coefficients
    a_W = D
    a_E = D
    a_P = 2 * D  # Ensures balance for each cell

    # Construct the main matrix
    matrix = np.zeros((n, n))

    # Populate the main diagonals
    np.fill_diagonal(matrix, -a_P)
    np.fill_diagonal(matrix[1:], a_W)
    np.fill_diagonal(matrix[:, 1:], a_E)

    # Adjust boundary rows for Dirichlet conditions as a function of D
    matrix[0, 0] = -3 * D
    matrix[0, 1] =  D

    matrix[-1, -1] = -3 * D
    matrix[-1, -2] =  D


    # Create the source term vector with boundary conditions
    vector_b = np.zeros(n)
    vector_b[0]= -2*D*Tboundary_left
    vector_b[-1] = -2*D *Tboundary_right


    return matrix, vector_b


# SYSTEM PARAMETERS
Gamma = 1
length = 2  # meters

# BOUNDARIES
Temp_L = 100
Temp_R = 300

# GRID GENERATION
num_cells = 16
dx = length / num_cells
x_locations = np.linspace(0.5 * dx, length - 0.5 * dx, num_cells)

# Diffusive conductance
D = Gamma / dx

# Solve the system
solution_matrix, source_matrix = Matrix_A(num_cells, dx, D, Temp_L, Temp_R)
concentration = np.linalg.solve(solution_matrix, source_matrix)

# Plotting
x = np.linspace(0, length, 100)
fig, ax = plt.subplots()
ax.plot(x_locations, concentration, 'b-o')
plt.title('FVM Solution for 1D Heat Transfer with Dirichlet Boundaries')
plt.xlabel('Distance along hallway (m)')
plt.ylabel('Temperature')

# Add timestamp and revision ID
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)

repo = Repo('.', search_parent_directories=True)
revsha = repo.head.object.hexsha[:8]
ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)

plt.show()

