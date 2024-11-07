import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from git import Repo


def Matrix_A(n, dx,m, Tboundary_left):


    # Internal coefficients
    a_W = 1
    a_E = 1
    a_P = (2 + m*dx*dx)

    # Construct the main matrix
    matrix = np.zeros((n, n))

    # Populate the main diagonals
    np.fill_diagonal(matrix, -a_P)
    np.fill_diagonal(matrix[1:], a_W)
    np.fill_diagonal(matrix[:, 1:], a_E)

    # Adjust boundary rows for Dirichlet conditions as a function of D
    matrix[0, 0] = -3 -m*dx*dx
    matrix[0, 1] =  1

    matrix[-1, -1] = -1 - m*dx*dx
    matrix[-1, -2] =  1


    # Create the source term vector with boundary conditions
    vector_b = np.zeros(n)
    vector_b[0]= -2*Tboundary_left



    return matrix, vector_b


# SYSTEM PARAMETERS

length = 0.5 # meters
h = 34
t= 0.9
k=401
A =0.02
# BOUNDARIES
Theta_wall = 150


# GRID GENERATION
num_cells = 56
dx = length / num_cells
x_locations = np.linspace(0.5 * dx, length - 0.5 * dx, num_cells)
m = h*t/(A*k)

# Solve the system
solution_matrix, source_matrix = Matrix_A(num_cells, dx,m, Theta_wall)
concentration = np.linalg.solve(solution_matrix, source_matrix)

# Plotting
x = np.linspace(0, length, 100)
fig, ax = plt.subplots()
ax.plot(x_locations, concentration, 'b-o')
plt.title('FVM Solution for 1D fin heat transfer')
plt.xlabel('Distance (m)')
plt.ylabel('Temperature')

# Add timestamp and revision ID
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)

repo = Repo('.', search_parent_directories=True)
revsha = repo.head.object.hexsha[:8]
ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)

plt.show()

