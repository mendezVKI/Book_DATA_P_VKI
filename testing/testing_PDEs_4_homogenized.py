import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import imageio
import os

# Parameters
alpha = 1.0  # Thermal diffusivity
beta = 1.0   # Nonlinear coefficient
L = 2.0      # Length of the domain [-1, 1]
N = 100      # Number of spatial points
dx = L / N   # Spatial step size
dt = 0.0001  # Time step size
T = 2        # Total time
steps = int(T / dt)  # Number of time steps

# Spatial grid
x = np.linspace(-1, 1, N + 1)

# Initial condition for u
u_initial = 0.5 * (x + 1)

# Function for the boundary condition at x = 1
def g(t):
    return np.exp(-30 * t)+np.sin(2*np.pi*10*t)

# Function h(x, t) to transform boundary conditions
def h(x, t):
    return 0.5 * (x + 1) * g(t)

# Initial condition for v
v = u_initial - h(x, 0)

# Construct the Laplacian matrix for interior points
main_diag = -2 * np.ones(N - 1)
off_diag = np.ones(N - 2)
laplacian = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr') / dx**2

# Initialize the snapshot matrix for u
U = np.zeros((N + 1, steps + 1))
U[:, 0] = u_initial  # Store the initial condition

# Time-stepping loop
for n in range(steps):
    t = n * dt
    # Apply the Laplacian to the interior points
    v_interior = v[1:N]  # Exclude boundary points
    laplacian_term = laplacian.dot(v_interior)
    nonlinear_term = beta * (v_interior + h(x[1:N], t))**2
    # Modify the source term to incorporate boundary conditions
    laplacian_term[0] += v[0] / dx**2  # Left boundary influence
    laplacian_term[-1] += 0 / dx**2    # Right boundary (homogeneous in v)
    # Update interior points
    v_new_interior = v_interior + dt * (alpha * laplacian_term + nonlinear_term)
    # Update the solution, including boundary conditions
    v[1:N] = v_new_interior
    v[0] = 0  # v(-1, t) = 0
    v[N] = 0  # v(1, t) = 0
    # Reconstruct u from v and h
    u = v + h(x, t + dt)
    # Store the solution in the snapshot matrix
    U[:, n + 1] = u

# Create a list to store the filenames of the plots
filenames = []

# Generate and save plots for each time step
for n in range(0, steps + 1, steps // 100):  # Adjust the step to control the number of frames
    plt.figure()
    plt.plot(x, U[:, n], label=f'Time = {n * dt:.4f} s')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.ylim([0, 1])
    plt.title('Nonlinear Heat Equation Solution')
    plt.legend()
    plt.grid(True)
    # Save the plot as an image file
    filename = f'frame_{n:04d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Create an animated GIF
with imageio.get_writer('heat_equation_solution_transformed.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Remove the individual frame files
for filename in filenames:
    os.remove(filename)

print("Animated GIF saved as 'heat_equation_solution_transformed.gif'")
