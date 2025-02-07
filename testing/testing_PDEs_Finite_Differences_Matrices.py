import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import imageio.v2
from tqdm import tqdm
import os


# Parameters
alpha = 1.0  # Thermal diffusivity
beta = 1.0   # Nonlinear coefficient
L = 2.0      # Length of the domain [-1, 1]
n_x = 99     # Number of spatial points (to have a nice dx ! )
dx = L / (n_x + 1)  # Spatial step size; adjusted for boundary points
dt = 0.0001  # Time step size
T = 2      # Total time
steps = int(T / dt)  # Number of time steps
times=np.arange(0,steps,1)*dt

# Spatial grid
x = np.linspace(-1, 1, n_x + 2)

# Initial condition
u = 0.5 * (x + 1)

# Generate the output folder
Fol_Plots = 'plots_exercise_7'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)


def g(t):
    return np.exp(-30 * t)+np.sin(2*np.pi*10*t)


# Construct the Laplacian matrix for interior points
main_diag = -2 * np.ones(n_x)
off_diag = np.ones(n_x - 1)
laplacian = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr') / dx**2

# Initialize the snapshot matrix
U = np.zeros((n_x + 2, steps ))
U[:, 0] = u  # Store the initial condition

# Time-stepping loop
for n in tqdm(range(steps-1),desc='Time Stepping'):
    t_n = n * dt
    # Apply the Laplacian to the interior points
    u_interior = u[1:n_x+1] # Exclude boundary points (+1 needed for python)
    laplacian_term = laplacian.dot(u_interior)
    nonlinear_term = beta * u_interior**2
    # Modify the source term to incorporate boundary conditions
    laplacian_term[0] += u[0] / dx**2  # Left boundary influence (is zero here)
    laplacian_term[-1] += g(t_n) / dx**2  # Right boundary influence
    # Update interior points
    u_new_interior = u_interior + dt * (alpha * laplacian_term + nonlinear_term)
    # Update the solution, including boundary conditions
    u[1:n_x+1] = u_new_interior
    u[0] = 0  # u(-1, t) = 0; not strictly needed
    u[-1] = g(t_n + dt)  # u(1, t) = g(t_n + dt)
    # Store the solution in the snapshot matrix
    U[:, n + 1] = u

# Create a list to store the filenames of the plots
filenames = []

# Generate and save plots for each time step
for n in range(0, steps , steps // 100):  # Adjust the step to control the number of frames
    plt.figure()
    plt.plot(x, U[:, n], label=f'Time = {n * dt:.4f} s')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.ylim([-1,1])
    plt.title('Nonlinear Parabolic PDE')
    plt.legend(loc='upper left')
    plt.grid(True)
    # Save the plot as an image file
    filename = f'frame_{n:04d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Create an animated GIF
with imageio.get_writer('heat_equation_solution_matrix.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Remove the individual frame files
import os
for filename in filenames:
    os.remove(filename)

print("Animated GIF saved as 'heat_equation_solution_matrix.gif'")

PLOT=[0,10,50,100,200,500,1000,5000,10000,15000,20000]
# Make some plots of the solution
fig, ax = plt.subplots(figsize=(6, 3))
for k in range(len(PLOT)):
 ax.plot(x, U[:, PLOT[k]], label=f'$t={times[PLOT[k]]:.3f}$')
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$u(x)$', fontsize=18)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'some_solutions.pdf', dpi=200)



fig, ax = plt.subplots(figsize=(4, 4))
plt.contourf(x,times,U.T)
plt.ylim([0,0.5])
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$t$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'space_time.png', dpi=200)














