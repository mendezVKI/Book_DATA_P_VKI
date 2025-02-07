import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import os

# Parameters
alpha = 1.0  # Thermal diffusivity
beta = 1.0   # Nonlinear coefficient
L = 2.0      # Length of the domain [-1, 1]
n_x = 99     # Number of spatial points (to have a nice dx ! )
dx = L / n_x   # Spatial step size
dt = 0.0001  # Time step size
T = 2      # Total time
steps = int(T / dt)  # Number of time steps
times=np.arange(0,steps,1)*dt

# Spatial grid
x = np.linspace(-1, 1, n_x + 2)

# Initial condition
u = 0.5 * (x + 1)

def g(t):
    return np.exp(-30 * t)+np.sin(2*np.pi*10*t)

# Initialize the snapshot matrix
U = np.zeros((n_x + 2, steps + 1))
U[:, 0] = u  # Store the initial condition

# Time-stepping loop
for n in tqdm(range(steps),desc='Time Stepping'):
    t_n = n * dt; u_new = np.copy(u)
    # Update interior points
    for i in range(1, n_x+1):
        u_new[i] = u[i] + dt * (
            alpha * (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2 +
            beta * u[i]**2)
    # Apply boundary conditions
    u_new[0] = 0  # u(-1, t) = 0
    u_new[n_x+1] = g(t_n + dt)  # u(1, t) = g(t + dt)
    # Update solution
    u = u_new
    # Store the solution in the snapshot matrix
    U[:, n + 1] = u

# Plot the final solution
plt.plot(x, u, label='Numerical Solution at T={:.2f}'.format(T))
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title('Nonlinear Heat Equation Solution')
plt.legend()
plt.show()


# Make a gif of the solution

# Create a list to store the filenames of the plots
filenames = []

# Generate and save plots for each time step
for n in range(0, steps + 1, steps // 50):  # Adjust the step to control the number of frames
    plt.figure()
    plt.plot(x, U[:, n], label=f'Time = {n * dt:.4f} s')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Nonlinear Parabolic PDE')
    plt.legend(loc='upper left')
    plt.ylim([-1,1])
    plt.grid(True)
    # Save the plot as an image file
    filename = f'frame_{n:04d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()


# Create an animated GIF
with imageio.get_writer('heat_equation_solution_for_loop.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Remove the individual frame files
import os
for filename in filenames:
    os.remove(filename)

print("Animated GIF saved as 'heat_equation_solution_for_loop.gif'")



