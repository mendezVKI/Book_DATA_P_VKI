# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:45:00 2025

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Parameters
alpha = 1.0  # Thermal diffusivity
beta = 1.0   # Nonlinear coefficient
N = 20       # Number of Chebyshev polynomials
dt = 0.0001  # Time step size
T = 2      # Total time
steps = int(T / dt)  # Number of time steps

# Chebyshev–Gauss–Lobatto points
x = np.cos(np.pi * np.arange(N + 1) / N)

# Initial condition
u = 0.5 * (x + 1)

# Function for the boundary condition at x = 1
def g(t):
    return np.exp(-30*t)

# Chebyshev differentiation matrix
def chebyshev_diff_matrix(N):
    if N == 0:
        return np.zeros((1, 1))
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2
    c = c * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D

D = chebyshev_diff_matrix(N)
D2 = np.dot(D, D)  # Second derivative matrix

# Initialize the snapshot matrix
U = np.zeros((N + 1, steps + 1))
U[:, 0] = u  # Store the initial condition

# Time-stepping loop
for n in range(steps):
    t = n * dt
    # Nonlinear term
    nonlinear_term = beta * u**2
    # Right-hand side
    rhs = u + dt * (alpha * np.dot(D2, u) + nonlinear_term)
    # Apply boundary conditions
    rhs[0] = 0
    rhs[-1] = g(t + dt)
    # Update the solution
    u = rhs
    # Store the solution
    U[:, n + 1] = u

# Create a list to store the filenames of the plots
filenames = []

# Generate and save plots for each time step
for n in range(0, steps + 1, steps // 100):  # Adjust the step to control the number of frames
    plt.figure()
    plt.plot(x, U[:, n], 'o-', label=f'Time = {n * dt:.4f} s')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.ylim([0,1])
    plt.title('Nonlinear Heat Equation Solution (Galerkin Method)')
    plt.legend()
    plt.grid(True)
    # Save the plot as an image file
    filename = f'frame_{n:04d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Create an animated GIF
with imageio.get_writer('heat_equation_solution_galerkin.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Remove the individual frame files
for filename in filenames:
    os.remove(filename)

print("Animated GIF saved as 'heat_equation_solution_galerkin.gif'")
