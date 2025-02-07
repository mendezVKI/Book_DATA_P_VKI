# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:52:19 2024

@author: mendez

Script for Exercise 1: statistics of a random process
"""
# Study the statistics of an Ornstein-Uhlenbeck Process and analyze its properties.

import numpy as np
import matplotlib.pyplot as plt
import os

# Configure plot aesthetics for LaTeX rendering and fonts.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

# Create output folder for plots if it doesn't exist.
Fol_Plots = 'plots_exercise_1'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

# Define time parameters for the simulation.
t_0 = 0            # Initial time
t_end = 10         # Final time
n_t = 1001         # Number of time samples

t = np.linspace(t_0, t_end, n_t)  # Time scale
dt = t[2] - t[1]   # Time step size

# Parameters for the Ornstein-Uhlenbeck process.
kappa = 1.2        # Mean reversion speed
theta = 3          # Long-term mean
sigma = 0.5        # Volatility

# Function to generate an Ornstein-Uhlenbeck process.
def U_O_Process(kappa, theta, sigma, t):
    n_T = len(t)               # Number of time steps
    y = np.zeros(n_T)          # Initialize the process output

    # Define drift and diffusion functions for the process.
    drift = lambda y, t: kappa * (theta - y)
    diff = lambda y, t: sigma

    # Generate random noise for the process.
    noise = np.random.normal(loc=0, scale=1, size=n_T) * np.sqrt(dt)

    # Solve the stochastic difference equation.
    for i in range(1, n_T):
        y[i] = y[i-1] + drift(y[i-1], i * dt) * dt + diff(y[i-1], i * dt) * noise[i]

    return y

# Generate an ensemble of realizations of the process.
n_r = 5000                    # Number of realizations
U_N = np.zeros((n_t, n_r))     # Matrix to store all realizations

# Populate the matrix with realizations of the Ornstein-Uhlenbeck process.
for l in range(n_r):
    U_N[:, l] = U_O_Process(kappa, theta, sigma, t)

# Plot a few sample realizations.
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t, U_N[:, 1])
ax.plot(t, U_N[:, 10])
ax.plot(t, U_N[:, 22])
ax.plot(t, U_N[:, 55])
ax.set_xlabel('$t[s]$', fontsize=18)
ax.set_ylabel('$\mathbf{u}(\mathbf{t}_k)$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_samples.pdf', dpi=300)

# Compute the sample mean and standard deviation across realizations.
U_Mean = np.mean(U_N, axis=1)  # Ensemble mean
U_STD = np.std(U_N, axis=1)    # Ensemble standard deviation

# Plot the mean and standard deviation.
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t, U_Mean)
ax.plot(t, U_Mean + U_STD, 'r--')
ax.plot(t, U_Mean - U_STD, 'r--')
ax.set_xlabel('$t[s]$', fontsize=18)
ax.set_ylabel(r'$\tilde{\mu}_u\pm\tilde{\sigma}_u$', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_Means.pdf', dpi=300)

# Compute ensemble autocorrelations.
def Ensemble_Cross_Corr(U_N, W_N, k, j):
    n_r, n_t = np.shape(U_N)   # Dimensions of the ensemble matrix
    # Extract realizations at time indices k and j.
    U_N_k = U_N[k, :]
    W_N_j = W_N[j, :]
    # Mean center the values.
    U_N_k_p = U_N_k - U_N_k.mean()
    W_N_j_p = W_N_j - W_N_j.mean()
    # Compute the cross-correlation coefficient.
    R_UW = np.sum(U_N_k_p * W_N_j_p) / (len(U_N_k) * np.std(U_N_k) * np.std(W_N_j))
    return R_UW

# Compute full autocorrelation matrix (not used directly here).
C_UW_matrix = 1 / n_r * U_N @ U_N.T

# Analyze autocorrelations at specific lags.
lag = 50
N_S = 100                      # Number of random pairs
R_UW = np.zeros(N_S)           # Array to store autocorrelations

# Select random pairs of time indices and compute autocorrelations.
J = np.random.randint(500, 800, N_S)
K = J + lag
for n in range(N_S):
    R_UW[n] = Ensemble_Cross_Corr(U_N, U_N, J[n], K[n])

# Analyze convergence of ensemble mean and standard deviation as a function of n_r.
n_R = np.round(np.logspace(0.1, 3, num=41))  # Range of ensemble sizes

# Initialize arrays for statistics at two specific time indices.
mu_10 = np.zeros(len(n_R))
sigma_10 = np.zeros(len(n_R))
mu_700 = np.zeros(len(n_R))
sigma_700 = np.zeros(len(n_R))

# Loop over ensemble sizes and compute statistics.
for n in range(len(n_R)):
    print(f'Computing n={n} of {len(n_R)}')
    n_r = int(n_R[n])
    U_N = np.zeros((n_t, n_r))

    # Generate ensemble of size n_r.
    for l in range(n_r):
        U_N[:, l] = U_O_Process(kappa, theta, sigma, t)

    # Compute mean and standard deviation at specific time indices.
    mu_10[n] = np.mean(U_N[10, :])
    sigma_10[n] = np.std(U_N[10, :])
    mu_700[n] = np.mean(U_N[700, :])
    sigma_700[n] = np.std(U_N[700, :])

# Plot convergence of mean.
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(n_R, mu_10, 'ko:', label='k=10')
ax.plot(n_R, mu_700, 'rs:', label='k=700')
ax.set_xscale('log')
ax.set_xlabel('$n_r$', fontsize=18)
ax.set_ylabel('$\mu_U$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_Mean_Conv.pdf', dpi=300)

# Plot convergence of standard deviation.
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(n_R, sigma_10, 'ko:', label='k=10')
ax.plot(n_R, sigma_700, 'rs:', label='k=700')
ax.set_xscale('log')
ax.set_xlabel('$n_r$', fontsize=18)
ax.set_ylabel('$\sigma_U$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex1_Sigma_Conv.pdf', dpi=300)
