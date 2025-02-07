# Simulate and sample an Ornstein-Uhlenbeck (OU) process

import numpy as np
import matplotlib.pyplot as plt
import os

# Configure plot aesthetics for LaTeX rendering and fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

# Create output folder for plots if it doesn't exist
Fol_Plots = 'plots_exercise_2'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

# Parameters for the Ornstein-Uhlenbeck process
kappa = 0.5  # Mean reversion rate (s^-1)
sigma = 1    # Volatility

# Time parameters
t_0 = 0  # Start time
t_end = 50  # End time
sampling_frequency = 50  # Sampling frequency (Hz)

# Derived time parameters
dt = 1 / sampling_frequency  # Time step size
t = np.arange(t_0, t_end, dt)  # Time points
n_t = len(t)  # Number of time points

# Function to simulate the Ornstein-Uhlenbeck process 
# as a Gaussian process (efficiently, i.e., with a recursive relation)
def sample_ou_as_gp_recursive(kappa, sigma, t):
    # Number of time steps
    n_t = len(t)
    # Time step size
    dt = t[1] - t[0]
    # Pre-compute constants for the OU process
    alpha = np.exp(-kappa * dt)  
    beta = sigma * np.sqrt((1 - alpha**2) / (2 * kappa))  
    # Initialize the process
    x = np.zeros(n_t)
    # Initial value from stationary distribution
    x[0] = np.random.normal(0, sigma / np.sqrt(2 * kappa))
    # Sequentially generate samples using the Markov property
    for i in range(1, n_t):
        x[i] = alpha * x[i-1] + beta * np.random.normal()
    return x

# Sample the OU process efficiently
x_1 = sample_ou_as_gp_recursive(kappa, sigma, t)
x_2 = sample_ou_as_gp_recursive(kappa, sigma, t)
x_3 = sample_ou_as_gp_recursive(kappa, sigma, t)

# Plot the sampled OU process (Gaussian process representation)
fig, ax = plt.subplots(figsize=(5, 3))
plt.plot(t, x_1, label="Realization 1")
plt.plot(t, x_2, label="Realization 2")
plt.plot(t, x_3, label="Realization 3")
plt.xlabel(r"$\mathbf{t}_k$ [s]", fontsize=14)
plt.ylabel(r"$\mathbf{x}[k]$", fontsize=14)
plt.title(r"Zero-mean OU Process with $\kappa=0.5, \sigma=1$")
# plt.legend();  plt.grid()
plt.show()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex2_Samples.pdf', dpi=300)


# Build snapshot matrix of the n_r realizations
n_r=500
# Initialize the snapshot matrix
U_N=np.zeros((n_t,n_r))
# Fill the snapshot matrix
for r in range(n_r):
  U_N[:,r]= sample_ou_as_gp_recursive(kappa, sigma, t)

# Compute the Ensamble autocorrelation
from DATA_P_VKI_functions import Ensemble_Cross_Corr
tau_lags=t
r_u_E=np.zeros(n_t)
for k in range(n_t):
  r_u_E[k]=Ensemble_Cross_Corr(U_N, U_N, 0, k)

# Theoretical autocorrelation functions:
r_u_t=np.exp(-kappa*t)


# For the time analysis, we build an extremely long sequence 
from DATA_P_VKI_functions import Cyclic_Cross_C, Linear_Cross_C

# Sample a very long sequence:
t_long = np.arange(t_0, t_end*100, dt)  # Time points
Signal_long=sample_ou_as_gp_recursive(kappa, sigma, t_long)    
# Compute the auto-correlation in time
r_u_L_long=Cyclic_Cross_C(Signal_long,Signal_long)
r_u_C_long=Linear_Cross_C(Signal_long,Signal_long)
# Half times:
t_half_long=t_long[0:len(r_u_L_long)]

fig, ax = plt.subplots(figsize=(5, 3))
plt.plot(t, r_u_t, 'k',label="Theory")
plt.plot(t, r_u_E, '--', label="Ensamble C")
plt.plot(t_half_long,r_u_C_long,':',label='Cyclic C')
plt.plot(t_half_long,r_u_L_long,'-.',label='Linear C')
plt.xlim([0,20])
plt.xlabel(r"$\tau$ [s]", fontsize=14)
plt.ylabel(r"$r_u(\tau)$", fontsize=14)
plt.title(r"Autocorrelation of the OH process")
plt.legend();  #plt.grid()
plt.show()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex_2_Auto_CORR_long.pdf', dpi=300)


#%% Repeat in case of shorter signal (using one of the realizations)
any_number=44
Signal_short=U_N[:,any_number]
r_u_L_short=Cyclic_Cross_C(Signal_short,Signal_short)
r_u_C_short=Linear_Cross_C(Signal_short,Signal_short)
# Half times:
t_half_short=t[0:len(r_u_L_short)]    

fig, ax = plt.subplots(figsize=(5, 3))
plt.plot(t, r_u_t, 'k',label="Theory")
plt.plot(t, r_u_E, '--', label="Ensamble C")
plt.plot(t_half_short,r_u_C_short,':',label='Cyclic C')
plt.plot(t_half_short,r_u_L_short,'-.',label='Linear C')
plt.xlim([0,20])
plt.xlabel(r"$\tau$ [s]", fontsize=14)
plt.ylabel(r"$r_u(\tau)$", fontsize=14)
plt.title(r"Autocorrelation of the OH process")
#plt.legend();  #plt.grid()
plt.show()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Ex_2_Auto_CORR_short.pdf', dpi=300)












