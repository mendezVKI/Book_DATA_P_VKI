# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 18:35:55 2025

@author: mendez
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm



# Customize plot settings for LaTeX and larger fonts
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 16,
    'font.family': 'serif'
})


# Generate the output folder
Fol_Plots = 'plots_exercise_7'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)


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
x = np.linspace(-1, 1, n_x + 2)  # Including boundary points

# Basis matrix function
def Phi_sines(x, n_b):
    x = np.asarray(x)
    N = len(x)  # Total number of grid points
    Phi = np.zeros((N, n_b))
    # normalization accounting the number of intervals
    normalization_factor = np.sqrt(2 / (N-1))
    for n in range(1, n_b + 1):
        Phi[:, n - 1] = np.sin(n * np.pi * (x + 1) / 2) * normalization_factor
    return Phi

# Second derivatives:
def Phi_sines_xx(x, n_b):
    x = np.asarray(x)
    N = len(x)  # Total number of grid points
    Phi_xx = np.zeros((N, n_b))
    # normalization accounting the number of intervals
    normalization_factor = np.sqrt(2 / (N-1))
    for n in range(1, n_b + 1):
        Phi_xx[:, n - 1] = -(n * np.pi/2)**2* np.sin(n * np.pi * (x + 1) / 2) * normalization_factor
    return Phi_xx    
    

# Construct the basis matrix
n_b = 10  # Number of basis functions
Phi_sin = Phi_sines(x, n_b) # Matrix of bases
Phi_sin_xx = Phi_sines_xx(x, n_b) # Matrix of second derivatives of the bases


D_xx=Phi_sin.T@Phi_sin_xx;
D_xx[np.abs(D_xx)<1e-5]=0


####check:
#num=5    
#Phi_xx_num=np.gradient(np.gradient(Phi_sin[:,num],x),x)
#Phi_xx_ana=Phi_sin_xx[:,num]
#plt.plot(x,Phi_xx_num,'ko:'); plt.plot(x,Phi_xx_ana)


# Check orthonormality
Check = Phi_sin.T @ Phi_sin
print("Orthonormality check (should be close to identity):")
print(Check)


# Initial condition for u
u_initial = 0.5 * (x + 1)

# Have a look at the problems in approximating this initial condition:
# s_initial=Phi_sin.T@u_initial
# u_initial_approx=Phi_sin@s_initial
# plt.plot(x,u_initial);plt.plot(x,u_initial_approx,'r--')

# Function h(x, t) satisfying boundary conditions
def h_fun(x, t):
    return 0.5 * (x + 1) * (np.exp(-30 * t)+np.sin(2*np.pi*10*t))


def h_dot_fun(x,t):
    return 0.5*(x+1)*(-30 * np.exp(-30 * t) + 20 * np.pi * np.cos(2 * np.pi * 10 * t))




# Construct the Laplacian matrix for interior points
main_diag = -2 * np.ones(n_x - 1)
off_diag = np.ones(n_x - 2)
laplacian = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr') / dx**2


# Initial condition for v on S
s = Phi_sin.T@(u_initial - h_fun(x, 0))

# Initialize the snapshot matrix for u
S = np.zeros((n_b, steps + 1))
# The initial condition would be zero in this case in the V problem.
S[:, 0] = s  # Store the initial condition

# prepare the tensor C
C = np.einsum('ij,ik->ijk', Phi_sin, Phi_sin)  # Efficiently computes C[i, j, k] = Phi[i, j] * Phi[i, k]

# Time-stepping loop
for n in tqdm(range(steps),desc='Time Stepping'):
    t_n = n * dt
    # Current reduced state
    s=S[:,n]
    # compute the h function at this and next time
    h_n = h_fun(x, t_n); h_n_p_1=h_fun(x, t_n+dt)
    # compute g_dot
    h_dot=h_dot_fun(x,t_n)
    # Term 1 
    T_1=alpha*D_xx.dot(s)     
    # Term 2 
    T_2=beta*Phi_sin.T@np.einsum('ijk,j,k->i', C, s, s)
    # Term 3
    T_3_1=beta*Phi_sin.T.dot(h_n)
    T_3_2=beta*np.linalg.multi_dot([Phi_sin.T,np.diag(h_n),Phi_sin])@s
    T_3_3=-Phi_sin.T@h_dot
    T_3=T_3_1+T_3_2+T_3_3
    # update the state vector (Euler method)
    S[:, n + 1] = s+dt*(T_1+T_2+T_3)



# Reconstruct the full solution:
U_ROM = np.zeros((n_x + 2, steps + 1))
# Reconstruct the full solution
V=Phi_sin@S    
# Add back the transformation of the BCs
for n in range(steps):
  t_n=n*dt  
  U_ROM[:,n]=V[:,n]+h_fun(x,t_n)   


# Create a list to store the filenames of the plots
filenames = []

# Generate and save plots for each time step
for n in range(0, steps, steps // 100):  # Adjust the step to control the number of frames
    plt.figure()
    plt.plot(x, U_ROM[:, n], label=f'Time = {n * dt:.4f} s')
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
with imageio.get_writer('heat_equation_solution_matrix_transformed_ROM.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Remove the individual frame files
for filename in filenames:
    os.remove(filename)

print("Animated GIF saved as 'heat_equation_solution_matrix_transformed_ROM'")



# Load the Finite Difference results and comprae some snapshots 

# Load the .npz file
data = np.load('U_matrix.npz')  # Use 'U_matrix_compressed.npz' if you used np.savez_compressed
# Access the matrix U
U_FD = data['U_FD']




PLOT=[50,500,5000,15000]
# Make some plots of the solution
fig, ax = plt.subplots(figsize=(6, 3))
for k in range(len(PLOT)):
 if k<len(PLOT)-1:   
  ax.plot(x, U_FD[:, PLOT[k]], 'k')
  ax.plot(x, U_ROM[:, PLOT[k]],'b--')
 else:   
  ax.plot(x, U_FD[:, PLOT[k]], 'k',label='FD')
  ax.plot(x, U_ROM[:, PLOT[k]],'b--',label='ROM')


ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$u(x)$', fontsize=18)
plt.legend(loc='upper left')
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Solution_ROM.pdf', dpi=200)














