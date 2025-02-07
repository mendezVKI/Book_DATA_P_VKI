# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:47:50 2024

@author: mendez

We analyze the temporal correlation matrix of the cylinder test case in its
first transient.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Setting for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

# Generate the output folder
Fol_Plots = 'plots_exercise_5'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)

# Time discretization information
n_t = 3000; Fs = 3000; dt = 1/Fs
# As input signal we take P2.
data = np.load('Snapshot_Matrices.npz')
# Extract data arrays from the file
Xg = data['Xg']; Yg = data['Yg']
D_U = data['D_U'][:, :n_t]  # Truncate to n_t columns
D_V = data['D_V'][:, :n_t]
# Assemble both components into a single matrix
D_M = np.concatenate([D_U, D_V], axis=0)
# Remove the time average for plotting purposes
D_MEAN = np.mean(D_M, axis=1)  # Temporal average along columns
D = D_M - D_MEAN[:, np.newaxis]  # Subtract mean, preserving dimensions
# Clean up memory by deleting the temporary matrix
del D_M



#% Compute the temporal correlation matrix K
K=1/n_s*D.T@D

fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlim([0,0.05])
ax.set_ylim([0,0.05])
plt.pcolor(t,t,K) # We normalize the result
ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$\mathbf{t}[s]$',fontsize=14)
ax.set_ylabel('$\mathbf{t}[s]$',fontsize=14)
# ax.set_xticks(np.arange(0,800,200))
# ax.set_yticks(np.arange(0,800,200))
# ax.set_xlim(0,800)
# ax.set_ylim(0,800)
plt.title(r'${\mathbf{K}}(\mathbf{t}_k,\mathbf{t}_k)$')
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
plt.clim(0,5) # From here  downward is just for plotting purposes
plt.tight_layout()
plt.savefig(Fol_Plots + os.sep +'Ex_5_Correlation.png', dpi=300)  



# Prepare the frequency axis for the spectral plots
Freq=np.fft.fftshift(np.fft.fftfreq(int(n_t)))*Fs # Frequency Axis

## Compute the 2D FFT of K
K_HAT_ABS=np.fliplr(np.abs(np.fft.fftshift(np.fft.fft2(K-K.mean()))));
# Show a contour plot of the K_hat

fig, ax = plt.subplots(figsize=(4,4))
#ax.set_xlim([-0.5,0.5])
#ax.set_ylim([-0.5,0.5])
plt.pcolor(Freq,Freq,K_HAT_ABS/np.size(D)) # We normalize the result
ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$\mathbf{f_n}[Hz]$',fontsize=14)
ax.set_ylabel('$\mathbf{f_n}[Hz]$',fontsize=14)
ax.set_xticks(np.arange(0,800,200))
ax.set_yticks(np.arange(0,800,200))
ax.set_xlim(0,810)
ax.set_ylim(0,810)
plt.title(r'$\widehat{\mathbf{K}}(\mathbf{f}_n,\mathbf{f}_n)$')
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
plt.clim(0,0.01) # From here  downward is just for plotting purposes
plt.tight_layout()
plt.savefig(Fol_Plots + os.sep +'Ex5_Correlation_Spectra.png', dpi=300)  





# Extract the diagonal of K_F
K_HAT_ABS=np.fliplr(np.abs(np.fft.fftshift(np.fft.fft2(K-np.mean(K)))));
# For Plotting purposes remove the 0 freqs.
ZERO_F=np.where(Freq==0)[0]
ZERO_F=ZERO_F[0]; diag_K=np.abs(np.diag((K_HAT_ABS)));
diag_K[ZERO_F-1:ZERO_F+1]=0;


fig, ax = plt.subplots(figsize=(6,4))
ax.plot(Freq,diag_k_hat/np.max(diag_k_hat),linewidth=1.2)
plt.xlim(0,1500)
ax.set_xlabel('$\mathbf{f}_n [Hz]$', fontsize=18)
ax.set_ylabel('$|\widehat{}| $', fontsize=18)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep +'Ex_5_Diagonal_K_hat.pdf', dpi=300)











