# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:05:55 2025

@author: mendez
"""

#%% Tutorials in interpolation vs regression

import numpy as np
import matplotlib.pyplot as plt
import os

# Customize plot settings for LaTeX and larger fonts
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 16,
    'font.family': 'serif'
})


# Generate the output folder
Fol_Plots = 'plots_tutorial_6'
if not os.path.exists(Fol_Plots):
    os.makedirs(Fol_Plots)


#%% Unknown function
def f(x):
    return np.exp(-3*x**2) * np.sin(2 * np.pi * x)+x

#%% Interpolation problem

# Define the Chebyshev polynomials using the recurrence relation at x
def chebyshev_polynomials(x, n_b):
    T = np.zeros((len(x), n_b ))
    T[:, 0] = 1  # T_0(x) = 1
    if n_b > 0:
        T[:, 1] = x  # T_1(x) = x
    for n in range(1, n_b-1):
        T[:, n + 1] = 2 * x * T[:, n] - T[:, n - 1]  # Recurrence relation
    return T

def chebyshev_roots(n):
    k = np.arange(1, n + 1)
    roots = np.cos((2 * k - 1) * np.pi / (2 * n))
    return roots

#%% Place collocation points 

# Degree of the Chebyshev polynomial
N = 10 # try 5, 10, 20
# Define the collocation points 
x_star=chebyshev_roots(N)
# Sample the function at the collocation points
f_star=f(x_star)
# define a grid only for plotting purposes
x_plot=np.linspace(-1,1,200)
f_plot=f(x_plot)


# show the function
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_plot, f_plot,label='$f(x)$')
ax.plot(x_star,f_star,'ro',label='$\mathbf{x}^*,\mathbf{f}^*$')
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$f(x)$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Function_and_Samples_Interp.pdf', dpi=200)


# Create the matrix for the interpolation:
B_interp = chebyshev_polynomials(x_star, N)
# Create the chebyshev matrix for plotting purposes
B_plot=chebyshev_polynomials(x_plot,N)

# Show 3 of these
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_plot, B_plot[:,3],'--',label='$T_{3}(x)$')
ax.plot(x_plot, B_plot[:,7],'-.',label='$T_{7}(x)$')
ax.plot(x_plot, B_plot[:,9],':',label='$T_{10}(x)$')
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$T_n(x)$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Chebyshev_bases.pdf', dpi=200)
    

# Find the coefficient c. Assembly the linear system
A=B_interp.T@B_interp; b=B_interp.T@f_star
# fast inversion of A if A is diagonal (check it!)
A_inv=np.diag(1/np.diag(A))
# compute c:
c=A_inv.dot(b)    
# The chebyshev interpolated solution at any arbitrary x is 
f_interp_plot=B_plot@c


# visual check :  
fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(B_interp.T@B_interp,aspect='equal',origin='upper')
ax.set_xlabel('$n$', fontsize=16)
ax.set_ylabel('$n$', fontsize=16)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
# ax.set_xlim([0,9])
# ax.set_ylim([0,9])
ax.set_title('$\mathbf{A}$',fontsize=16)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Chebyshev_orthogonality_Int.png', dpi=200)


# Interpolation results

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_plot, f_plot,label='$f(x)$')
ax.plot(x_plot,f_interp_plot,'r--',label='n=10 Appr')
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$f(x)$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Interpolation_Function.pdf', dpi=200)



#%% Regression problem

n_x=100 # number of samples
# Data for Regression
x_random=2*np.random.rand(n_x)-1
y_random=f(x_random)+0.2*np.random.normal(size=n_x)


fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_plot, f_plot,label='$f(x)$')
ax.plot(x_random,y_random,'ro',label='$\mathbf{x}^*,\mathbf{f}^*$')
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$f(x)$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Regression_Problem.pdf', dpi=200)


# Construct the Basis matrix on a given x domain
B_regression = chebyshev_polynomials(x_random, N)
# Find the coefficient c. Assembly the linear system
A=B_regression.T@B_regression; b=B_regression.T@y_random
# compute c by solving the linear system
c=np.linalg.solve(A,b)    
# The chebyshev interpolated solution at any arbitrary x is 
f_regression_plot=B_plot@c


fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_plot, f_plot,label='$f(x)$')
ax.plot(x_plot,f_regression_plot,'r--',label='Regression')
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$f(x)$', fontsize=18)
plt.legend()
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Regression_Function.pdf', dpi=200)


# visual check :  
fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(B_regression.T@B_regression,aspect='equal',origin='upper')
ax.set_xlabel('$n$', fontsize=16)
ax.set_ylabel('$n$', fontsize=16)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
# ax.set_xlim([0,9])
# ax.set_ylim([0,9])
ax.set_title('$\mathbf{A}$',fontsize=16)
fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
fig.savefig(Fol_Plots + os.sep + 'Chebyshev_orthogonality_Reg.png', dpi=200)


