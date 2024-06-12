import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Parameters
alpha = 1.0
beta = 0.1
delta = 0.9
c = 0.1
b = 0.05
d = 0.01

# State variables (discretized)
N_vals = np.linspace(0, 100, 101)
D_vals = np.linspace(1, 50, 50)
W_vals = np.linspace(0, 10, 11)

@njit(parallel=True)
def value_iteration(N_vals, D_vals, W_vals, alpha, beta, delta, c, b, d, max_iter=100):
    V = np.zeros((len(N_vals), len(D_vals), len(W_vals)))
    optimal_G = np.zeros((len(N_vals), len(D_vals)))
    optimal_R = np.zeros((len(N_vals), len(D_vals)))

    for iteration in range(max_iter):
        print(iteration)
        V_new = np.copy(V)
        for i in prange(len(N_vals)):
            for j in range(len(D_vals)):
                for k in range(len(W_vals)):
                    max_val = -np.inf
                    for G in np.linspace(0, 10, 101):  # Discretized G
                        for R in np.linspace(0, min(N_vals[i] / D_vals[j], 10), 101):  # Discretized R
                            if R <= N_vals[i] / D_vals[j]:
                                N_prime = N_vals[i] - c * R
                                D_prime = D_vals[j] + b * G
                                W_prime = W_vals[k] + d * G
                                # Find closest indices for next state
                                i_prime = np.argmin(np.abs(N_vals - N_prime))
                                j_prime = np.argmin(np.abs(D_vals - D_prime))
                                k_prime = np.argmin(np.abs(W_vals - W_prime))
                                # Bellman equation
                                utility = np.log(G) - beta * R ** 2
                                val = utility + delta * V[i_prime, j_prime, k_prime]
                                if val > max_val:
                                    max_val = val
                                    optimal_G[i, j] = G
                                    optimal_R[i, j] = R
                    V_new[i, j, k] = max_val
        V = V_new
    return V, optimal_G, optimal_R

# Run the value iteration
V, optimal_G, optimal_R = value_iteration(N_vals, D_vals, W_vals, alpha, beta, delta, c, b, d)
np.save(os.path.join(os.getcwd(),'V.npy'), V)
np.save(os.path.join(os.getcwd(),'optimal_G.npy'), optimal_G)
np.save(os.path.join(os.getcwd(),'optimal_R.npy'), optimal_R)

# Plot the value function
plt.figure(figsize=(10, 6))
plt.imshow(V[:, :, 0], origin='lower', aspect='auto', extent=[D_vals[0], D_vals[-1], N_vals[0], N_vals[-1]])
plt.colorbar(label='Value Function')
plt.xlabel('Population Density (D)')
plt.ylabel('Nutrient Availability (N)')
plt.title('Value Function at W=0')
plt.show()

# Plot the optimal growth rate G
plt.figure(figsize=(10, 6))
plt.imshow(optimal_G, origin='lower', aspect='auto', extent=[D_vals[0], D_vals[-1], N_vals[0], N_vals[-1]])
plt.colorbar(label='Optimal Growth Rate (G)')
plt.xlabel('Population Density (D)')
plt.ylabel('Nutrient Availability (N)')
plt.title('Optimal Growth Rate (G)')
plt.show()

# Plot the optimal resource allocation R
plt.figure(figsize=(10, 6))
plt.imshow(optimal_R, origin='lower', aspect='auto', extent=[D_vals[0], D_vals[-1], N_vals[0], N_vals[-1]])
plt.colorbar(label='Optimal Resource Allocation (R)')
plt.xlabel('Population Density (D)')
plt.ylabel('Nutrient Availability (N)')
plt.title('Optimal Resource Allocation (R)')
plt.show()
