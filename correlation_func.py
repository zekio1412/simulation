import numpy as np

def time_evolution_correlation(system, A, q0, p0, dt, num_steps):
    correlation = np.zeros(num_steps)
    q, p = q0, p0
    A_zero = A(q, p)  # Initial value of A
    for i_steps in range(num_steps):
        correlation[i_steps] = np.mean(A(q, p) * A_zero)  # Correlation with the initial value
        q, p = system.step(q, p, dt)
    return correlation