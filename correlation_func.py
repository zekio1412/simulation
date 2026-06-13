import numpy as np

def time_evolution_correlation(system, A, q0, p0, dt, num_steps):
    """
    Time-evolution correlation function: C(t) = <[A(t) - <A>_eq] * [A(0) - <A>_eq]>
    where <A>_eq is the thermal equilibrium average.
    
    Args:
        system: Dynamical system
        A: Observable function A(q, p) - works with arrays
        q0, p0: Initial conditions (num_samples,) shaped arrays (already thermalized)
        dt: Time step
        num_steps: Number of steps to integrate
    
    Returns:
        correlation: C(t) for each time step
    """
    num_samples = len(q0)
    correlation = np.zeros(num_steps)
    
    # Initial values (already at thermal equilibrium)
    A_zero = A(q0, p0)  # Shape (num_samples,)
    A_eq = np.mean(A_zero)  # Equilibrium average
    
    # Center initial values
    A_zero_centered = A_zero - A_eq
    
    # Evolve all samples together
    q, p = q0.copy(), p0.copy()
    
    for i_steps in range(num_steps):
        A_t = A(q, p)
        # Correlation: <[A(t) - <A>_eq] * [A(0) - <A>_eq]>
        correlation[i_steps] = np.mean((A_t - A_eq) * A_zero_centered)
        q, p = system.step(q, p, dt)
    
    return correlation