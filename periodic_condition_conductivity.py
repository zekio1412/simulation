import numpy as np
import matplotlib.pyplot as plt
from Langevin import LangevinSystem, PeriodicLangevinSystem
from correlation_func import time_evolution_correlation


def current_A(q, p):
    return  p  # J = 

def dHdq(q, p):
    return np.sin(q)  # potential energy gradient
def dHdp(q, p):
    return p  # kinetic energy gradient

# Energy and temperature in k_B*T scaling (dimensionless)
# k_B*T is the fundamental energy unit
k_B = 1.0  # Boltzmann constant (dimensionless in k_B*T scaling)
T = 1.0    # Temperature (dimensionless in k_B*T scaling)

# Physical parameters
gamma = 0.5  # Friction coefficient (scaled appropriately)
period = 2 * np.pi  # Period for periodic boundary conditions
dt = 0.01  # Time step (in appropriate units)

def sample_initial_conditions(num_samples):
    # In k_B*T scaling: standard deviation = sqrt(k_B*T) = 1
    q_tempo = np.array([np.random.normal(0, 1.0) for _ in range(num_samples)])
    p_tempo = np.array([np.random.normal(0, 1.0) for _ in range(num_samples)])
    for _ in range(3000):  # Equilibration steps
        q_tempo, p_tempo = system.step(q_tempo, p_tempo, dt)
        if _ % 500 == 0:
            print(f"Equilibration step: {_}/{3000}")
    print(f"Equilibration completed for {num_samples} samples.")
    return q_tempo, p_tempo

num_steps = 5000  # Number of steps to simulate
num_samples = 30000  # Number of samples for correlation
system = PeriodicLangevinSystem(dHdq, dHdp, gamma=gamma, k_B=k_B, T=T, period=period)
q0, p0 = sample_initial_conditions(num_samples)  


j_correlation = time_evolution_correlation(system, current_A, q0, p0, dt, num_steps)

# Debug: Check initial correlation value and statistics
C0 = j_correlation[0]
p0_sq = np.mean(p0**2)
print(f"C(0) = {C0:.6f}")
print(f"<p0^2> = {p0_sq:.6f}  (theoretical: 1.0)")
print(f"C(0)/C_max ratio: {C0/np.max(np.abs(j_correlation)):.3f}")

# Apply time-dependent damping correction if needed

integrated_value = np.trapezoid(j_correlation, dx=dt)
print(f"Integrated correlation (conductivity): {integrated_value:.4e}")

times = np.arange(0, num_steps*dt, dt)
plt.figure(figsize=(10, 6))
plt.plot(times, j_correlation, linewidth=2)
plt.title('Current Correlation Function')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.show()