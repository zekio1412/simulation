import numpy as np
import matplotlib.pyplot as plt
from Langevin import LangevinSystem, PeriodicLangevinSystem
from correlation_func import time_evolution_correlation


def current_A(q, p):
    return  p  # J = p

def dHdq(q, p):
    return np.sin(q)  # potential energy gradient
def dHdp(q, p):
    return p  # kinetic energy gradient

T = 300  # Temperature in Kelvin
k_B = 1.380649e-23  # Boltzmann constant in J/K
gamma = 0.5  # Friction coefficient
period = 2 * np.pi  # Period for periodic boundary conditions
dt = 0.01  # Time step in seconds

def sample_initial_conditions(num_samples):
    q_tempo = np.array([np.random.normal(0, np.sqrt(k_B * T)) for _ in range(num_samples)])
    p_tempo = np.array([np.random.normal(0, np.sqrt(k_B * T)) for _ in range(num_samples)])
    for _ in range(3000):  # Equilibration steps
        q_0, p_0 = system.step(q_tempo, p_tempo, dt)
    return q_0, p_0

system = PeriodicLangevinSystem(dHdq, dHdp, gamma=gamma, k_B=k_B, T=T, period=period)
q0, p0 = sample_initial_conditions(100)  # Sample 100 initial conditions
num_steps = 2000  # Number of steps to simulate
num_samples = 2000  # Number of samples for correlation

j_correlation = time_evolution_correlation(system, current_A, q0, p0, dt, num_steps)
integrated_value = np.trapezoid(j_correlation, dx=dt)
print(f"Integrated correlation (conductivity): {integrated_value:.4e}")

times = np.arange(0, num_steps*dt, dt)
plt.plot(times, j_correlation)
plt.title('Current Correlation Function')
plt.xlabel('Time (s)')
plt.ylabel('Correlation')
plt.show()