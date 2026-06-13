import numpy as np
def maxwell_boltzmann_distribution(v, T, m):
    """
    Calculate the Maxwell-Boltzmann distribution for a given velocity, temperature, and mass.

    Parameters:
    v (float): Velocity of the particles (in m/s)
    T (float): Temperature (in Kelvin)
    m (float): Mass of the particles (in kg)

    Returns:
    float: The value of the Maxwell-Boltzmann distribution at the given velocity
    """
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    prefactor = (m / (2 * np.pi * k_B * T)) ** (3/2)
    exponent = -m * v**2 / (2 * k_B * T)
    
    return prefactor * np.exp(exponent)
# Example usage:
import matplotlib.pyplot as plt
velocities = np.linspace(0, 2000, 1000)  # Velocity range from 0 to 2000 m/s
temperature = 1000  # Temperature in Kelvin
mass = 4.65e-26  # Mass of an oxygen molecule (O2)
distribution = maxwell_boltzmann_distribution(velocities, temperature, mass)
plt.plot(velocities, distribution)
plt.title('Maxwell-Boltzmann Distribution')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Probability Density')
plt.show()