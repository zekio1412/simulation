import numpy as np
import matplotlib.pyplot as plt

class ResponseFunction:
    def __init__(self, num_steps, dt, force, correlation):
        self.force = force
        self.correlation = correlation
        self.k_B = 1.0  # Boltzmann constant (dimensionless in k_B*T scaling)
        self.T = 1.0    # Temperature (dimensionless in k_B*T scaling)
        self.num_steps = num_steps
        self.dt = dt
        

    def get_response(self):
        response = np.convolve(self.force, self.correlation, mode='full')[:self.num_steps] * self.dt / (self.k_B * self.T)
        return response
    
    def plot_response(self):
        response = self.get_response()
        times = np.arange(0, self.num_steps * self.dt, self.dt)
        plt.figure(figsize=(10, 6))
        plt.plot(times, response, linewidth=2)
        plt.title('Response Function')
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.grid()
        plt.show()

