import numpy as np

class LangevinSystem:
    def __init__(self, dHdq, dHdp, gamma, k_B, T):
        self.dHdq = dHdq
        self.dHdp = dHdp
        self.gamma = gamma
        self.k_B = k_B
        self.T = T

    def step(self, q, p, dt):
        # Langevin dynamics step using Euler-Maruyama method
        noise = np.random.normal(0, np.sqrt(2 * self.gamma * self.k_B * self.T * dt))
        p_new = p - self.dHdq(q, p) * dt - self.gamma * p * dt + noise
        q_new = q + self.dHdp(q, p) * dt
        return q_new, p_new

class PeriodicLangevinSystem(LangevinSystem):
    def __init__(self, dHdq, dHdp, gamma, k_B, T, period):
        super().__init__(dHdq, dHdp, gamma, k_B, T)
        self.period = period

    def step(self, q, p, dt):
        q_new, p_new = super().step(q, p, dt)
        # Apply periodic boundary conditions
        q_new = q_new % self.period
        return q_new, p_new