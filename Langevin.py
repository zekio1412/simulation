import numpy as np

class LangevinSystem:
    def __init__(self, dHdq, dHdp, gamma, k_B=1.0, T=1.0):
        """Langevin system with k_B*T energy scaling.
        In this scaling, k_B and T are set to 1 (dimensionless),
        and all energies are measured in units of k_B*T.
        """
        self.dHdq = dHdq
        self.dHdp = dHdp
        self.gamma = gamma
        self.k_B = k_B  # Dimensionless: k_B = 1
        self.T = T      # Dimensionless: T = 1

    def step(self, q, p, dt):
        # Velocity Verlet method - 2nd order accurate
        # Better energy conservation than Euler-Maruyama method
        
        # Half-step momentum update (deterministic part)
        p_half = p - 0.5 * self.dHdq(q, p) * dt
        
        # Full-step position update
        q_new = q + self.dHdp(q, p_half) * dt
        # Stochastic Langevin thermostat (Ornstein-Uhlenbeck process)
        # Exact integration: p_new = p*exp(-gamma*dt) + sqrt(1-exp(-2*gamma*dt)) * noise
        c_friction = np.exp(-self.gamma * dt)
        # Correct noise variance for equipartition: <p^2> = T = 1
        noise_variance = self.k_B * self.T * (1.0 - c_friction**2)  # Proper variance maintaining equipartition
        noise = np.random.normal(0, np.sqrt(noise_variance), size=len(p))
        p_half = p_half * c_friction + noise
        
        # Half-step momentum update (deterministic part)
        p_new = p_half - 0.5 * self.dHdq(q_new, p_half) * dt
        
        return q_new, p_new

class PeriodicLangevinSystem(LangevinSystem):
    def __init__(self, dHdq, dHdp, gamma, k_B, T, period):
        super().__init__(dHdq, dHdp, gamma, k_B, T)
        self.period = period

    def step(self, q, p, dt):
        q_new, p_new = super().step(q, p, dt)
        # Apply periodic boundary conditions with origin at center
        # Maps to [-period/2, period/2] instead of [0, period]
        q_new = ((q_new + self.period/2) % self.period) - self.period/2
        return q_new, p_new