import numpy as np
from numba import njit, prange

# Numbaを使ってC言語並みの速度にコンパイルし、マルチスレッドで並列化する関数
# クラスの外に定義するか、staticmethodにする必要があります。
@njit(parallel=True, fastmath=True)
def compute_forces_numba(q_2d, num_particles, box_size, epsilon, sigma):
    # 結果を格納する配列 (Nx2)
    forces = np.zeros((num_particles, 2), dtype=np.float64)
    
    # 事前に計算できる定数をまとめる
    box_half = box_size / 2.0
    sig6 = sigma**6
    sig12 = sigma**12
    coeff = 24.0 * epsilon
    
    # prangeにより、各粒子(i)の計算を複数CPUコアに自動的に割り振って並列化
    for i in prange(num_particles):
        fx = 0.0
        fy = 0.0
        
        # 粒子iに対する他の全ての粒子jからの力を計算
        for j in range(num_particles):
            if i == j:
                continue
                
            # 相対距離
            dx = q_2d[i, 0] - q_2d[j, 0]
            dy = q_2d[i, 1] - q_2d[j, 1]
            
            # 【最小移動像法】 (Numba内でnp.roundは遅いため、手動で処理)
            dx = dx - box_size * np.round(dx / box_size)
            dy = dy - box_size * np.round(dy / box_size)
            
            # 距離の2乗
            r2 = dx*dx + dy*dy
            
            # 力の計算 (ゼロ除算回避のため非常に近い場合は無視するなどの処理を入れても良い)
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            
            # F = 24 * epsilon * [2 * (sigma/r)^12 - (sigma/r)^6] * (1 / r^2)
            # すでに r で割る前のスカラーに dx, dy を掛ける形に最適化
            f_mag = coeff * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2
            
            fx += f_mag * dx
            fy += f_mag * dy
            
        forces[i, 0] = fx
        forces[i, 1] = fy
        
    return forces

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
    
class InteractingParticles2D(LangevinSystem):
    def __init__(self, num_particles, box_size, gamma, epsilon=1.0, sigma=1.0, mass=1.0, T=1.0):
        """
        2次元空間における相互作用（Lennard-Jones）を持つ多粒子ランジュバンシステム
        
        Parameters:
            num_particles (int): 粒子数 N
            box_size (float): 立方体（正方形）の箱の1辺の長さ
            gamma (float): 摩擦係数
            epsilon (float): LJポテンシャルのエネルギー深さ
            sigma (float): LJポテンシャルの粒子サイズパラメータ
            mass (float): 粒子の質量
            T (float): 設定温度
        """
        self.num_particles = num_particles
        self.box_size = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.mass = mass
        
        # 親クラスのコンストラクタに、自身で定義した微分関数を紐付けて初期化
        super().__init__(dHdq=self._dHdq, dHdp=self._dHdp, gamma=gamma, k_B=1.0, T=T)

    def _dHdp(self, q, p):
        """dH/dp = p / m (運動量から速度への変換)"""
        return p / self.mass

    def _dHdq(self, q, p):
        """dH/dq = -Force (位置微分 = 粒子に働く力のマイナス値)"""
        # 1次元配列(2*N,) を計算のために (N, 2) の2次元形状に変形
        q_2d = q.reshape((self.num_particles, 2))
        
        """
        # 粒子間の相対距離ベクトルを一括計算 (ブロードキャストを利用)
        # diff[i, j] は 粒子i から見た 粒子j の位置ベクトル
        diff = q_2d[:, None, :] - q_2d[None, :, :]
        
        # 【最小移動像法】周期境界を跨いだ最も近い鏡像との距離を計算
        diff = diff - self.box_size * np.round(diff / self.box_size)
        
        # 距離の2乗を計算 (形状: N x N)
        r2 = np.sum(diff**2, axis=-1)
        
        # 自分自身との距離(0)によるゼロ除算を防ぐため、対角成分を無限大にする
        np.fill_diagonal(r2, np.inf)
        
        # Lennard-Jones 力の計算
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        
        # 各ペアから受ける力の大きさ（スカラー部）
        # F = 24 * epsilon * [2 * (sigma/r)^12 - (sigma/r)^6] * (1 / r^2)
        force_mag = (24 * self.epsilon * (2 * (self.sigma**12) * inv_r12 - (self.sigma**6) * inv_r6) * inv_r2)
        
        # 方向（diff）を掛け合わせてベクトル化し、全粒子からの力を合計 (形状: N x 2)
        forces = np.sum(force_mag[:, :, None] * diff, axis=1)
        """

        forces = compute_forces_numba(q_2d, self.num_particles, self.box_size, self.epsilon, self.sigma) 
        
        # dH/dq = -Force。親クラスが処理できるよう、最後に1次元配列(2*N,)に平坦化して返す
        return -forces.flatten()

    def step(self, q, p, dt):
        """1ステップシミュレーションを進め、周期境界条件を適用する"""
        # 親クラスの更新処理を呼び出す (内部で自動的に1次元のまま計算される)
        q_new, p_new = super().step(q, p, dt)
        
        # 位置に周期境界条件を適用 (原点を中心とした [-box_size/2, box_size/2] の範囲に収める)
        q_2d = q_new.reshape((self.num_particles, 2))
        q_2d = ((q_2d + self.box_size / 2) % self.box_size) - self.box_size / 2
        
        return q_2d.flatten(), p_new