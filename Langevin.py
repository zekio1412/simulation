import numpy as np
from numba import njit, prange

# Numbaを使ってC言語並みの速度にコンパイルし、マルチスレッドで並列化する関数
# クラスの外に定義するか、staticmethodにする必要があります。
@njit(parallel=True, fastmath=True)
def compute_forces_numba_2d(q_2d, num_particles, box_size, epsilon, sigma):
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
            if 1e-6 * sigma**2 < r2 <  2.5 * sigma**2:  # 距離が極端に小さい場合や大きい場合は無視
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

@njit(parallel=True, fastmath=True)
def compute_forces_numba_3d(q_3d, num_particles, box_size, epsilon, sigma):
    # 結果を格納する配列 (Nx3)
    forces = np.zeros((num_particles, 3), dtype=np.float64)
    
    # 事前に計算できる定数をまとめる
    box_half = box_size / 2.0
    sig6 = sigma**6
    sig12 = sigma**12
    coeff = 24.0 * epsilon
    
    # prangeにより、各粒子(i)の計算を複数CPUコアに自動的に割り振って並列化
    for i in prange(num_particles):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        
        # 粒子iに対する他の全ての粒子jからの力を計算
        for j in range(num_particles):
            if i == j:
                continue
                
            # 相対距離
            dx = q_3d[i, 0] - q_3d[j, 0]
            dy = q_3d[i, 1] - q_3d[j, 1]
            dz = q_3d[i, 2] - q_3d[j, 2]
            
            # 【最小移動像法】 (Numba内でnp.roundは遅いため、手動で処理)
            dx = dx - box_size * np.round(dx / box_size)
            dy = dy - box_size * np.round(dy / box_size)
            dz = dz - box_size * np.round(dz / box_size)
            
            # 距離の2乗
            r2 = dx*dx + dy*dy + dz*dz

            if 1e-6 * sigma**2 < r2 <  2.5 * sigma**2:  # 距離が極端に小さい場合や大きい場合は無視
                # 力の計算 (ゼロ除算回避のため非常に近い場合は無視するなどの処理を入れても良い)
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                inv_r12 = inv_r6 * inv_r6
                
                # F = 24 * epsilon * [2 * (sigma/r)^12 - (sigma/r)^6] * (1 / r^2)
                # すでに r で割る前のスカラーに dx, dy, dz を掛ける形に最適化
                f_mag = coeff * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2
                
                fx += f_mag * dx
                fy += f_mag * dy
                fz += f_mag * dz

        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz

    return forces

@njit(parallel=True, fastmath=True)
def compute_forces_coulomb_3d(q_3d, charges, num_particles, box_size, epsilon, sigma, qq_coeff=10.0):
    forces = np.zeros((num_particles, 3), dtype=np.float64)
    
    sig2 = sigma**2
    sig6 = sigma**6
    sig12 = sigma**12
    coeff_lj = 24.0 * epsilon
    
    # クーロン力の強さを決める係数 (1 / 4*pi*epsilon_0 に相当する無次元パラメータ)
    # 系の挙動を見ながら調整してください
    
    r_cut2 = (4.0 * sigma)**2  # 静電相互作用のためにカットオフを少し長めに設定
    r_min2 = 0.64 * sig2
    
    for i in prange(num_particles):
        fx, fy, fz = 0.0, 0.0, 0.0
        q_i = charges[i]  # 粒子iの電荷
        
        for j in range(num_particles):
            if i == j: continue
                
            dx = q_3d[i, 0] - q_3d[j, 0]
            dy = q_3d[i, 1] - q_3d[j, 1]
            dz = q_3d[i, 2] - q_3d[j, 2]
            
            dx -= box_size * np.round(dx / box_size)
            dy -= box_size * np.round(dy / box_size)
            dz -= box_size * np.round(dz / box_size)
            
            r2 = dx*dx + dy*dy + dz*dz
            
            if r2 < r_cut2:
                r2_eff = max(r2, r_min2)
                inv_r2 = 1.0 / r2_eff
                
                # 1. Lennard-Jones 力の計算 (近距離のみ)
                f_lj = 0.0
                if r2 < (2.5 * sigma)**2:
                    inv_r6 = inv_r2**3
                    inv_r12 = inv_r6**2
                    f_lj = coeff_lj * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2
                
                # 2. クーロン力の計算 (引力または反発力)
                # F_c = qq_coeff * (q_i * q_j) / r^3 * r_vector
                # 分母の r^3 に合わせるため、(1/r^2) に (1/r) をかける
                r_inv = np.sqrt(inv_r2)
                f_coulomb = qq_coeff * (q_i * charges[j]) * inv_r2 * r_inv
                
                # 合算
                f_total = f_lj + f_coulomb
                
                fx += f_total * dx
                fy += f_total * dy
                fz += f_total * dz
            
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        
    return forces

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_forces_diatomic_3d(q_3d, num_particles, box_size, epsilon, sigma, k_bond, r0_bond, charges):
    forces = np.zeros((num_particles, 3), dtype=np.float64)
    
    sig2 = sigma**2
    sig6 = sigma**6
    sig12 = sigma**12
    coeff_lj = 24.0 * epsilon
    
    # クーロン力の強さを決める係数 (1 / 4*pi*epsilon_0 に相当する無次元パラメータ)
    # 系の挙動を見ながら調整してください
    
    r_cut2 = (4.0 * sigma)**2  # 静電相互作用のためにカットオフを少し長めに設定
    r_min2 = 0.64 * sig2
    
    for i in prange(num_particles):
        fx, fy, fz = 0.0, 0.0, 0.0
        q_i = charges[i]   # 粒子iの電荷（chargesが与えられている場合）
        # 自分の結合相手を決定（0と1、2と3... がペアになる）
        if i % 2 == 0:
            bonded_j = i + 1
        else:
            bonded_j = i - 1
            
        for j in range(num_particles):
            if i == j: 
                continue
                
            dx = q_3d[i, 0] - q_3d[j, 0]
            dy = q_3d[i, 1] - q_3d[j, 1]
            dz = q_3d[i, 2] - q_3d[j, 2]
            
            # 最小移動像法（周期境界）
            dx -= box_size * np.round(dx / box_size)
            dy -= box_size * np.round(dy / box_size)
            dz -= box_size * np.round(dz / box_size)
            
            r2 = dx*dx + dy*dy + dz*dz
            
            # --- 相手が同じ分子のペアの場合（バネの力） ---
            if j == bonded_j:
                r = np.sqrt(r2)
                if r > 0.0001:  # ゼロ除算防止
                    # バネの力: f_mag = -k * (r - r0) / r
                    f_mag = -k_bond * (r - r0_bond) / r
                    fx += f_mag * dx
                    fy += f_mag * dy
                    fz += f_mag * dz
                    
            # --- それ以外の粒子間相互作用（Lennard-Jones 力, クーロン力） ---          
            if r2 < r_cut2:
                # クーロン力の計算 (引力または反発力)
                q_j = charges[j]   # 粒子jの電荷（chargesが与えられている場合）
                r2_eff = max(r2, r_min2)  # ゼロ除算防止のためのキャップ
                inv_r2 = 1.0 / r2_eff
                r_inv = np.sqrt(inv_r2)
                f_coulomb = 10.0 * (q_i * q_j) * inv_r2 * r_inv  # クーロン力の強さを決める係数は適宜調整してください

                # Lennard-Jones 力の計算 (近距離のみ)
                r2_eff = max(r2, r_min2)  # 爆発防止キャップ
                inv_r2 = 1.0 / r2_eff
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                
                # LJ力: f_mag = 24*eps * (2*(s/r)^12 - (s/r)^6) * (1/r^2)
                f_mag = coeff_lj * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2
                fx += f_mag * dx + f_coulomb * dx
                fy += f_mag * dy + f_coulomb * dy
                fz += f_mag * dz + f_coulomb * dz
                    
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        
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

        forces = compute_forces_numba_2d(q_2d, self.num_particles, self.box_size, self.epsilon, self.sigma) 
        
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
    
class LennardJonesParticles3D(LangevinSystem):
    def __init__(self, num_particles, box_size, gamma, epsilon=1.0, sigma=1.0, mass=1.0, T=1.0):
        self.num_particles = num_particles
        self.box_size = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.mass = mass
        
        super().__init__(dHdq=self._dHdq, dHdp=self._dHdp, gamma=gamma, k_B=1.0, T=T)

    def _dHdp(self, q, p):
        return p / self.mass

    def _dHdq(self, q, p):
        # 3次元に変形
        q_3d = q.reshape((self.num_particles, 3))
        
        # 3次元用のNumba関数を呼び出し
        forces = compute_forces_numba_3d(q_3d, self.num_particles, self.box_size, self.epsilon, self.sigma) 
        
        return -forces.flatten()

    def step(self, q, p, dt):
        q_new, p_new = super().step(q, p, dt)
        
        # 3次元に変形して周期境界条件を適用
        q_3d = q_new.reshape((self.num_particles, 3))
        q_3d = ((q_3d + self.box_size / 2) % self.box_size) - self.box_size / 2
        
        return q_3d.flatten(), p_new
    
class CoulombParticles3D(LangevinSystem):
    def __init__(self, num_particles, box_size, gamma, charges, epsilon=1.0, sigma=1.0, mass=1.0, T=1.0):
        self.num_particles = num_particles
        self.box_size = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.mass = mass
        self.charges = charges  # 粒子ごとの電荷を保持
        
        super().__init__(dHdq=self._dHdq, dHdp=self._dHdp, gamma=gamma, k_B=1.0, T=T)

    def _dHdp(self, q, p):
        return p / self.mass

    def _dHdq(self, q, p):
        # 3次元に変形
        q_3d = q.reshape((self.num_particles, 3))
        
        # クーロン力を含む3次元用のNumba関数を呼び出し
        forces = compute_forces_coulomb_3d(q_3d, self.charges, self.num_particles, self.box_size, self.epsilon, self.sigma) 
        
        return -forces.flatten()

    def step(self, q, p, dt):
        q_new, p_new = super().step(q, p, dt)
        
        # 3次元に変形して周期境界条件を適用
        q_3d = q_new.reshape((self.num_particles, 3))
        q_3d = ((q_3d + self.box_size / 2) % self.box_size) - self.box_size / 2
        
        return q_3d.flatten(), p_new
    
class DiatomicMolecules3D(LangevinSystem):
    def __init__(self, num_particles, box_size, gamma, charges, epsilon=1.0, sigma=1.0, 
                 mass=1.0, T=1.0, k_bond=500.0, r0_bond=1.0):
        
        # 二原子分子を作成するため、全体粒子数は偶数必須
        if num_particles % 2 != 0:
            raise ValueError("二原子分子モデルのため、num_particlesは偶数に設定してください。")
            
        self.num_particles = num_particles
        self.box_size = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.mass = mass
        self.k_bond = k_bond
        self.r0_bond = r0_bond
        self.charges = charges  # 電荷はオプション（必要に応じてクラスを拡張して対応）
        
        # 親クラスの初期化
        super().__init__(dHdq=self._dHdq, dHdp=self._dHdp, gamma=gamma, k_B=1.0, T=T)

    def _dHdp(self, q, p):
        return p / self.mass

    def _dHdq(self, q, p):
        # 3次元に変形
        q_3d = q.reshape((self.num_particles, 3))
        
        # 二原子分子用の力計算を呼び出し
        forces = compute_forces_diatomic_3d(
            q_3d, self.num_particles, self.box_size, 
            self.epsilon, self.sigma, self.k_bond, self.r0_bond, self.charges
        ) 
        
        return -forces.flatten()

    def step(self, q, p, dt):
        q_new, p_new = super().step(q, p, dt)
        
        # 3次元に変形して周期境界条件を適用
        q_3d = q_new.reshape((self.num_particles, 3))
        q_3d = ((q_3d + self.box_size / 2) % self.box_size) - self.box_size / 2
        
        return q_3d.flatten(), p_new