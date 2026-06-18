import numpy as np
import matplotlib.pyplot as plt
from Langevin import InteractingParticles2D
from tqdm import trange
from numba import njit, prange

@njit(parallel=True)
def calculate_vacf_numba(p_array, max_lag):
    steps, num_particles, _ = p_array.shape
    vacf_indiv = np.zeros(max_lag)
    vacf_total = np.zeros(max_lag)
    
    V_total = np.sum(p_array, axis=1) # (steps, 2)
    
    # 規格化用の分散（t=0）
    # 個別粒子の平均自乗速度
    var_indiv = 0.0
    for i in range(num_particles):
        var_indiv += np.mean(p_array[:, i, 0]**2 + p_array[:, i, 1]**2)
    var_indiv /= num_particles
    
    # 全速度和の平均自乗
    var_total = np.mean(V_total[:, 0]**2 + V_total[:, 1]**2)
    
    for tau in prange(max_lag):
        sum_i = 0.0
        sum_t = 0.0
        count = steps - tau
        
        for t in range(count):
            # 個別
            dot_i = (p_array[t+tau, :, 0] * p_array[t, :, 0] + 
                     p_array[t+tau, :, 1] * p_array[t, :, 1])
            sum_i += np.sum(dot_i)
            # 全体
            dot_t = (V_total[t+tau, 0] * V_total[t, 0] + 
                     V_total[t+tau, 1] * V_total[t, 1])
            sum_t += dot_t
            
        vacf_indiv[tau] = (sum_i / num_particles) / count
        vacf_total[tau] = sum_t / count
        
    return vacf_indiv / var_indiv, vacf_total / var_total

# --- 1. シミュレーションの設定 ---
num_particles = 256
box_size = 20
gamma = 0.5
dt = 0.005
equilibration_steps = 1000  # 空回し（熱平衡化）
production_steps = 10000     # 測定ステップ

# インスタンスの作成
sim = InteractingParticles2D(
    num_particles=num_particles, 
    box_size=box_size, 
    gamma=gamma, 
    epsilon=1.0, 
    sigma=0.2, 
    T=1.0
)

# --- 2. 初期状態の設定 ---
grid_dim = int(np.ceil(np.sqrt(num_particles)))
x = np.linspace(-box_size/3, box_size/3, grid_dim)
y = np.linspace(-box_size/3, box_size/3, grid_dim)
X, Y = np.meshgrid(x, y)
q = np.stack([X.flatten(), Y.flatten()], axis=1)[:num_particles].flatten()
p = np.random.normal(0, np.sqrt(sim.mass * sim.T), size=2 * num_particles)

print("Starting equilibration...")
for step in trange(equilibration_steps):
    q, p = sim.step(q, p, dt)
print("Equilibration complete.")

# --- 3. 基準時刻 (t=0) のデータの取得 ---
p_0 = p.copy().reshape((num_particles, 2))

# 全粒子の速度和 V_total(0)
V_total_0 = np.sum(p_0, axis=0)

vacf_indiv_list = []
vacf_total_list = []
time_list = []


# --- 修正版：相関関数の計算（時間平均をとる手法） ---

print("Starting production (saving data)...")
p_history = []
q_history = []
for step in trange(production_steps):
    p_current = p.reshape((num_particles, 2))
    p_history.append(p_current.copy())
    q_history.append(q.copy())
    q, p = sim.step(q, p, dt)

print("Calculating time-averaged correlations...")

# 配列化: shape = (時間ステップ数, 粒子数, 2)
p_array = np.array(p_history)
max_lag = int(production_steps * 0.5)  
norm_indiv, norm_total = calculate_vacf_numba(p_array, max_lag)

# --- 確認用 ---
print("VACF Indiv first 5:", norm_indiv[:5])
print("VACF Total first 5:", norm_total[:5])

if np.all(norm_indiv == 0):
    print("Warning: VACF is all zeros. Check p_array content.")


# 全粒子の速度和の履歴: shape = (時間ステップ数, 2)
V_total_history = np.sum(p_array, axis=1)



vacf_indiv_list = np.zeros(max_lag)
vacf_total_list = np.zeros(max_lag)





# 【重要】プロット用の時間軸をここで確実に max_lag の長さに合わせて作成する
time_array = np.arange(max_lag) * dt  # 長さ 500 の配列になる
theoretical_decay = np.exp(-gamma * time_array)

# --- グラフ化 (2つのサブプロット) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左のグラフ：個別粒子のVACF (ケージ効果)
ax1.plot(time_array, norm_indiv, label='Individual VACF', color='blue', linewidth=2)
ax1.plot(time_array, theoretical_decay, 'r--', label=r'Theoretical $e^{-\gamma t}$', linewidth=2)
ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
ax1.set_xlabel('Time $t$', fontsize=12)
ax1.set_ylabel('Normalized VACF', fontsize=12)
ax1.set_title('Individual Particle VACF (Cage Effect)', fontsize=14)
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 20)

# 右のグラフ：全速度和のVACF (ランジュバン減衰)
ax2.plot(time_array, norm_total, label= 'Total Velocity VACF', color='green', linewidth=2)
ax2.plot(time_array, theoretical_decay, 'r--', label= r'Theoretical $e^{-\gamma t}$', linewidth=2)
ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_title('Total Velocity VACF (Langevin Decay)', fontsize=14)
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 20)

plt.tight_layout()
plt.show()

from analysis import calculate_rdf

print("Calculating Radial Distribution Function g(r)...")
# q_history: 測定フェーズで保存した位置座標のリスト
# dr=0.02 程度にするとより解像度の高い綺麗な山が見られます
r, g_r = calculate_rdf(q_history, box_size, num_particles, dr=0.02)

# グラフ描画
plt.figure(figsize=(7, 5))
plt.plot(r, g_r, color='purple', linewidth=2, label='Simulation $g(r)$')
plt.axhline(1.0, color='gray', linestyle='--', label='Ideal Gas ($g(r)=1$)')
plt.xlabel('Distance $r$', fontsize=12)
plt.ylabel('Radial Distribution Function $g(r)$', fontsize=12)
plt.title('Radial Distribution Function', fontsize=14)
plt.xlim(0, box_size / 2.0)
plt.ylim(0, max(g_r) * 1.1)  # 第1ピークが綺麗に収まるように調整
plt.legend()
plt.grid(True)
plt.show()