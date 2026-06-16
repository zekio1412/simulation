import numpy as np
import matplotlib.pyplot as plt
from Langevin import InteractingParticles2D

# --- 1. シミュレーションの設定 ---
num_particles = 256
box_size = 100.0
gamma = 0.5
dt = 0.01
equilibration_steps = 1000  # 空回し（熱平衡化）
production_steps = 1000     # 測定ステップ

# インスタンスの作成
sim = InteractingParticles2D(
    num_particles=num_particles, 
    box_size=box_size, 
    gamma=gamma, 
    epsilon=1.0, 
    sigma=1.0, 
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
for step in range(equilibration_steps):
    q, p = sim.step(q, p, dt)
    if step == 50:
        print("Equilibration 5% complete...")
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
for step in range(production_steps):
    p_current = p.reshape((num_particles, 2))
    p_history.append(p_current.copy())
    q, p = sim.step(q, p, dt)

# 配列化: shape = (時間ステップ数, 粒子数, 2)
p_array = np.array(p_history)

# 全粒子の速度和の履歴: shape = (時間ステップ数, 2)
V_total_history = np.sum(p_array, axis=1)

max_lag = int(production_steps * 0.5)  # 1000ステップの半分で 500

vacf_indiv_list = np.zeros(max_lag)
vacf_total_list = np.zeros(max_lag)

print("Calculating time-averaged correlations...")
for tau in range(max_lag):
    if tau == 0:
        dot_indiv = np.sum(p_array * p_array, axis=2)
        vacf_indiv_list[tau] = np.mean(dot_indiv)
        
        dot_total = np.sum(V_total_history * V_total_history, axis=1)
        vacf_total_list[tau] = np.mean(dot_total)
    else:
        # スライスによる時間平均
        dot_indiv = np.sum(p_array[tau:] * p_array[:-tau], axis=2)
        vacf_indiv_list[tau] = np.mean(dot_indiv)
        
        dot_total = np.sum(V_total_history[tau:] * V_total_history[:-tau], axis=1)
        vacf_total_list[tau] = np.mean(dot_total)

# 規格化 (t=0 の値で割る)
norm_indiv = vacf_indiv_list / vacf_indiv_list[0]
norm_total = vacf_total_list / vacf_total_list[0]

# 【重要】プロット用の時間軸をここで確実に max_lag の長さに合わせて作成する
time_array = np.arange(max_lag) * dt  # 長さ 500 の配列になる
theoretical_decay = np.exp(-gamma * time_array)

# --- グラフ化 (2つのサブプロット) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左のグラフ：個別粒子のVACF (ケージ効果)
ax1.plot(time_array, norm_indiv, label='Individual VACF', color='blue', linewidth=2)
ax1.plot(time_array, theoretical_decay, 'r--', label='Theoretical $e^{-\gamma t}$', linewidth=2)
ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
ax1.set_xlabel('Time $t$', fontsize=12)
ax1.set_ylabel('Normalized VACF', fontsize=12)
ax1.set_title('Individual Particle VACF (Cage Effect)', fontsize=14)
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, time_array[-1])

# 右のグラフ：全速度和のVACF (ランジュバン減衰)
ax2.plot(time_array, norm_total, label='Total Velocity VACF', color='green', linewidth=2)
ax2.plot(time_array, theoretical_decay, 'r--', label='Theoretical $e^{-\gamma t}$', linewidth=2)
ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_title('Total Velocity VACF (Langevin Decay)', fontsize=14)
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, time_array[-1])

plt.tight_layout()
plt.show()