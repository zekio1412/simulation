from Langevin import InteractingParticles2D
import numpy as np
import matplotlib.pyplot as plt

# --- シミュレーションの設定 ---
num_particles = 512    # 粒子数
box_size = 50.0       # 箱の大きさ（-5.0 〜 5.0）
gamma = 0.5           # 摩擦（ダンピング）
dt = 0.01             # タイムステップ（急激な反発があるため小さめに設定）
steps = 1000           # シミュレーションのステップ数

# インスタンスの作成
sim = InteractingParticles2D(
    num_particles=num_particles, 
    box_size=box_size, 
    gamma=gamma, 
    epsilon=1.0, 
    sigma=1.0, 
    T=1.0
)

# --- 初期状態の設定 ---
# 重なりを防ぐため、ランダムではなく綺麗な格子状に初期配置してみます
grid_dim = int(np.ceil(np.sqrt(num_particles)))
x = np.linspace(-box_size/3, box_size/3, grid_dim)
y = np.linspace(-box_size/3, box_size/3, grid_dim)
X, Y = np.meshgrid(x, y)
q_init = np.stack([X.flatten(), Y.flatten()], axis=1)[:num_particles]

# 1次元配列(2*N,)に平坦化
q = q_init.flatten()
# 初期運動量はマクスウェル・ボルツマン分布（標準正規分布）からサンプリング
p = np.random.normal(0, np.sqrt(sim.mass * sim.T), size=2 * num_particles)

# --- シミュレーションループ ---
steps_list = []
p_list = []
ver_p_list = []
trajectory = []

for step in range(steps):
    q, p = sim.step(q, p, dt)
    
    # 解析や描画のために、(N, 2)の形に戻して保存
    trajectory.append(q.reshape((num_particles, 2)).copy())
    steps_list.append(step)
    p_list.append(p)
    ver_p_list.append(np.var(p))

    if step % 100 == 0:
        print(f"Step {step}/{steps} completed.")

trajectory = np.array(trajectory)
plt.figure(figsize=(10, 6))
plt.plot(steps_list, ver_p_list, linewidth=2)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Variance of Momentum', fontsize=14)
plt.title('Thermalization of Interacting Particles', fontsize=16)
plt.grid()
plt.show()

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
# %matplotlib notebook  # 必要に応じて

def animate_trajectory(trajectory, box_size):
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = box_size / 2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    
    scat = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], c='blue', edgecolors='k', s=100)
    
    def update(frame):
        scat.set_offsets(trajectory[frame])
        ax.set_title(f"Step: {frame}")
        return scat,

    # ここで作ったアニメーションオブジェクトを
    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=30, blit=False)
    
    plt.show()
    return ani  # 関数の外へ引き渡す

# === 【超重要】ここを修正します ===
# ただ関数を呼び出すだけでなく、グローバル変数「my_anim」に代入してメモリ上に固定する！
my_anim = animate_trajectory(trajectory, box_size=10.0)

# シミュレーション後半（平衡状態と予想される期間）のデータを抽出
final_p = np.array(p_list[int(steps*0.8):]).flatten() # 後半20%を使用

plt.hist(final_p, bins=30, density=True, alpha=0.6, label='Simulation')
x = np.linspace(-4, 4, 100)
plt.plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2), 'r--', label='Theoretical (Normal Dist)')
plt.legend()
plt.title("Momentum Distribution at Equilibrium")
plt.show()