import numpy as np
import matplotlib.pyplot as plt
from Langevin import DiatomicMolecules3D
from tqdm import trange
from numba import njit, prange
from analysis import calculate_rdf_3d, calculate_vacf_numba_3d, animate_3d_particles, animate_particles_colored



import numpy as np
from tqdm import trange

# --- 1. シミュレーションの設定 ---
num_particles = 1000
num_molecules = num_particles // 2  # 分子の数は粒子の半分 (500個)
box_size = 15
gamma = 0.5
dt = 0.001
equilibration_steps = 1000  
production_steps = 10000     

# 分子の設定パラメータ
r0_bond = 1.0  # バネの自然長（この距離でペアを初期配置します）
T_target = 0.5

# --- 電荷の設定（極性分子モデルの作成） ---
# ランダムにシャッフルするのではなく、分子内のペアで電荷を分けるのが自然です。
# 例：ペアの片方(偶数インデックス)を +1.0、もう片方(奇数インデックス)を -1.0 とする
charges = np.zeros(num_particles, dtype=np.float64)
charges[0::2] = 1.0   # 粒子0, 2, 4... は正電荷
charges[1::2] = -1.0  # 粒子1, 3, 5... は負電荷

# インスタンスの作成 
# ※DiatomicMolecules3Dクラスが charges を受け取り、
# クーロン力も計算できるように先ほどのコードをマージしていることを前提とします。
sim = DiatomicMolecules3D(
    num_particles=num_particles, 
    box_size=box_size, 
    gamma=gamma, 
    charges=charges,
    epsilon=1.0, 
    sigma=1.0, 
    T=T_target,
    k_bond=500.0,
    r0_bond=r0_bond
)

# --- 2. 初期状態の設定 (二原子分子用) ---
print("Setting up initial positions for diatomic molecules...")

# 1. まず「分子の重心」を配置するためのグリッドを作成
grid_dim = int(np.ceil(num_molecules ** (1/3)))
spacing = box_size / grid_dim  # グリッドの間隔

# 壁に近すぎないように配置空間を少し絞る
x = np.linspace(-box_size/2 + spacing/2, box_size/2 - spacing/2, grid_dim)
y = np.linspace(-box_size/2 + spacing/2, box_size/2 - spacing/2, grid_dim)
z = np.linspace(-box_size/2 + spacing/2, box_size/2 - spacing/2, grid_dim)

X, Y, Z = np.meshgrid(x, y, z)
# num_molecules 分の重心座標を取得
molecule_centers = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)[:num_molecules]

# 2. 重心の位置を基準に、ペアとなる2つの粒子を配置
q_initial = np.zeros((num_particles, 3))

for i in range(num_molecules):
    center = molecule_centers[i]
    
    # 粒子1 (偶数インデックス): 中心からx軸マイナス方向へ自然長の半分ずらす
    q_initial[2*i]     = center + np.array([-r0_bond / 2.0, 0.0, 0.0])
    
    # 粒子2 (奇数インデックス): 中心からx軸プラス方向へ自然長の半分ずらす
    q_initial[2*i + 1] = center + np.array([r0_bond / 2.0, 0.0, 0.0])

# 1次元配列に平坦化
q = q_initial.flatten()

# 3. 運動量の初期化
p = np.random.normal(0, np.sqrt(sim.mass * sim.T), size=3 * num_particles)



print("Starting equilibration...")
for step in trange(equilibration_steps):
    q, p = sim.step(q, p, dt)
print("Equilibration complete.")

p_history = []
q_history = []



# --- 3. 基準時刻 (t=0) のデータの取得 ---
p_0 = p.copy().reshape((num_particles, 3))


# --- 修正版：相関関数の計算（時間平均をとる手法） ---

print("Starting production (saving data)...")
for step in trange(production_steps):
    p_current = p.reshape((num_particles, 3))
    p_history.append(p_current.copy())
    q_history.append(q.copy())
    q, p = sim.step(q, p, dt)


"""
print("Calculating time-averaged correlations...")

# 配列化: shape = (時間ステップ数, 粒子数, 3)
p_array = np.array(p_history)
max_lag = int(production_steps * 0.5)  
norm_indiv, norm_total = calculate_vacf_numba_3d(p_array, max_lag)




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
"""


print("Calculating Radial Distribution Function g(r)...")
# q_history: 測定フェーズで保存した位置座標のリスト
# dr=0.02 程度にするとより解像度の高い綺麗な山が見られます
r, g_r = calculate_rdf_3d(q_history, box_size, num_particles, dr=0.02)

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


from scipy.special import j0  # 第1種0次ベッセル関数

def calculate_sq(r, g_r, rho, q_range=(0.1, 15), num_q=100):
    """
    g(r) をハンケル変換して S(q) を計算する
    q_range: 変換後の波数 q の範囲
    """
    q_values = np.linspace(q_range[0], q_range[1], num_q)
    s_q = np.zeros_like(q_values)
    
    # dr を計算（rが等間隔であることを想定）
    dr = r[1] - r[0]
    
    for i, q in enumerate(q_values):
        # 積分計算: integral = integral r * (g(r)-1) * J0(qr) dr
        integrand = r * (g_r - 1.0) * j0(q * r)
        integral = np.sum(integrand) * dr
        
        # S(q) = 1 + 2 * pi * rho * integral
        s_q[i] = 1.0 + 2.0 * np.pi * rho * integral
        
    return q_values, s_q

# --- 実行 ---
# 密度 rho を計算（calculate_rdf内での定義と同じ値）
rho = num_particles / (box_size ** 3)

# S(q) を計算
q, s_q = calculate_sq(r, g_r, rho)

# プロット
plt.figure(figsize=(7, 5))
plt.plot(q, s_q, color='teal', linewidth=2, label='$S(q)$')
plt.axhline(1.0, color='gray', linestyle='--')
plt.xlabel('Wavenumber $q$', fontsize=12)
plt.ylabel('Static Structure Factor $S(q)$', fontsize=12)
plt.title('Static Structure Factor from RDF', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

ani = animate_particles_colored(q_history[::50], sim.charges, num_particles, box_size, dt, interval=30)