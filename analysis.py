import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _calculate_rdf_numba_2d(q_array, box_size, num_particles, dr=0.05, r_max=None):
    """Numbaでコンパイル・並列化される計算のコア部分"""
    num_steps = q_array.shape[0]
    num_bins = int(np.ceil(r_max / dr))
    
    # 複数スレッドが同時に書き込んでも競合しないよう、ステップ（フレーム）ごとに独立したヒストグラムを用意
    hist_per_step = np.zeros((num_steps, num_bins), dtype=np.float64)
    
    # 各フレームをCPUコアに分散して並列処理
    for step in prange(num_steps):
        q_current = q_array[step]
        
        # 粒子iとjのペアを二重ループで計算
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                dx = q_current[i, 0] - q_current[j, 0]
                dy = q_current[i, 1] - q_current[j, 1]
                
                # 最小移動像法の適用
                dx = dx - box_size * np.round(dx / box_size)
                dy = dy - box_size * np.round(dy / box_size)
                
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < r_max:
                    bin_idx = int(dist / dr)
                    if bin_idx < num_bins:
                        # i->j, j->i の双方向をカウントするため 2.0 を足す
                        hist_per_step[step, bin_idx] += 2.0
                        
    # 分散して計算したヒストグラムを全て合算
    rdf_hist = np.zeros(num_bins, dtype=np.float64)
    for step in range(num_steps):
        for b in range(num_bins):
            rdf_hist[b] += hist_per_step[step, b]
            
    # --- 理想気体（均一分布）をベースにした規格化 ---
    rho = num_particles / (box_size * box_size)
    g_r = np.zeros(num_bins, dtype=np.float64)
    r_centers = np.zeros(num_bins, dtype=np.float64)
    
    for i in range(num_bins):
        r_in = i * dr
        r_out = r_in + dr
        r_centers[i] = r_in + dr / 2.0
        
        dV = np.pi * (r_out*r_out - r_in*r_in)
        ideal_count = rho * dV
        
        # 実際の1粒子あたりの平均カウント
        actual_count_per_particle = rdf_hist[i] / (num_steps * num_particles)
        
        if ideal_count > 0:
            g_r[i] = actual_count_per_particle / ideal_count
            
    return r_centers, g_r

@njit(parallel=True, fastmath=True)
def _calculate_rdf_numba_3d(q_array, box_size, num_particles, dr=0.05, r_max=None):
    """Numbaでコンパイル・並列化される計算のコア部分 (3次元版)"""
    num_steps = q_array.shape[0]
    num_bins = int(np.ceil(r_max / dr))
    
    # 複数スレッドが同時に書き込んでも競合しないよう、ステップ（フレーム）ごとに独立したヒストグラムを用意
    hist_per_step = np.zeros((num_steps, num_bins), dtype=np.float64)
    
    # 各フレームをCPUコアに分散して並列処理
    for step in prange(num_steps):
        q_current = q_array[step]
        
        # 粒子iとjのペアを二重ループで計算
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                dx = q_current[i, 0] - q_current[j, 0]
                dy = q_current[i, 1] - q_current[j, 1]
                dz = q_current[i, 2] - q_current[j, 2]  # z成分を追加
                
                # 最小移動像法の適用
                dx = dx - box_size * np.round(dx / box_size)
                dy = dy - box_size * np.round(dy / box_size)
                dz = dz - box_size * np.round(dz / box_size)  # z成分を追加
                
                # 3次元の距離計算
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if dist < r_max:
                    bin_idx = int(dist / dr)
                    if bin_idx < num_bins:
                        # i->j, j->i の双方向をカウントするため 2.0 を足す
                        hist_per_step[step, bin_idx] += 2.0
                        
    # 分散して計算したヒストグラムを全て合算
    rdf_hist = np.zeros(num_bins, dtype=np.float64)
    for step in range(num_steps):
        for b in range(num_bins):
            rdf_hist[b] += hist_per_step[step, b]
            
    # --- 理想気体（均一分布）をベースにした規格化 (3次元対応) ---
    # バルク密度 rho = N / V (3次元なので体積 L^3)
    rho = num_particles / (box_size * box_size * box_size)
    g_r = np.zeros(num_bins, dtype=np.float64)
    r_centers = np.zeros(num_bins, dtype=np.float64)
    
    for i in range(num_bins):
        r_in = i * dr
        r_out = r_in + dr
        r_centers[i] = r_in + dr / 2.0
        
        # 3次元の微小球殻（シェル）の体積: dV = (4/3) * pi * (r_out^3 - r_in^3)
        dV = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)
        ideal_count = rho * dV
        
        # 実際の1粒子あたりの平均カウント
        actual_count_per_particle = rdf_hist[i] / (num_steps * num_particles)
        
        if ideal_count > 0:
            g_r[i] = actual_count_per_particle / ideal_count
            
    return r_centers, g_r

def calculate_rdf_2d(q_history, box_size, num_particles, dr=0.05, r_max=None):
    """
    ユーザーが呼び出すラッパー関数。
    入力データをNumbaが処理しやすい3次元のNumPy配列に整形してからコア関数に渡す。
    """
    if r_max is None:
        r_max = box_size / 2.0
        
    # 入力がリストの場合はNumPy配列に変換
    if isinstance(q_history, list):
        q_array = np.array(q_history, dtype=np.float64)
    else:
        q_array = q_history
        
    # 形状が (steps, 2*N) に平坦化されている場合は (steps, N, 2) に変形
    if q_array.ndim == 2:
        q_array = q_array.reshape((q_array.shape[0], num_particles, 2))
        
    # 高速化されたNumba関数を呼び出す
    r_centers, g_r = _calculate_rdf_numba_2d(q_array, float(box_size), int(num_particles), float(dr), float(r_max))
    
    return r_centers, g_r

def calculate_rdf_3d(q_history, box_size, num_particles, dr=0.05, r_max=None):
    """
    ユーザーが呼び出すラッパー関数。
    入力データをNumbaが処理しやすい3次元のNumPy配列に整形してからコア関数に渡す。
    """
    if r_max is None:
        r_max = box_size / 2.0
        
    # 入力がリストの場合はNumPy配列に変換
    if isinstance(q_history, list):
        q_array = np.array(q_history, dtype=np.float64)
    else:
        q_array = q_history
        
    # 形状が (steps, 3*N) に平坦化されている場合は (steps, N, 3) に変形
    if q_array.ndim == 2:
        q_array = q_array.reshape((q_array.shape[0], num_particles, 3))
        
    # 高速化されたNumba関数を呼び出す
    r_centers, g_r = _calculate_rdf_numba_3d(q_array, float(box_size), int(num_particles), float(dr), float(r_max))
    
    return r_centers, g_r

@njit(parallel=True)
def calculate_vacf_numba_2d(p_array, max_lag):
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


@njit(parallel=True, fastmath=True)
def calculate_vacf_numba_3d(p_array, max_lag):
    # p_array の shape は (steps, num_particles, 3) を想定
    steps, num_particles, _ = p_array.shape
    vacf_indiv = np.zeros(max_lag)
    vacf_total = np.zeros(max_lag)
    
    # 時間ごとの全速度和 (steps, 3)
    V_total = np.sum(p_array, axis=1) 
    
    # 規格化用の分散（t=0）
    # 個別粒子の平均自乗速度 (z成分を追加)
    var_indiv = 0.0
    for i in range(num_particles):
        var_indiv += np.mean(p_array[:, i, 0]**2 + p_array[:, i, 1]**2 + p_array[:, i, 2]**2)
    var_indiv /= num_particles
    
    # 全速度和の平均自乗 (z成分を追加)
    var_total = np.mean(V_total[:, 0]**2 + V_total[:, 1]**2 + V_total[:, 2]**2)
    
    for tau in prange(max_lag):
        sum_i = 0.0
        sum_t = 0.0
        count = steps - tau
        
        for t in range(count):
            # 個別 (x, y, z成分の内積)
            dot_i = (p_array[t+tau, :, 0] * p_array[t, :, 0] + 
                     p_array[t+tau, :, 1] * p_array[t, :, 1] + 
                     p_array[t+tau, :, 2] * p_array[t, :, 2])
            sum_i += np.sum(dot_i)
            
            # 全体 (x, y, z成分の内積)
            dot_t = (V_total[t+tau, 0] * V_total[t, 0] + 
                     V_total[t+tau, 1] * V_total[t, 1] + 
                     V_total[t+tau, 2] * V_total[t, 2])
            sum_t += dot_t
            
        vacf_indiv[tau] = (sum_i / num_particles) / count
        vacf_total[tau] = sum_t / count
        
    return vacf_indiv / var_indiv, vacf_total / var_total


import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def animate_3d_particles(q_history, num_particles, box_size, dt, interval=50):
    """
    q_history: (steps, num_particles*3) のリストまたは配列
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 箱のサイズ設定
    limit = box_size / 2.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_title("3D Particle Simulation")

    # 最初のフレームの散布図を作成
    q_0 = q_history[0].reshape((num_particles, 3))
    scat = ax.scatter(q_0[:, 0], q_0[:, 1], q_0[:, 2], s=10, c='blue', alpha=0.6)


    def update(frame):
        t = frame * dt * 50  # 現在の時間を計算
        # 現在のフレームの座標を取得
        q_frame = q_history[frame].reshape((num_particles, 3))
        # 散布図のデータを更新
        scat._offsets3d = (q_frame[:, 0], q_frame[:, 1], q_frame[:, 2])
        ax.set_title(f"3D Particle Simulation at t={t:.2f}")
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(q_history), interval=interval, blit=False
    )
    
    plt.show()
    return ani

def animate_particles_colored(q_history, charges, num_particles, box_size, dt, interval=50):
    """
    charges: 各粒子の電荷が入った配列 (+1.0 または -1.0)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    limit = box_size / 2.0
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    
    # 最初の位置データ
    q_0 = q_history[0].reshape((num_particles, 3))
    
    # 【ここがポイント】
    # c=charges とし、cmap（カラーマップ）に 'bwr' (Blue-White-Red) を指定します。
    # これにより、-1 が青、+1 が赤にマッピングされます。
    scat = ax.scatter(q_0[:, 0], q_0[:, 1], q_0[:, 2], 
                      s=15, c=charges, cmap='bwr', vmin=-1.0, vmax=1.0, alpha=0.7)

    def update(frame):
        t = frame * dt * 50  # 間引き率（ここでは5）に合わせる
        
        q_frame = q_history[frame].reshape((num_particles, 3))
        scat._offsets3d = (q_frame[:, 0], q_frame[:, 1], q_frame[:, 2])
        
        ax.set_title(f"Time $t = {t:.2f}$ (Red: +, Blue: -)")
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(q_history), interval=interval, blit=False
    )
    
    plt.show()
    return ani

