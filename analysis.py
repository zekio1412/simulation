import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _calculate_rdf_numba(q_array, box_size, num_particles, dr=0.05, r_max=None):
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

def calculate_rdf(q_history, box_size, num_particles, dr=0.05, r_max=None):
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
    r_centers, g_r = _calculate_rdf_numba(q_array, float(box_size), int(num_particles), float(dr), float(r_max))
    
    return r_centers, g_r

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
