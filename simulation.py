import numpy as np
from scipy.stats import norm

def simulate_ou_paths(θ, μ, σ, X0, T, n_paths=1000):
    """模拟多条OU过程未来路径"""
    Δt = 1  # 日度数据
    paths = np.zeros((T, n_paths))
    paths[0] = X0
    
    # 使用精确迭代方法
    exp_term = np.exp(-θ * Δt)
    mean_term = μ * (1 - exp_term)
    std_term = np.sqrt((σ**2) * (1 - np.exp(-2 * θ * Δt)) / (2 * θ))
    
    # 使用向量化操作模拟路径
    noise = np.random.normal(0, 1, (T, n_paths))
    for t in range(1, T):
        paths[t] = μ + (paths[t-1] - μ) * exp_term + std_term * noise[t]
    
    return paths

def calculate_exceedance_probability(θ, μ, σ, X0, C, T, n_sim=10000):
    """计算超标概率：单点概率和期间至少一次超标概率"""
    # 方法1：解析解 (单点概率)
    exp_term = np.exp(-θ * T)
    mean_T = X0 * exp_term + μ * (1 - exp_term)
    var_T = (σ**2) * (1 - np.exp(-2 * θ * T)) / (2 * θ)
    prob_single_point = 1 - norm.cdf(C, loc=mean_T, scale=np.sqrt(var_T))
    
    # 方法2：蒙特卡洛模拟 (期间至少一次超标)
    paths = simulate_ou_paths(θ, μ, σ, X0, T, n_sim)
    exceedance_occurred = np.any(paths > C, axis=0)
    prob_at_least_once = np.mean(exceedance_occurred)
    
    # 使用向量化操作计算首次超标时间分布
    exceedance = paths > C
    first_exceedance = np.full(n_sim, np.nan)
    for i in range(n_sim):
        exceed_idx = np.where(exceedance[:, i])[0]
        if len(exceed_idx) > 0:
            first_exceedance[i] = exceed_idx[0] + 1  # +1因为从第1天开始
    
    return prob_single_point, prob_at_least_once, first_exceedance