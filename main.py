import numpy as np
from data_processing import load_and_preprocess_data
from ou_model import check_stationarity, estimate_ou_params, diagnose_model
from simulation import simulate_ou_paths, calculate_exceedance_probability
from visualization import plot_time_series, plot_residual_diagnosis, plot_simulated_paths, plot_first_exceedance_distribution, plot_exceedance_probability
from scipy.stats import norm
# 设置随机种子保证结果可复现
np.random.seed(40)

def main():
    # 参数设置
    FILE_PATH = 'data.xlsx'  # 替换为实际文件路径
    THRESHOLD = 75  # PM2.5超标阈值 (μg/m3)
    FORECAST_DAYS = 30  # 预测未来天数
    N_PATHS = 500  # 模拟路径数量
    N_SIM = 10000  # 蒙特卡洛模拟次数
    
    # 1. 加载和处理数据
    pm25 = load_and_preprocess_data(FILE_PATH)
    print(f"数据时间范围: {pm25.index.min()} 到 {pm25.index.max()}")
    print(f"样本数量: {len(pm25)}")
    
    # 绘制原始数据
    plot_time_series(pm25, THRESHOLD)
    
    # 2. 平稳性检验
    print("\n进行平稳性检验:")
    is_stationary = check_stationarity(pm25)
    
    if not is_stationary:
        # 如果不平稳，进行一阶差分
        print("\n数据不平稳，尝试一阶差分...")
        pm25_diff = pm25.diff().dropna()
        is_stationary_diff = check_stationarity(pm25_diff)
        
        if is_stationary_diff:
            print("差分后序列平稳，继续分析")
            pm25 = pm25_diff
        else:
            print("差分后仍不平稳，可能需要其他预处理。继续分析但结果可能不可靠。")
    
    # 3. 参数估计 (使用MLE方法)
    print("\n估计OU过程参数...")
    θ, μ, σ = estimate_ou_params(pm25, method='MLE')
    print(f"估计参数: θ = {θ:.4f}, μ = {μ:.2f}, σ = {σ:.4f}")
    print(f"长期均值: {μ:.2f} μg/m3")
    print(f"回归半衰期: {np.log(2)/θ:.2f} 天")
    
    # 4. 模型诊断
    residuals = diagnose_model(pm25, θ, μ, σ)
    plot_residual_diagnosis(residuals)
    
    # 5. 模拟未来路径
    X0 = pm25.iloc[-1]  # 使用最后一个观测值作为起点
    paths = simulate_ou_paths(θ, μ, σ, X0, FORECAST_DAYS, N_PATHS)
    plot_simulated_paths(paths, μ, THRESHOLD, FORECAST_DAYS, N_PATHS)
    
    # 6. 计算超标概率
    prob_point, prob_period, first_exceed = calculate_exceedance_probability(
        θ, μ, σ, X0, THRESHOLD, FORECAST_DAYS, N_SIM)
    
    print("\n超标概率结果:")
    print(f"在{FORECAST_DAYS}天后单日超标的概率: {prob_point:.4f} 或 {prob_point*100:.2f}%")
    print(f"未来{FORECAST_DAYS}天内至少有一天超标的概率: {prob_period:.4f} 或 {prob_period*100:.2f}%")
    
    plot_first_exceedance_distribution(first_exceed, FORECAST_DAYS)
    
    # 7. 计算不同预测期的单点超标概率
    days_range = np.arange(1, FORECAST_DAYS+1)
    point_probs = []
    
    for T in days_range:
        exp_term = np.exp(-θ * T)
        mean_T = X0 * exp_term + μ * (1 - exp_term)
        var_T = (σ**2) * (1 - np.exp(-2 * θ * T)) / (2 * θ)
        prob = 1 - norm.cdf(THRESHOLD, loc=mean_T, scale=np.sqrt(var_T))
        point_probs.append(prob)
    
    plot_exceedance_probability(days_range, point_probs, X0)

if __name__ == "__main__":
    main()