import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm, jarque_bera
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox

def check_stationarity(series):
    """进行ADF单位根检验"""
    result = adfuller(series, autolag='AIC')
    print('ADF统计量:', result[0])
    print('p值:', result[1])
    print('临界值:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] < 0.05:
        print("序列是平稳的 (拒绝原假设)")
        return True
    else:
        print("序列是非平稳的 (不能拒绝原假设)")
        return False

def estimate_ou_params(series, method='MLE'):
    """估计OU过程参数 (θ, μ, σ)"""
    Δt = 1  # 日度数据，时间间隔为1天
    n = len(series)
    
    if method == 'OLS':
        # OLS方法估计
        X = series.values[:-1]
        Y = series.values[1:]
        
        # 添加常数项
        X = np.column_stack([np.ones(len(X)), X])
        
        # 执行OLS回归
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        # 提取参数
        a, b = beta
        σ_η = np.std(Y - X @ beta)  # 残差标准差
        
        # 计算OU参数
        θ = (1 - b) / Δt
        μ = a / (θ * Δt)
        σ = σ_η / np.sqrt(Δt)
        
        return θ, μ, σ
    
    elif method == 'MLE':
        # 最大似然估计，使用向量化操作提高效率
        def log_likelihood(params):
            θ, μ, σ = params
            if θ <= 0 or σ <= 0:  # 确保参数为正
                return 1e10
                
            n = len(series)
            X = series.values
            dt = Δt
            
            # OU过程转移密度参数
            exp_term = np.exp(-θ * dt)
            mean_term = μ * (1 - exp_term)
            var_term = (σ**2) * (1 - np.exp(-2 * θ * dt)) / (2 * θ)
            
            # 避免方差为零或负
            if var_term <= 0:
                return 1e10
                
            # 使用向量化操作计算对数似然
            mean_vals = X[:-1] * exp_term + mean_term
            logL = np.sum(norm.logpdf(X[1:], loc=mean_vals, scale=np.sqrt(var_term)))
                
            return -logL  # 返回负对数似然用于最小化
        
        # 使用OLS结果作为初始值
        θ_ols, μ_ols, σ_ols = estimate_ou_params(series, method='OLS')
        
        # 优化
        result = minimize(log_likelihood, [θ_ols, μ_ols, σ_ols], 
                          method='L-BFGS-B',
                          bounds=[(1e-5, None), (None, None), (1e-5, None)])
        
        if result.success:
            θ_mle, μ_mle, σ_mle = result.x
            return θ_mle, μ_mle, σ_mle
        else:
            raise ValueError("MLE优化失败: " + result.message)

def diagnose_model(series, θ, μ, σ):
    """计算标准化残差并进行诊断检验"""
    Δt = 1
    X = series.values
    n = len(X)
    
    # 计算条件均值和方差
    exp_term = np.exp(-θ * Δt)
    mean_term = μ * (1 - exp_term)
    var_term = (σ**2) * (1 - np.exp(-2 * θ * Δt)) / (2 * θ)
    
    # 使用向量化操作计算标准化残差
    mean_vals = X[:-1] * exp_term + mean_term
    residuals = (X[1:] - mean_vals) / np.sqrt(var_term)
    
    # 残差正态性检验 (Jarque-Bera)
    jb_stat, jb_p = jarque_bera(residuals)
    print(f"\n残差正态性检验 (Jarque-Bera):")
    print(f"统计量: {jb_stat:.4f}, p值: {jb_p:.4f}")
    
    # 残差自相关检验 (Ljung-Box)
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    print("\n残差自相关检验 (Ljung-Box):")
    print(lb_test)
    
    return residuals