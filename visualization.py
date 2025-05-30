import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib

# 设置 Matplotlib 使用中文字体（Seaborn 也会继承 Matplotlib 的字体设置）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_time_series(pm25, threshold):
    """绘制原始时间序列"""
    plt.figure(figsize=(12, 6))
    pm25.plot(title='PM2.5时间序列')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold} μg/m3)')
    plt.ylabel('PM2.5浓度 (μg/m3)')
    plt.legend()
    plt.show()

def plot_residual_diagnosis(residuals):
    """绘制残差诊断图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 残差直方图与正态分布比较
    sns.histplot(residuals, kde=True, ax=axes[0], stat='density')
    x = np.linspace(-4, 4, 100)
    axes[0].plot(x, norm.pdf(x), 'r-', lw=2)
    axes[0].set_title('标准化残差分布')
    
    # 残差自相关图
    plot_acf(residuals, ax=axes[1])
    axes[1].set_title('残差自相关')
    plt.tight_layout()
    plt.show()

def plot_simulated_paths(paths, μ, threshold, forecast_days, n_paths):
    """绘制模拟路径"""
    plt.figure(figsize=(12, 6))
    plt.plot(paths, alpha=0.1, color='b', linewidth=0.5)
    plt.plot(np.mean(paths, axis=1), 'r-', lw=2, label='平均路径')
    plt.axhline(y=μ, color='g', linestyle='-', label=f'长期均值 ({μ:.1f})')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold})')
    plt.title(f'{n_paths}条PM2.5未来{forecast_days}天模拟路径', fontsize=14, pad=15)
    plt.ylabel('PM2.5浓度 (μg/m3)', fontsize=12)
    plt.xlabel('预测天数', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_first_exceedance_distribution(first_exceed, forecast_days):
    """绘制首次超标时间分布"""
    plt.figure(figsize=(10, 6))
    valid_times = first_exceed[~np.isnan(first_exceed)]
    if len(valid_times) > 0:
        plt.hist(valid_times, bins=30, alpha=0.7)
        plt.axvline(x=np.mean(valid_times), color='r', linestyle='--', 
                   label=f'平均时间: {np.mean(valid_times):.1f}天')
        plt.title('首次超标时间分布')
        plt.xlabel('天数')
        plt.ylabel('频数')
        plt.legend()
        plt.show()
    else:
        print(f"在{forecast_days}天内没有发生超标事件的模拟路径")

def plot_exceedance_probability(days_range, point_probs, X0):
    """绘制不同预测期的单点超标概率"""
    plt.figure(figsize=(10, 6))
    plt.plot(days_range, point_probs, 'bo-')
    plt.title(f'不同预测期的单日超标概率 (当前浓度: {X0:.1f} μg/m3)')
    plt.xlabel('预测天数')
    plt.ylabel('超标概率')
    plt.grid(True)
    plt.show()