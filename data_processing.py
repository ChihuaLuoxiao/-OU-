import numpy as np
import pandas as pd

def load_and_preprocess_data(file_path):
    """加载数据，处理缺失值，并提取PM2.5序列"""
    # 假设数据文件为CSV格式
    df = pd.read_excel(file_path)
    
    # 确保日期是datetime格式并设为索引
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)
    
    # 提取PM2.5序列
    pm25 = df['PM2.5'].copy()
    
    # 处理缺失值 - 使用线性插值
    pm25 = pm25.interpolate(method='linear')
        
    return pm25