import efinance as ef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import os
from matplotlib.font_manager import FontProperties
import time

# 设置中文字体
try:
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SimHei.ttf')
    if os.path.exists(font_path):
        custom_font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = custom_font.get_name()
    else:
        print("警告：未找到SimHei.ttf字体文件，将使用默认字体")
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字体设置错误: {str(e)}")
    plt.rcParams['font.family'] = 'sans-serif'

def get_stock_data(stock_code='GOOGL', start_date='20180101', end_date='20241130'):
    """获取股票数据"""
    file_path = f'{stock_code}_history.csv'
    
    if os.path.exists(file_path):
        print(f"从本地加载{stock_code}的历史数据...")
        df = pd.read_csv(file_path)
    else:
        print(f"从网络获取{stock_code}的历史数据...")
        df = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date)
        df.to_csv(file_path, encoding='utf-8', index=False)
    
    df['日期'] = pd.to_datetime(df['日期'])
    cols_to_keep = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
    df = df[cols_to_keep]
    
    return df

def create_features(df):
    """创建特征"""
    df = df.copy()
    
    # 确保数值类型
    numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 基础滞后特征
    price_cols = ['收盘', '开盘', '最高', '最低']
    for col in price_cols:
        for i in range(1, 11):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # 移动平均线
    for window in [5, 10, 20, 50, 100]:
        df[f'MA_{window}'] = df['收盘'].rolling(window=window).mean()
        df[f'volume_MA_{window}'] = df['成交量'].rolling(window=window).mean()
    
    # 波动率指标
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['收盘'].rolling(window=window).std()
    
    # 价格动量指标
    for period in [5, 10, 20, 50]:
        df[f'momentum_{period}'] = df['收盘'].pct_change(period)
    
    # 相对强弱指标 (RSI)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI_14'] = calculate_rsi(df['收盘'])
    df['RSI_28'] = calculate_rsi(df['收盘'], periods=28)
    
    # MACD指标
    exp12 = df['收盘'].ewm(span=12, adjust=False).mean()
    exp26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 布林带指标
    df['BB_middle'] = df['收盘'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['收盘'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['收盘'].rolling(window=20).std()
    
    # 成交量相关特征
    df['volume_price_ratio'] = df['成交量'] / df['收盘']
    df['volume_change'] = df['成交量'].pct_change()
    df['volume_std_20'] = df['成交量'].rolling(window=20).std()
    
    # 价格变化特征
    df['price_change_1d'] = df['收盘'].pct_change()
    df['price_change_5d'] = df['收盘'].pct_change(5)
    df['price_change_20d'] = df['收盘'].pct_change(20)
    
    # 价格区间特征
    df['price_range'] = (df['最高'] - df['最低']) / df['收盘']
    df['close_open_ratio'] = df['收盘'] / df['开盘']
    
    return df.dropna()

def prepare_data(df, target_col='收盘', test_size=0.2):
    """准备训练和测试数据"""
    df = create_features(df)
    
    feature_columns = [col for col in df.columns if col != target_col and col != '日期']
    X = df[feature_columns].astype(float)
    y = df[target_col].astype(float)
    
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, df['日期'][split_idx:]

def tune_xgboost(X_train, y_train):
    """XGBoost模型调优"""
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                             scoring='neg_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳分数: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    """主函数"""
    try:
        np.random.seed(42)
        
        stock_code = input("请输入股票代码（例如：GOOGL）：")
        start_date = input("请输入开始日期（YYYYMMDD）：")
        end_date = input("请输入结束日期（YYYYMMDD）：")
        df = get_stock_data(stock_code, start_date=start_date, end_date=end_date)

        print(f"数据获取完成，共 {len(df)} 条记录")
        
        X_train, X_test, y_train, y_test, scaler, feature_columns, test_dates = prepare_data(df)
        print(f"数据预处理完成，训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        print("\n开始调优XGBoost模型...")
        model = tune_xgboost(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n模型评估结果:")
        print(f"R2分数: {r2:.4f}")
        print(f"均方误差(MSE): {mse:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")
        print(f"平均绝对误差(MAE): {mae:.4f}")
        
        # 保存模型
        import joblib
        model_path = f"{stock_code}_xgboost_model.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存至: {model_path}")
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 6))
        plt.bar(feature_columns, model.feature_importances_)
        plt.xticks(rotation=45, ha='right')
        plt.title('特征重要性排序', fontproperties=custom_font, fontsize=14)
        plt.xlabel('特征', fontproperties=custom_font)
        plt.ylabel('重要性', fontproperties=custom_font)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return model, scaler, feature_columns
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    model, scaler, feature_columns = main()
