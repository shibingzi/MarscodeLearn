import meteostat
from meteostat import Point, Daily
from datetime import datetime
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.impute import SimpleImputer

import numpy as np
import matplotlib.pyplot as plt

# 配置日志记录
logging.basicConfig(filename='weather_data.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_weather_data(latitude: str, longitude: str, elevation: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get weather data for a specific location and time period.

    Parameters:
    latitude (str): Latitude of the location.
    longitude (str): Longitude of the location.
    elevation (str): Elevation of the location.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: Processed weather data containing time, average temperature, minimum temperature, and maximum temperature.
    """
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Create a Point object
    latitude, longitude, elevation = [float(x) for x in [latitude, longitude, elevation]]
    #latitude, longitude, elevation = [np.deg2rad(x) for x in [latitude, longitude, elevation]]

    location = Point(latitude, longitude, elevation)

    # Check if cache file exists
    cache_file = f'weather_data_bingjing_{start_date.date()}_{end_date.date()}.csv'
    if os.path.exists(cache_file):
        # If cache file exists, read data directly
        data = pd.read_csv(cache_file, parse_dates=['time'])
    else:
        # If cache file does not exist, fetch weather data
        data = Daily(location, start_date, end_date)
        data = data.fetch()

        # Ensure date is the index of the DataFrame
        data.index = pd.to_datetime(data.index)
        # Reset index, keeping the date as a regular column
        data = data.reset_index()
        # Rename columns for better readability
        data.rename(columns={'index': 'time'}, inplace=True)

        # Save data to cache file
        data.to_csv(cache_file, index=False)

    # Check column names and process data
    if 'time' not in data.columns:
        error_message = "'time' column not found in data. Please check the data source or the API documentation."
        logging.error(error_message)
        raise KeyError(error_message)

    data = data[['time', 'tavg', 'tmin', 'tmax']]
    data.rename(columns={'time': '日期', 'tavg': '平均气温', 'tmin': '最低气温', 'tmax': '最高气温'}, inplace=True)

    # Return processed data
    return data

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicator features from the date and retain only the numerical columns to return new variables.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.

    Returns:
    pd.DataFrame: DataFrame with additional features.
    """
    # Extract year, month, day, and weekday from the date
    df['年'] = df['日期'].dt.year
    df['月'] = df['日期'].dt.month
    df['日'] = df['日期'].dt.day
    df['星期'] = df['日期'].dt.weekday

    # Calculate moving averages (1, 7, 14, 20 days)
    for days in [1, 7, 14, 20]:
        df[f'平均气温_{days}日移动平均'] = df['平均气温'].rolling(window=days).mean()

    # Calculate lag features
    for col in ['平均气温', '最低气温', '最高气温']:
        df[f'{col}_lag1'] = df[col].shift(1)

    # Calculate statistical features
    df['平均气温_7日标准差'] = df['平均气温'].rolling(window=7).std()

    # Use sine and cosine transformations to capture periodicity
    df['平均气温_sin'] = np.sin(2 * np.pi * df['日期'].dt.dayofyear / 365)
    df['平均气温_cos'] = np.cos(2 * np.pi * df['日期'].dt.dayofyear / 365)

    # Calculate the rate of change of average temperature
    df['平均气温变化率'] = df['平均气温'].diff()

    # Outlier handling using IQR
    Q1 = df['平均气温'].quantile(0.25)
    Q3 = df['平均气温'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['平均气温'] < (Q1 - 1.5 * IQR)) | (df['平均气温'] > (Q3 + 1.5 * IQR)))]

    # Drop the original date column
    df.drop('日期', axis=1, inplace=True)

    # Return the DataFrame with new features
    return df

def prepare_data(df: pd.DataFrame, target_col: str, test_size: float) -> tuple:
    """
    Prepare the data for training, including feature standardization and splitting into training and test sets.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    target_col (str): Name of the target column.
    test_size (float): Proportion of the data to be used as the test set.

    Returns:
    tuple: A tuple containing the training and test sets.
    """
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(target_col, axis=1))
    y = df[target_col]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor model.

    Parameters:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training target.

    Returns:
    RandomForestRegressor: Trained model.
    """
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    #   model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 假设 X_train 和 y_train 是包含缺失值的训练数据
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充
    X_train_imputed = imputer.fit_transform(X_train)
    
    # 将 y_train 转换为 numpy 数组并进行 reshape
    y_train_np = y_train.to_numpy()
    y_train_imputed = imputer.fit_transform(y_train_np.reshape(-1, 1))

    # 使用填充后的数据训练模型
    model.fit(X_train_imputed, y_train_imputed)

    # Train the model
    #model.fit(X_train, y_train)

    # Return the trained model
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate the predictions using R2 score, MSE, and RMSE.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    # Calculate evaluation metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Return the evaluation metrics
    return {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def plot_predictions(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, city: str):
    """
        Plot the predictions against the true values.

    Parameters:
    dates (pd.Series): Dates for the predictions.
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    city (str): Name of the city.
    """
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot the true values
    plt.plot(dates, y_true, label='True Values')

    # Plot the predicted values
    plt.plot(dates, y_pred, label='Predicted Values')

    # Add title and labels
    plt.title(f'Temperature Predictions for {city}')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
def main():
    # 日期范围
    start_date = '2023-01-01'
    end_date = '2024-11-30'

    # 地点信息
    latitude = '39.9042'
    longitude = '116.4074'
    elevation = '50'

    # 获取天气数据
    weather_data = get_weather_data(latitude, longitude, elevation, start_date, end_date)

    # 创建特征
    weather_data_with_features = create_features(weather_data)

    # 准备数据
    target_col = '平均气温'
    test_size = 0.2
    X_train, X_test, y_train, y_test = prepare_data(weather_data_with_features, target_col, test_size)

    # 训练模型
    model = train_model(X_train, y_train)

        # 评估模型
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    print(metrics)

    # 绘制预测结果
    # 将 X_test 转换为 DataFrame 并获取其索引
    X_test_df = pd.DataFrame(X_test, columns=weather_data_with_features.columns[:-1])
    plot_predictions(X_test_df.index, y_test, y_pred, 'Beijing')


if __name__ == '__main__':
    main()
