import meteostat # type: ignore
from meteostat import Point, Daily # type: ignore
from datetime import datetime
import pandas as pd
import logging
import os

# 配置日志记录
logging.basicConfig(filename='weather_data.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_weather_data(point, start_date, end_date):
    """
    获取指定地点和时间范围内的天气数据，并返回处理后的数据。

    参数:
    point (list): 包含纬度、经度和海拔的列表。
    start_date (str): 开始日期，格式为 'YYYY-MM-DD'。
    end_date (str): 结束日期，格式为 'YYYY-MM-DD'。

    返回:
    pandas.DataFrame: 处理后的数据，包含时间、平均气温、最低气温和最高气温。
    """
    # 将字符串日期转换为 datetime 对象
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # 创建 Point 对象
    # Bug修复: 改为使用正确的参数名latitude、longitude、altitude
    location = Point(point[0], point[1], point[2])

    # 检查缓存文件是否存在
    # Bug修复: 简化文件名，只使用日期而不使用时间
    cache_file = f'weather_data_{point[0]}_{point[1]}_{start_date.date()}_{end_date.date()}.csv'
    if os.path.exists(cache_file):
        # 如果缓存文件存在，直接读取数据
        data = pd.read_csv(cache_file, parse_dates=['time'])
    else:
        # 如果缓存文件不存在，获取天气数据
        data = Daily(location, start_date, end_date)
        data = data.fetch()

        # 确保日期是DataFrame的索引
        data.index = pd.to_datetime(data.index)
        # 重置索引，将日期作为普通列，并保留原有索引
        data = data.reset_index()
        # 重命名列，使其更易于理解
        data.rename(columns={'index': 'time'}, inplace=True)

        # 将数据保存到缓存文件
        data.to_csv(cache_file, index=False)

    # 检查列名并处理数据
    if 'time' not in data.columns:
        error_message = "'time' column not found in data. Please check the data source or the API documentation."
        logging.error(error_message)
        raise KeyError(error_message)

    data = data[['time', 'tavg', 'tmin', 'tmax']]
    data.rename(columns={'time': '日期', 'tavg': '平均气温', 'tmin': '最低气温', 'tmax': '最高气温'}, inplace=True)

    # 返回处理后的数据
    return data

# 示例调用
point = [39.9042, 116.4074, 44]  # 北京的经纬度和海拔
start_date = '2023-01-01'
end_date = '2024-11-30'

try:
    weather_data = get_weather_data(point, start_date, end_date)
    print(weather_data)
except KeyError as e:
    print(f"An error occurred: {e}")
