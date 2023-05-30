import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product
import os
import ast
import pandas as pd
from math import sqrt, degrees, atan2, acos, radians, sin, cos
from sklearn.metrics import mean_squared_error

from radar_config import angle_correct_config

def calculate_distance_azimuth_elevation(point):
    x, y, z = point
    # 计算距离
    distance = math.sqrt(x**2 + y**2 + z**2)
    
    # 计算方位角，转换为度
    azimuth = math.atan2(y, x)
    azimuth = math.degrees(azimuth)
    if azimuth < 0:
        azimuth = azimuth + 360

    # 计算仰角，转换为度
    elevation = math.atan2(z, math.sqrt(x**2 + y**2))
    elevation = math.degrees(elevation)

    return distance, azimuth, elevation

def draw_3d_points_and_box(points, percentile=80):
    # 转换为numpy array方便处理
    points = np.array(points)

    # 获取x, y, z坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 计算箱子的边界
    x_min, x_max = np.percentile(x, [(100-percentile)/2, 100-(100-percentile)/2])
    y_min, y_max = np.percentile(y, [(100-percentile)/2, 100-(100-percentile)/2])
    z_min, z_max = np.percentile(z, [(100-percentile)/2, 100-(100-percentile)/2])

    # 打印箱子的边界
    '''
    print(f'x: {x_min:.2f} ~ {x_max:.2f}')
    print(f'y: {y_min:.2f} ~ {y_max:.2f}')
    print(f'z: {z_min:.2f} ~ {z_max:.2f}')
    ''' 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维点
    ax.scatter(x, y, z, c='b', marker='o')

    # 绘制箱子
    # 箱子的8个顶点
    corners = [(x_min, y_min, z_min), (x_min, y_min, z_max), (x_min, y_max, z_min), (x_min, y_max, z_max),
               (x_max, y_min, z_min), (x_max, y_min, z_max), (x_max, y_max, z_min), (x_max, y_max, z_max)]

    # 连接箱子的各个顶点
    edges = [(0,1), (0,2), (0,4), (2,3), (2,6), (1,3), (1,5), (4,5), (4,6), (3,7), (5,7), (6,7)]
    
    for edge in edges:
        ax.plot3D(*zip(corners[edge[0]], corners[edge[1]]), color="r", linewidth=2)

    plt.show()

def calculate_distance(point):
    x, y, z = point
    return math.sqrt(x**2 + y**2 + z**2)

def calculate_coordinates(distance, azimuth, elevation):
    # 转换角度为弧度
    azimuth = math.radians(azimuth)
    elevation = math.radians(elevation)

    # 计算坐标
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)

    return x, y, z

def calculate_rmse(result, ground_truth):
    # 将list转换为numpy array
    result = np.array(result)
    ground_truth = np.array(ground_truth)

    # 计算预测值和真实值的差值
    difference = result - ground_truth

    # 计算差值的平方
    square_difference = difference ** 2

    # 计算平方差的平均值
    mean_square_difference = np.mean(square_difference)

    # 计算平方根，得到RMSE
    rmse = np.sqrt(mean_square_difference)

    return rmse

def euclidean_distance(point1, point2):
    # 计算两点之间的欧式距离
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def mean_error(result, ground_truth):
    # 计算平均误差
    return np.mean([euclidean_distance(p1, p2) for p1, p2 in zip(result, ground_truth)])

def rmse_error(result, ground_truth):
    # 计算RMSE误差
    diff = np.sqrt(np.sum((np.array(result) - np.array(ground_truth)) ** 2, axis=1))
    return np.sqrt(np.mean(diff ** 2))

def max_error(result, ground_truth):
    # 计算最大误差
    return np.max([euclidean_distance(p1, p2) for p1, p2 in zip(result, ground_truth)])

def read_config(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('x\ty\tdistance\truntime'):
                x, y, distance, _ = lines[lines.index(line)+1].split('\t')
                return float(x), float(y), float(distance)

def read_result(path):
    with open(path, 'r') as f:
        result = ast.literal_eval(f.read())
    return result

def config_to_coord(x, y, distance):
    # 将x,y,distance 转换为空间坐标
    return x*15, y*15, distance

def read_folder(folder_path):
    config_path = os.path.join(folder_path, 'config.txt')
    result_path = os.path.join(folder_path, 'result.txt')

    # 读取config.txt并转换坐标
    x, y, distance = read_config(config_path)
    coord = config_to_coord(x, y, distance)

    # 读取result.txt
    result = read_result(result_path)

    return coord, result

def correct_angle(azimuth, elevation):
    corrected_azimuth = angle_correct_config.azimuth_a * azimuth + angle_correct_config.azimuth_b
    corrected_elevation = angle_correct_config.elevation_a * elevation + angle_correct_config.elevation_b
    return corrected_azimuth, corrected_elevation

def spherical_to_cartesian(azimuth, elevation, distance):
    x = distance * cos(radians(elevation)) * cos(radians(azimuth))
    y = distance * cos(radians(elevation)) * sin(radians(azimuth))
    z = distance * sin(radians(elevation))
    return x, y, z

def calculate_error_for_folder(folder_path):
    # 读取数据并计算真实坐标
    config_path = os.path.join(folder_path, 'config.txt')
    x, y, distance = read_config(config_path)
    true_coord = config_to_coord(x, y, distance)
    true_distance, true_azimuth, true_elevation = calculate_distance_azimuth_elevation(true_coord)

    # 读取结果并获取计算的azimuth、elevation和distance
    result_path = os.path.join(folder_path, 'result.txt')
    result = read_result(result_path)

    coord_result = []
    azimuth_result = []
    elevation_result = []
    distance_result = []

    for item in result:
        result_azimuth, result_elevation, result_distance = item[:3]
        # 先校正
        result_azimuth, result_elevation = correct_angle(result_azimuth, result_elevation)
        # 再计算坐标
        result_coord = calculate_coordinates(result_distance, result_azimuth, result_elevation)
        azimuth_result.append(result_azimuth)
        elevation_result.append(result_elevation)
        distance_result.append(result_distance)
        coord_result.append(result_coord)
    
    coord_rmse = rmse_error(coord_result, [true_coord] * len(coord_result))
    azimuth_rmse = calculate_rmse(azimuth_result, [true_azimuth] * len(azimuth_result))
    elevation_rmse = calculate_rmse(elevation_result, [true_elevation] * len(elevation_result))
    distance_rmse = calculate_rmse(distance_result, [true_distance] * len(distance_result))

    return {
        'coord_rmse': coord_rmse,
        'azimuth_rmse': azimuth_rmse,
        'elevation_rmse': elevation_rmse,
        'distance_rmse': distance_rmse,
    }

def calculate_error_for_all():
    # 创建一个空的 DataFrame 来存储所有的误差数据
    df = pd.DataFrame(columns=['coord_rmse', 'azimuth_rmse', 'elevation_rmse', 'distance_rmse'])

    for i in range(1, 49):
        folder_path = f'../data/position/{i}'
        print(folder_path)
        error = calculate_error_for_folder(folder_path)
        df = df.append(error, ignore_index=True)

    return df

# 执行函数并打印误差表
df = calculate_error_for_all()
print(df)
# 保存误差表
df.to_csv('../data/position_error.csv', index=False)