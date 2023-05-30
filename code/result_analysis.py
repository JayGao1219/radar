import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product

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

