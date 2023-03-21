# 用于拟合校正
import numpy as np
import matplotlib.pyplot as plt

from data_analysis import get_data, kalman_filter

def fit_linear(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x_sq = np.sum(x**2)

    a = (n*sum_xy - sum_x*sum_y) / (n*sum_x_sq - sum_x**2)
    b = (sum_y - a*sum_x) / n

    return a, b

def connect_data(info):
    xwhat = []
    ground_truth = []
    for angle in info:
        for index in info[angle]:
            data = get_data(angle, index)
            data = kalman_filter(data, data, None, False)
            for item in data:
                xwhat.append(item)
                ground_truth.append(angle)

    return xwhat, ground_truth

# 定义误差计算函数
def calculate_error(predicted, actual, method):
    if method == 'mae':
        error = np.mean(np.abs(predicted - actual))
    elif method == 'mse':
        error = np.mean((predicted - actual)**2)
    elif method == 'rmse':
        error = np.sqrt(np.mean((predicted - actual)**2))
    elif method == 'mape':
        error = np.mean(np.abs((actual - predicted) / actual))
    else:
        raise ValueError('无效的误差计算方法')
    return error


def get_error(predicted,actual):
    # 计算误差
    mae_error = calculate_error(predicted, actual, 'mae')
    mse_error = calculate_error(predicted, actual, 'mse')
    rmse_error = calculate_error(predicted, actual, 'rmse')

    # 打印误差结果
    print(f'MAE误差：{mae_error:.3f}')
    print(f'MSE误差：{mse_error:.3f}')
    print(f'RMSE误差：{rmse_error:.3f}')


def correct():
    info={
        -10:[0,1,2,3],
        -20:[0,1],
        -30:[0,1,2,3],
        0:[0,1,2,3],
        10:[0,1,2,3],
        20:[0,1],
        30:[0,1]
    }
    xwhat, ground_truth = connect_data(info)
    a, b = fit_linear(np.array(xwhat), np.array(ground_truth))
    print(f"a = {a:.2f}, b = {b:.2f}")
    print('before correction:')
    get_error(np.array(xwhat), np.array(ground_truth))
    print('after correction:')
    get_error(a*np.array(xwhat)+b, np.array(ground_truth))

    plt.figure()
    plt.plot(xwhat, 'k+', label='raw')
    plt.plot(a*np.array(xwhat)+b, 'b-', label='after correction')
    plt.plot(ground_truth, 'g', label='ground truth')
    plt.legend()
    plt.xlabel('index')  
    plt.ylabel('angle')
    plt.savefig('../analysis/correction.png')

if __name__ == '__main__':
    correct()
    # a = 1.17, b = -5.77