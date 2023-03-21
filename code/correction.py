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

    plt.figure()
    plt.plot(xwhat, 'k+', label='raw')
    plt.plot(a*np.array(xwhat)+b, 'b-', label='after correction')
    plt.plot(ground_truth, 'g', label='ground truth')
    plt.legend()
    plt.xlabel('index')  
    plt.ylabel('angle')
    plt.savefig('correction.png')

if __name__ == '__main__':
    correct()
    # a = 1.17, b = -5.77