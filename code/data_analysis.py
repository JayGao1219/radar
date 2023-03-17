from pandas_profiling import ProfileReport
from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)

def list2df(list):
    df = pd.DataFrame(list)
    return df

def get_data(angle,index):
    file_path='../data/%d/%d.txt'%(angle,index)
    with open(file_path,'r') as f:
        data = f.read()
        data = eval(data)
    res=[]
    for i in data:
        if i>39 or i<-39:
            continue
        res.append(i)
    return res

def data_analysis(angle,index):
    data = get_data(angle,index)
    df = list2df(data)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(output_file="../analysis/%d_%d.html"%(angle,index))

def kalman_filter(data,ground_truth, file_path):
    n_iter = len(data)
    sz = (n_iter,) # size of array
    x = ground_truth # truth value (typo in example at top of p. 13 calls this z)
    z = np.array(data)
    xhat=np.zeros(sz)      # x 滤波估计值  
    P=np.zeros(sz)         # 滤波估计协方差矩阵  

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.R = np.array([0.1**2])
    kf.P = np.array([1.0])
    kf.Q = 1e-5 
    xhat[0] = 0.0  
    P[0] = 1.0 
    for k in range(1,n_iter):  
        kf.predict()
        xhat[k] = kf.x
        kf.update(z[k], 0.1**2, np.array([1]))

    plt.figure()  
    plt.plot(z,'k+',label='noisy measurements')     #观测值  
    plt.plot(xhat,'b-',label='a posteri estimate')  #滤波估计值  
    plt.axhline(x,color='g',label='truth value')    #真实值  
    plt.legend()
    plt.xlabel('index')  
    plt.ylabel('angle')
    plt.savefig(file_path)

def kalman_filter_test(angle,index):
    data = get_data(angle,index)
    kalman_filter(data,angle, '../analysis/%d_%d.png'%(angle,index))

if __name__ == '__main__':
    for i in [0,10,20,30]:
        for j in range(2):
            kalman_filter_test(i,j)
            # data_analysis(i,j)
