from pandas_profiling import ProfileReport
from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)

from radar_config import position_cofig

def list2df(list):
    df = pd.DataFrame(list)
    return df

def get_data(angle,index):
    file_path='../data/%d/%d.txt'%(angle,index)
    with open(file_path,'r') as f:
        data = f.read().split('\n')
        for item in data:
            if '[' in item:
                data = item
                break
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

def kalman_filter(data,ground_truth, file_path, plot=False):
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

    if plot:
        plt.figure()  
        plt.plot(z,'k+',label='noisy measurements')     #观测值  
        plt.plot(xhat,'b-',label='a posteri estimate')  #滤波估计值  
        plt.axhline(x,color='g',label='truth value')    #真实值  
        plt.legend()
        plt.xlabel('index')  
        plt.ylabel('angle')
        plt.savefig(file_path)
    else:
        return xhat.tolist()

def kalman_filter_test(angle,index):
    data = get_data(angle,index)
    kalman_filter(data,angle, '../analysis/%d_%d.png'%(angle,index), plot=True)

# data analysis for position
def get_real_position(x,y):
    return x*position_cofig.dx, y*position_cofig.dy, position_cofig.distance

def get_position_data(index):
    file_path='%s%d/'%(position_cofig.root,index)

    info={}
    names=[]
    with open(file_path+'config.txt','r') as f:
        context = f.read().split('\n')
        flag=False
        for item in context:
            if flag:
                value = item.split('\t')
                for i in range(len(names)):
                    info[names[i]]=int(value[i])
                break

            if 'distance' in item:
                flag=True
                names = item.split('\t')

    x,y,z = get_real_position(info['x'],info['y'])
    info['coordinate']=[x,y,z]

    raw_data=np.load(file_path+'radar_raw_data.npy')

    with open(file_path+"result.txt") as f:
        result=eval(f.read())

    with open(file_path+"timestamps.txt") as f:
        timestamps=eval(f.read())

    return info,raw_data,result,timestamps

def change_list_2_dict(names,l):
    res={}
    for item in names:
        res[item]=[]
    for item in l:
        for i in range(len(names)):
            res[names[i]].append(item[i])
    return res            

def change_list_2_df(names,l):
    dictionary = change_list_2_dict(names,l)
    return pd.DataFrame(dictionary)

def analysis_position(index):
    info,raw_data,result,timestamps = get_position_data(index)
    names=['azimuth','elevation','range','x','y','z']
    df = change_list_2_df(names,result)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(output_file="../analysis/position/position_%d_(%d_%d_%d).html"%(index,info['coordinate'][0],info['coordinate'][1],info['coordinate'][2]))



if __name__ == '__main__':
    for i in range(18):
        analysis_position(i+1)
