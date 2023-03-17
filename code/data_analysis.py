from pandas_profiling import ProfileReport
import pandas as pd

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

if __name__ == '__main__':
    for i in [0,10,20,30]:
        for j in range(2):
            data_analysis(i,j)