# @Time : 2022/4/3 12:54 

# @Author : qyellS

# @File : dataprocess.py 

# @Software: PyCharm
# @Time : 2022/3/28 12:34

# @Author : xx

# @File : dataprocess_Diqos.py

# @Software: PyCharm
import os
import numpy as np
import pandas as pd
import random
import pickle


home_path = os.path.dirname(os.path.abspath(__file__))
QWS_file_path = os.path.join(home_path, 'qws.csv')
df=pd.read_csv(QWS_file_path)
arr=np.array(df.iloc[:,:8])
## (1) Response Time	-
## (2) Availability		+
## (3) Throughput		+
## (4) Successability	+
## (5) Reliability		+
## (6) Compliance		+
## (7) Best Practices	+
## (8) Latency			-
row_num,col_nuw=arr.shape
print("数据集的维度:",row_num,"X",col_nuw)
colmax=arr.max(axis=0) # 每列的最大值
colmin=arr.min(axis=0) # 每列的最大值
print("每列的最大值:",colmax)
print("每列的最小值:",colmin)
neg_or_pos=['-','+','+','+','+','+','+','-'] # 正负属性
for k,att in enumerate(neg_or_pos): # 全部转化成负属性
	if att=='-':
		arr[:,k]=(arr[:,k]-colmin[k])/(colmax[k]-colmin[k])
	else:
		arr[:,k]=(colmax[k]-arr[:,k])/(colmax[k]-colmin[k])

Constriant_nos=random.sample(range(row_num), 6)
Constriants=arr[Constriant_nos,:]
print("QoS constraints:")
print(Constriants.shape)
Candidates=np.delete(arr,Constriant_nos,axis=0) # 删除数组的行
print("Service candidates:")
print(Candidates.shape)

def get_user_history(data):
    """
    在数据集中随机选取随机数量的web服务信息作为用户的历史调用记录，确定勿动
    :return:用户历史
    """
    user_history=[]
    service_num=random.randint(10,15)
#     随机生成service num个随机数，并从大到小排列，在data中选择对应序列的service作为调用历史。
    data_service_index=np.random.randint(0,len(data),service_num)
    for i in range(service_num):
        user_history.append(data[data_service_index[i]])
    user_history=np.asarray(user_history)
    return user_history

Historys=[]

for i in range(6):
    history=get_user_history(arr)
    Historys.append(history)


# 保存
with open('QWS.pickle', 'wb') as f:
    pickle.dump(Constriants, f)  # QoS Constraints
    pickle.dump(Candidates, f)   # QoS of Web service candidates
    pickle.dump(Historys,f)      #Q History call of users
