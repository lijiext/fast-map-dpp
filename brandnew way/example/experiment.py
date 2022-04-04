# @Time : 2022/4/3 16:15 

# @Author : xx

# @File : experiment.py 

# @Software: PyCharm

# @Time : 2022/3/29 9:43

# @Author : xx

# @File : RMSDE1.py

# @Software: PyCharm
import pickle
import numpy as np
import pickle
import random
from scipy import spatial
import numpy as np
import scipy.stats as stats
import math
from itertools import combinations
from collections import Counter



def get_cos_similar(v1,v2):
    """
    获取余弦相似度
    :param v1:
    :param v2:
    :return:
    """
    l1=np.linalg.norm(v1)
    l2=np.linalg.norm(v2)
    if l1*l2==0:
        return 0
    else:
        sim=(float(np.matmul(v1,v2)))/(l1*l2)



    return sim

def Diversity(list=None):
	lenth = len(list)
	if (lenth <= 1):
		return 0
	diversity = 0
	for i in range(lenth):
		for j in range(lenth):
			a = 1 - getscore(list[i],list[j])
			diversity+=a
	diversity = 2 / (lenth * (lenth - 1)) * diversity

	return diversity
def Kmatrix(constraint=None,candidate=None,fu=None,alpha=None):
    scores=[]
    kernel_matrix=np.random.randn(1200,1200)

#     获取每个服务的得分
    for i in range(len(candidate)):
        score=getscore(candidate[i],constraint)
        scores.append(score)

    #为kernel matrix赋值
    for j in range(len(candidate)):

        for k in range(len(candidate)):
            if(j==k):
                kernel_matrix[j][k]=scores[j]*scores[k]
            else:

                kernel_matrix[j][k] = fu * alpha * scores[j] * scores[k] *getscore(Candidates[j],Candidates[k])
    print("matrix success")
    return kernel_matrix
def getscore(item1,item2):

    dis = spatial.distance.euclidean(item1, item2)
    tau, p_value = stats.kendalltau(item1, item2)
    temp_score = 0.5 * (1.0 - dis / math.sqrt(8.0)) + (1 - 0.5) * tau
    return temp_score

def DCG(Constraint=None, items=None, alph=0.5):
	scores = []
	DCG_value = 0.0
	for one_row in items:
		tau, p_value = stats.kendalltau(Constraint, one_row)
		dis = spatial.distance.euclidean(Constraint, one_row)
		temp_score = alph * (1.0 - dis / math.sqrt(2.0)) + (1 - alph) * tau
		scores.append(temp_score)
	for index, score in enumerate(scores):
		pi = index + 1
		DCG_value = DCG_value + (2 ** score - 1) / math.log2(1 + pi)
	return DCG_value
def get_fu(Shannon_Entropys=None,H0=None):
    """
    Formula(5)
    :param Shannon_Entropys: All Shanoon_Entropy in users
    :param H0:  hyper parameter
    :return: normal shannon entropy in users
    """
    Hmin=min(Shannon_Entropys)
    Hmax=max(Shannon_Entropys)

    fus=[]

    for each in Shannon_Entropys:
        fu=(each - Hmin + each) / (Hmax - Hmin + each)
        fus.append(fu)

    return fus

def shannon_Entropy(History=None):
    # 获取用户的香浓熵， Formula(4)
    Shannon_Entropy=0
    indexs=[]
    for i in History:
        maxindex=np.argmax(i)
        indexs.append(maxindex)
    num_Count=Counter(indexs)
    important_index=[]
    for i in num_Count.values():
        important_index.append(i)
    fenmu = len(indexs)
    for i in important_index:
        Shannon_Entropy += -i / fenmu * (math.log2(i / fenmu))

    return Shannon_Entropy

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def rmsde(recommend_list,user_history,top_k=None):
    diversity_u=Diversity(recommend_list)
    diversity_h=Diversity(user_history)
    RMSDE_value=math.sqrt((diversity_u-diversity_h)**2/top_k)
    return RMSDE_value


with open('QWS.pickle', 'rb') as f:
    Constraints = pickle.load(f)  # (6, 8)
    Candidates = pickle.load(f)  # (2501, 8)
    Historys = pickle.load(f)

Shannon_Entropys=[]

for i in Historys:
    Shannon_Entropy = shannon_Entropy(i)
    Shannon_Entropys.append(Shannon_Entropy)

Constraints=Constraints[ :,:4]


Services=Candidates[:1200,:4]

# 归一化


divs=[]
dcgs=[]
RMSDEs=[]
H0s=[0,0.5,1.0,1.5,2]
alphas=[0.7,0.8,0.9,1.0,1.1]
for H0 in H0s:
    for alpha in alphas:
        for i in range(len(Constraints)):
            shangs=get_fu(Shannon_Entropys,H0)
            K=Kmatrix(Constraints[i],Services,fu=shangs[i],alpha=alpha)
            index=dpp(k,max_length=5)
            results=[]
            for k in index:
                results.append(Services[k])
            dcg=DCG(Constraints[i],results)
            dcgs.append(dcg)
            div=Diversity(results)
            divs.append(div)
            RMSDE=rmsde(results,Historys[i],5)
            RMSDEs.append(RMSDE)
    print("H0:",H0," alpha:",alpha," DCG value:",np.mean(dcgs)," Diverty value:",np.mean(divs)," RMSDE value:",np.mean(RMSDEs))


