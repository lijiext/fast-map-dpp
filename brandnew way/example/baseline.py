# @Time : 2022/4/3 13:43

# @Author : qyellS

# @File : baseline.py

# @Software: PyCharm
# @Time : 2022/3/28 12:49

# @Author : xx

# @File : baseline.py

# @Software: PyCharm
import os
import pickle
import random
from scipy import spatial
import numpy as np
import scipy.stats as stats
import math
from itertools import combinations
from sklearn.cluster import KMeans
from collections import Counter


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

def RS(Constraint=None, Candidates=None, top_k=None):
    row_num, col_nuw = Candidates.shape
    selected_nos = random.sample(range(row_num), top_k)
    return selected_nos


def KNN(Constraint=None, Candidates=None, top_k=None):
    row_num, col_nuw = Candidates.shape
    distances = []
    for k in range(row_num):
        distance = spatial.distance.euclidean(Constraint, Candidates[k, :])
        distances.append(distance)
    distance_tuple = list(zip(range(row_num), distances))
    distance_tuple.sort(key=lambda x: x[1])  # reverse=True
    selected_nos = [x[0] for x in distance_tuple]
    return selected_nos[:top_k]


def DQCSR_CC(Constraint=None, Candidates=None, top_k=None):
    clf = KMeans(n_clusters=top_k)
    clf.fit(Candidates)
    centers = clf.cluster_centers_  # ndarray
    labels = clf.labels_  # 每个数据点所属分组 ndarray
    label_dict = {}
    for index, class_no in enumerate(labels):
        temp_list = label_dict.get(class_no, [])
        temp_list.append(index)
        label_dict[class_no] = temp_list
    selected_nos = []
    for class_no in range(top_k):
        distances = centers[class_no, :] - Candidates[label_dict[class_no], :]
        distances = (distances ** 2).sum(axis=1)
        distances = distances ** (1 / 2)
        ranked_index = distances.argsort()  # 升序排序,返回索引
        selected_service_no = label_dict[class_no][ranked_index[0]]
        selected_nos.append(selected_service_no)
    return selected_nos


def DQCSR_CR(Constraint=None, Candidates=None, top_k=None):
    clf = KMeans(n_clusters=top_k)
    clf.fit(Candidates)
    centers = clf.cluster_centers_  # ndarray
    labels = clf.labels_  # 每个数据点所属分组 ndarray
    label_dict = {}
    for index, class_no in enumerate(labels):
        temp_list = label_dict.get(class_no, [])
        temp_list.append(index)
        label_dict[class_no] = temp_list
    selected_nos = []
    for class_no in range(top_k):
        class_radius = []
        for Si in label_dict[class_no]:
            Si_radius = [0]
            for Sj in label_dict[class_no]:
                if Si != Sj:
                    temp_radius = spatial.distance.euclidean(Candidates[Si, :], Candidates[Sj, :])
                    Si_radius.append(temp_radius)
            max_Si_radius = max(Si_radius)
            class_radius.append(max_Si_radius)
        ranked_index = np.array(class_radius).argsort()  # 升序排序,返回索引
        selected_nos.append(label_dict[class_no][ranked_index[0]])
    return selected_nos


def DiQoS(sn=None, Constraint=None, Candidates=None, top_k=None, lamda=0.5):
    results = []
    K = len(sn["nodes"])  # 节点数量
    candidate_nos = list(range(K))
    for _ in range(top_k):  # 迭代选择top_k个服务
        selected_no = candidate_nos[0]
        max_rank_score = (1 - lamda) * sn["nodes"][selected_no] + lamda * ddiversity(sn=sn,
                                                                                    resutls=results + [selected_no])
        for temp_no in candidate_nos[1:]:
            temp_rank_score = (1 - lamda) * sn["nodes"][temp_no] + lamda * ddiversity(sn=sn, resutls=results + [temp_no])
            if temp_rank_score > max_rank_score:
                max_rank_score = temp_rank_score
                selected_no = temp_no
        candidate_nos.remove(selected_no)
        results.append(selected_no)
    return results


def A_dominates_B(A, B):  # A and B must be np.array
    if all(A <= B) and any(A < B):
        isDominate = True
    else:
        isDominate = False
    return isDominate


def mapping(Constraint=None, Candidates=None):
    Candidates = abs(Candidates - Constraint)
    return Candidates


def Skyline(Candidates=None):
    row_num, col_nuw = Candidates.shape
    skyline_nos = []
    for i in range(row_num):
        A = Candidates[i, :]
        dominated = True
        for j in range(row_num):
            B = Candidates[j, :]
            if i == j:
                continue
            if A_dominates_B(B, A):
                dominated = False
        if dominated:
            skyline_nos.append(i)
    return skyline_nos


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


def get_thresholds(Candidates=None, alph=0.5):
    # 计算欧氏距离,余弦相似度,QoS相似度
    sim_dict = {}
    i = 0;
    for Si, Sj in list(combinations(range(len(Candidates)), 2)):
        temp_krcc, p_value = stats.kendalltau(Candidates[Si, :], Candidates[Sj, :])
        temp_dis = spatial.distance.euclidean(Candidates[Si, :], Candidates[Sj, :])
        temp_sim = (1 - alph) * (1.0 - temp_dis / math.sqrt(2.0)) + alph * temp_krcc
        sim_dict[(Si, Sj)] = temp_sim
        i = i + 1
        if i % 10000 == 0:
            print(i)
    # thresh_sim=np.mean(list(sim_dict.values())) # average QoS similarity
    sim_arr = np.array(list(sim_dict.values()))
    thresh_sim = np.percentile(sim_arr, 80)
    return thresh_sim


def service_network(Constraint=None, Candidates=None, sim_threshold=None, alph=0.5):
    # 计算节点的score
    scores = []
    for Candidate in Candidates:  # compute similarity between Sr and Si
        tau, p_value = stats.kendalltau(Constraint, Candidate)
        dis = spatial.distance.euclidean(Constraint, Candidate)
        temp_score = alph * (1.0 - dis / math.sqrt(2.0)) + (1 - alph) * tau
        scores.append(temp_score)
    # 计算欧氏距离，余弦距离 --> 计算QoS相似度 --> 构建服务网络图的边
    edges = []
    for Si, Sj in list(combinations(range(len(Candidates)), 2)):
        temp_krcc, p_value = stats.kendalltau(Candidates[Si, :], Candidates[Sj, :])
        temp_dis = spatial.distance.euclidean(Candidates[Si, :], Candidates[Sj, :])
        temp_sim = (1 - alph) * (1.0 - temp_dis / math.sqrt(2.0)) + alph * temp_krcc
        if temp_sim >= sim_threshold:
            edges.append((Si, Sj))
    sn = {"nodes": scores, "edges": edges}
    return sn


def ddiversity(sn=None, resutls=None):
    K = len(sn["nodes"])  # 节点数量
    ES = set(resutls)  # Expanded Set
    for service_no in resutls:
        for edge in sn["edges"]:  # edge=(i,j)
            if service_no in edge:
                ES = ES | set(edge)
    ER = len(ES) / K  # Expansion Ratio
    return ER


def diversity(list=None):
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


def get_cos_similar(v1,v2):
    vector_a = np.mat(v1)
    vector_b = np.mat(v2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos



    return sim

def getscore(item1,item2):
    #Formula(1)

    dis = spatial.distance.euclidean(item1, item2)
    tau, p_value = stats.kendalltau(item1, item2)
    temp_score = 0.5 * (1.0 - dis / math.sqrt(8.0)) + (1 - 0.5) * tau
    return temp_score



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

def Kmatrix(constraint=None,candidate=None,fu=None,alpha=None):

    scores=[]
    kernel_matrix=np.random.randn(len(candidate),len(candidate))
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

                kernel_matrix[j][k] = fu * alpha * scores[j] * scores[k] * get_cos_similar(candidate[j], candidate[k])
    return kernel_matrix

def rmsde(recommend_list,user_history,top_k=None):
    diversity_u=diversity(recommend_list)
    diversity_h=diversity(user_history)
    RMSDE_value=math.sqrt((diversity_u-diversity_h)**2/top_k)
    return RMSDE_value

if __name__ == '__main__':
    divers = []
    a=[]
    with open('QWS.pickle', 'rb') as f:
        Constraints = pickle.load(f)  # (6, 8)
        Candidates = pickle.load(f)  # (2501, 8)
        Historys=pickle.load(f)
    # sim_threshold=get_thresholds(Candidates=Candidates,alph=0.5)
    # print("sim_threshold:",sim_threshold)
#my=0.6648299660485001
    sim_threshold = 0.7122835781845623  # 80%
    # sim_threshold = 0.5523330169011025	# mean

    # 参数变化
    n_list = [1000,1200,1400,1600,1800,2000]  # number of service candidates
    k_list = [3, 4, 5, 6, 7, 8]  # top-k
    d_list = [3, 4, 5, 6, 7, 8]  # QoS dimensions
    c_list = [0, 1, 2, 3, 4, 5]  # QoS constraints
    # 默认参数
    n = 1200
    top_k = 5
    d = 4
    c_k = 1
    lamda = 0.8
    H0=1
    a0=0.9

    # 参数变化结果
    DCG_arr = np.zeros((7, 6))
    Div_arr = np.zeros((7, 6))
    RMSDE_arr=np.zeros((7, 6))
    dives = []
    for index, n in enumerate(n_list):
    #for index,d in enumerate(d_list):
    # for index,top_k in enumerate(k_list):

        Sr = Constraints[c_k, :d]

        Services = Candidates[:n, :d]

        # 计算 Dynamic Skyline Services
        mapped_Candidates = mapping(Constraint=Sr, Candidates=Services)
        skyline_nos = Skyline(Candidates=mapped_Candidates)
        skyline_serves = Services[skyline_nos, :]

        # 构建QoS相似服务网络
        sn = service_network(Constraint=Sr, Candidates=skyline_serves, sim_threshold=sim_threshold, alph=0.5)

        # 构建 归一化香农熵
        history=Historys[0]
        Shannon_Entropys=[]
        for i in Historys:
            Shannon_Entropy=shannon_Entropy(i)
            Shannon_Entropys.append(Shannon_Entropy)
        fus=get_fu(Shannon_Entropys,H0)
        diversity_u=diversity(Historys[0])

        # 调用服务推荐方法
        ##########################################################

        # (1)调用DSL_RS方法
        DCG_values = []
        diversities = []
        RMSDE_values=[]

        for time in range(10):
            results_RS = RS(Constraint=Sr, Candidates=skyline_serves, top_k=top_k)
            results = skyline_serves[results_RS, :]

            temp_DCG = DCG(Constraint=Sr, items=results, alph=0.5)
            temp_diversity = diversity(results)
            temp_RMSDE=rmsde(results,history,top_k)
            DCG_values.append(temp_DCG)
            diversities.append(temp_diversity)
            RMSDE_values.append(temp_RMSDE)
        DCG_value = np.mean(DCG_values)

        div = np.mean(diversities)
        RMSDE=np.mean(RMSDE_values)

        DCG_arr[0, index] = DCG_value
        Div_arr[0, index] = div
        RMSDE_arr[0,index]=RMSDE



    #(2)调用DSL_KNN方法
        results_KNN = KNN(Constraint=Sr, Candidates=skyline_serves, top_k=top_k)
        results = skyline_serves[results_KNN, :]
        DCG_value = DCG(Constraint=Sr, items=results, alph=0.5)
        div = diversity(results)
        DCG_arr[1, index] = DCG_value
        Div_arr[1, index] = div
        RMSDE=rmsde(results,history,top_k)
        RMSDE_arr[1, index] = RMSDE


    #(3)调用DQCSR_CC方法
        results_CC = DQCSR_CC(Constraint=Sr, Candidates=skyline_serves, top_k=top_k)
        results = skyline_serves[results_CC, :]
        DCG_value = DCG(Constraint=Sr, items=results, alph=0.5)
        div = diversity( results)
        DCG_arr[2, index] = DCG_value
        Div_arr[2, index] = div
        RMSDE = rmsde(results, history,top_k)
        RMSDE_arr[2, index] = RMSDE
        # (4)调用DQCSR_CR方法
        results_CR = DQCSR_CR(Constraint=Sr, Candidates=skyline_serves, top_k=top_k)
        results = skyline_serves[results_CR, :]
        DCG_value = DCG(Constraint=Sr, items=results, alph=0.5)
        div = diversity( results)
        DCG_arr[3, index] = DCG_value
        Div_arr[3, index] = div
        RMSDE = rmsde(results, history,top_k)
        RMSDE_arr[3, index] = RMSDE

        # (5)调用DiQoS方法
        results_DQ = DiQoS(sn=sn, Constraint=Sr, Candidates=skyline_serves, top_k=top_k, lamda=lamda)
        results = skyline_serves[results_DQ, :]
        DCG_value = DCG(Constraint=Sr, items=results, alph=0.5)
        div = diversity( results)
        DCG_arr[4, index] = DCG_value
        Div_arr[4, index] = div
        RMSDE = rmsde(results, history,top_k)
        RMSDE_arr[4, index] = RMSDE


        # (6) 调用DPP方法
        kernel_matrix=Kmatrix(constraint=Sr,candidate=Services,fu=1,alpha=a0)
        results_DPP=dpp(kernel_matrix,top_k)

        results=skyline_serves[results_DPP,:]
        DCG_value=DCG(Constraint=Sr,items=results,alph=0.5)
        div=diversity(results)
        DCG_arr[5, index] = DCG_value
        Div_arr[5, index] = div
        RMSDE = rmsde(results, history,top_k)
        RMSDE_arr[5, index] = RMSDE

        
        # (7) 调用PDPP方法
        kernel_matrix=Kmatrix(constraint=Sr,candidate=Services,fu=fus[0],alpha=a0)
        results_PDPP = dpp(kernel_matrix, top_k)
        results=skyline_serves[results_PDPP,:]
        DCG_value=DCG(Constraint=Sr,items=results,alph=0.5)
        div=diversity(results)
        DCG_arr[6, index] = DCG_value
        Div_arr[6, index] = div
        RMSDE = rmsde(results, history,top_k)
        RMSDE_arr[6, index] = RMSDE


    # np.savetxt("DCG_value_n.csv", DCG_arr, delimiter=',', fmt='%.8f')

    # np.svetxt("RMSDE_value_n.scv",RMSDE_arr,delimiter=',', fmt='%.8f')
    # np.savetxt("DCG_value_d.csv",DCG_arr,delimiter=',',fmt='%.8f')
    # np.savetxt("results\\Div_value_d.csv",Div_arr,delimiter=',',fmt='%.8f')
    # np.savetxt("DCG_value_k.csv",DCG_arr,delimiter=',',fmt='%.8f')
    # np.savetxt("Div_value_d.csv",Div_arr,delimiter=',',fmt='%.8f')
    # np.savetxt("Div_value_n.csv",Div_arr,delimiter=',',fmt='%.8f')
