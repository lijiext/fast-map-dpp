import numpy as np, pickle, math, os
from itertools import combinations, permutations
from scipy import spatial
from scipy import stats
from tqdm import tqdm
from collections import Counter

home_path = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(home_path, 'qws.pickle')


def get_thresholds(Candidates=None, alph=0.5):
  # 计算欧氏距离,余弦相似度,QoS相似度
  sim_dict = {}
  i = 0
  # 排列
  print('calculating similarities among all items...')
  for Si, Sj in tqdm(list(combinations(range(len(Candidates)), 2))):
    temp_krcc, p_value = stats.kendalltau(Candidates[Si, :], Candidates[Sj, :])
    temp_dis = spatial.distance.euclidean(Candidates[Si, :], Candidates[Sj, :])
    temp_sim = (1 - alph) * (1.0 - temp_dis / math.sqrt(2.0)) + alph * temp_krcc
    sim_dict[(Si, Sj)] = temp_sim
    # i = i + 1
    # if i % 10000 == 0:
    #   pass
    # print(i)
  # thresh_sim=np.mean(list(sim_dict.values())) # average QoS similarity
  sim_arr = np.array(list(sim_dict.values()))
  # 80% 的数据小于此值
  thresh_sim = np.percentile(sim_arr, 80)
  return thresh_sim


def get_shannon_entropies(user_call_histories):
  """
  :param user_call_histories: 用户的服务调用历史记录
  :rtype list: 服务调用记录的熵值列表
  """
  indexs = [[np.argmax(i) for i in item] for item in user_call_histories]
  shannon_entropies = []
  for item in indexs:
    shannon_entropy = np.abs(
      sum([count / len(item) * (math.log2(count / len(item))) for count in Counter(item).values()]))
    shannon_entropies.append(shannon_entropy)
  return np.asarray(shannon_entropies)


def get_diversity_parameter(shannon_entropies, H0=1):
  """
  根据服务调用历史记录信息熵计算多样性程度
  :param shannon_entropies: 用户的服务调用历史记录多样性熵值列表
  :param H0: 超参数
  :return: 每一个用户的多样化程度
  """
  H_max = np.max(shannon_entropies)
  H_min = np.min(shannon_entropies)
  return np.asarray([(item - H_min + H0) / (H_max - H_min + H0) for item in shannon_entropies])


def get_similarity(item1, item2, alpha=0.5, dimensions=8.0):
  distance = spatial.distance.euclidean(item1, item2)
  tau = stats.kendalltau(item1, item2).correlation
  similarity = alpha * (1.0 - distance / np.sqrt(dimensions)) + (1.0 - alpha) * tau
  return similarity


def get_kernel_matrix(constrains_service, candidate_service, fu, alpha):
  similarities = np.asarray([get_similarity(item, constrains_service) for item in candidate_service])
  kernel_matrix = np.diag(np.square(similarities))
  for (i, j) in tqdm(list(combinations(range(len(candidate_service)), 2))):
    if i == j:
      continue
    kernel_matrix[i, j] = fu * alpha * similarities[i] * similarities[j] * get_similarity(candidate_service[i],
                                                                                          candidate_service[j])
    kernel_matrix[j, i] = kernel_matrix[i, j]
  return kernel_matrix

def dpp():
  pass

def get_diversity_of_list(hlist, alpha=0.5, dimensions=8.0):
  return 2 / (len(hlist) * (len(hlist) - 1)) * np.sum(
    [1 - get_similarity(hlist[i], hlist[j], alpha, dimensions) for i, j in list(permutations(range(len(hlist)), 2))])


if __name__ == '__main__':
  with open(pkl_path, 'rb') as f:
    constrains_service = pickle.load(f)
    candidates_service = pickle.load(f)
    histories = pickle.load(f)
    f.close()
  # 计算相似度门槛，80% 的数据小于此值

  # thresholds = get_thresholds(candidates_service, alph=0.5)
  # 0.7898718519689361
  thresholds = 0.7122821713107376
  print('thresholds:', thresholds)
  # 构建约束集
  constrains_index = np.random.randint(0, len(constrains_service), 1)
  dimensions = 3
  sr = constrains_service[constrains_index, dimensions]
  print(f"constrains_index: {constrains_index}\ndimensions: {dimensions}\nsr: {sr}")

  # 计算调用历史记录的信息熵(多样性)
  shannon_entropies = get_shannon_entropies(histories)
  # 归一化
  diversity_parameter = get_diversity_parameter(shannon_entropies)
  # 计算调用历史记录多样性程度
  print([get_diversity_of_list(item) for item in histories])
