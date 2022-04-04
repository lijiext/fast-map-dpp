import numpy as np, pickle, math, os
from itertools import combinations
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


def get_shannon_entropy(user_call_histories):
  """
  :param user_call_histories: 用户的服务调用历史记录
  :rtype list: 服务调用记录的熵值
  """
  indexs = [[np.argmax(i) for i in item] for item in user_call_histories]
  shannon_entropies = []
  for item in indexs:
    shannon_entropy = sum([count / len(item) * (math.log2(count / len(item))) for count in Counter(item).values()])
    shannon_entropies.append(shannon_entropy)
  return shannon_entropies


if __name__ == '__main__':
  with open(pkl_path, 'rb') as f:
    constrains_service = pickle.load(f)
    candidates_service = pickle.load(f)
    histories = pickle.load(f)
    f.close()
  # 计算相似度门槛，80% 的数据小于此值

  # thresholds = get_thresholds(candidates_service, alph=0.5)
  thresholds = 0.7122821713107376
  print('thresholds:', thresholds)
  # 构建约束集
  constrains_index = np.random.randint(0, len(constrains_service), 1)
  dimensions = 3
  sr = constrains_service[constrains_index, dimensions]
  print(f"constrains_index: {constrains_index}\ndimensions: {dimensions}\nsr: {sr}")
