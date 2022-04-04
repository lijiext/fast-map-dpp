import numpy as np, pickle, math, os
from itertools import combinations
from scipy import spatial
from scipy import stats
from tqdm import tqdm

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


if __name__ == '__main__':
  with open(pkl_path, 'rb') as f:
    constrains_service = pickle.load(f)
    candidates_service = pickle.load(f)
    histories = pickle.load(f)
    f.close()
  thresholds = get_thresholds(candidates_service, alph=0.5)
  print('thresholds:', thresholds)
  # 构建约束集
  constrains_index = np.random.randint(0, len(constrains_service), 1)
  dimensions = 3
  sr = constrains_service[constrains_index, dimensions]
  print(f"constrains_index: {constrains_index}\ndimensions: {dimensions}\nsr: {sr}")
