import numpy as np, pandas as pd, pickle, os

np.set_printoptions(linewidth=400)

home_path = os.path.dirname(os.path.abspath(__file__))
QWS_file_path = os.path.join(home_path, 'qws2.CSV')
def divide_datasets(num_of_constrains=6):
  data = pd.read_table(QWS_file_path, header=None, delimiter=',')
  pd_data = data.iloc[1:, :8].values.astype(float)
  item_size = pd_data.shape[0]
  data_dimensions = pd_data.shape[1]
  print(f'item_size: {item_size}\ndata_dimensions: {data_dimensions}')

  # 每一列的最大值和最小值
  column_max = np.max(pd_data, axis=0)
  column_min = np.min(pd_data, axis=0)
  # print(f'column_max: {column_max}\ncolumn_min: {column_min}')

  # 正负属性，如延迟越高越不好
  pos_or_neg = ['-', '+', '+', '+', '+', '+', '+', '-']
  for (index, value) in enumerate(pos_or_neg):
    if value == '-': # 如果是负属性，改变方向
      pd_data[:, index] = (pd_data[:, index] - column_min[index]) / (column_max[index] - column_min[index])
    else:
      pd_data[:, index] = (column_max[index] - pd_data[:, index]) / (column_max[index] - column_min[index])

  # 数据集的划分
  constrains_index = np.random.choice(item_size, num_of_constrains, replace=False)  # 随机选择6个索引
  constrains_service = pd_data[constrains_index, :]  # 选择6个索引的数据，作为约束集
  # print(f"constrains_service: \n{constrains_service}\nconstrains_service.shape: \n{constrains_service.shape}")

  candidates_service = np.delete(pd_data, constrains_index, axis=0)  # 删除6个索引的数据，作为候选集
  # print(f"candidates_service: \n{candidates_service}\ncandidates_service.shape: \n{candidates_service.shape}")

  return constrains_service, candidates_service


# 生成用户的服务调用历史
def generate_user_history(all_services):
  call_num = np.random.randint(10, 15 + 1)  # 随机生成10-15个服务调用
  call_history_index = np.random.choice(all_services.shape[0], call_num, replace=False)
  call_history_value = all_services[call_history_index - 1, :]
  return call_history_value


# 生成多个用户的服务调用历史记录
def generate_histories(num_of_users, all_services):
  histories = []
  for i in range(num_of_users):
    histories.append(generate_user_history(all_services))
  return histories


if __name__ == '__main__':
  constrains_service, candidates_service = divide_datasets()
  histories = generate_histories(10, candidates_service)
  print(f'constrains_service: \n{constrains_service}\nconstrains_service.shape: \n{constrains_service.shape}')
  print(f'candidates_service: \n{candidates_service}\ncandidates_service.shape: \n{candidates_service.shape}')
  [print(f'item_length:{len(item)}') for item in histories]
  pickle_path = os.path.join(home_path, 'qws.pickle')
  with open(pickle_path, 'wb') as f:
    pickle.dump(constrains_service, f)
    pickle.dump(candidates_service, f)
    pickle.dump(histories, f)
    f.close()
