from dpp import *

import time

# item 个数
item_size = 9
feature_dimension = 1
max_length = 3

# 初始化，生成item_size维度的array 标准正态分布
scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)
# 特征向量，每个item都有一个feature_dimension维度的特征向量
feature_vectors = np.random.randn(item_size, feature_dimension)

# 按行向量处理，保持矩阵的维度
# 将item表示成为特征向量，夹角越小越相似
feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
# 求乘法，作为相似度
similarities = np.dot(feature_vectors, feature_vectors.T)
# 半正定的内核矩阵，Gram矩阵

kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))

print('kernel matrix generated!')


t = time.time()
result = dpp(kernel_matrix, max_length)
print('dpp finished!\n', result)
print('algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))

# 基于窗口的长列表推荐
window_size = 10
t = time.time()
result_sw = dpp_sw(kernel_matrix, window_size, max_length)
print('dpp finished!\n', result_sw)
print('sw algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
