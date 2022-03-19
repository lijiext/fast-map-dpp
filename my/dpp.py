import numpy as np
import math

np.set_printoptions(linewidth=400)

# 候选个数
item_size = 5
# 特征维度
feature_dimension = 3

max_length = 5
# epsilon
epsilon = 1E-10

# 生成模拟数据，符合标准正态分布，
# 正 归一化 ？
scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)
print('scores:', scores)

# 生成特征向量，每个 item 有 feature_dimension 个维度
feature_vectors = np.random.randn(item_size, feature_dimension)
print('feature_vectors:', feature_vectors, sep='\n')

# feature_vectors 的 l2 范式
# sqrt()
feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
print('l2_norm_feature_vectors:', feature_vectors, sep='\n')

# 根据特征向量矩阵，求相似矩阵
similarities = np.dot(feature_vectors, feature_vectors.T)
print('similarities:', similarities, sep='\n')

# 内核矩阵，gram 矩阵，半正定
# scores 是一维数组，1行 m 列
# score.reshape((item_size, 1))：将scores变成 m行1列的矩阵
# score.reshape((1, item_size))：将scores变成 1行m列的矩阵
kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))
print('reshaeped score:', scores.reshape((item_size, 1)) * scores.reshape((1, item_size)), sep='\n')
# 内核矩阵的对角线元素，为特征向量的范数？？？
print('kernel_matrix:', kernel_matrix, sep='\n')

cis = np.zeros((max_length, item_size))
di2s = np.copy(np.diag(kernel_matrix))

selected_items = list()

selected_item = np.argmax(di2s)
selected_items.append(selected_item)
while len(selected_items) < max_length:
    k = len(selected_items) - 1
    # 子行列式
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
print('scores:', scores)
print('selected_items_index:', selected_items)
print("selected_items_value:", scores[selected_items])
