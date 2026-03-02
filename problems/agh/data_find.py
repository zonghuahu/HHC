# import torch
#
# # 参数设置
# num_samples = 100  # 例如生成10个样本
# size = 50         # 每个样本有6个服务类型
#
# # 服务类型和对应的概率
# service_to_select = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# prob = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# # 按照概率逐行采样
# need = torch.stack([
#     service_to_select[torch.multinomial(prob, size, replacement=True)]
#     for _ in range(num_samples)
# ])
#
# # 打印结果查看
# print("生成的 need:")
# print(need)
#
# # 检查每一行的 5 的比例（可选）
# for i, row in enumerate(need):
#     ratio_5 = (row == 7).sum().item() / size
#     print(f"第 {i} 行中，7 的比例: {ratio_5:.2%}")

# 原始列表
# data = [  0,  3, 27, 22, 39,  0, 17,  0, 47, 30,  0,  1, 35, 33,  0,  2,  0,  4,  0,  5,  0,  6,  0,  7,  0,  8,  0,  9,  0, 10,  0, 11,  0, 12,  0, 46, 19,  0, 13,  0, 15,  0, 16,  0, 18,  0, 20,  0, 21,  0, 23,  0, 25,  0, 26,  0, 28,  0, 29,  0, 31,  0, 32,  0, 34,  0, 36,  0, 37,  0, 38,  0, 40,  0, 41,  0, 42,  0, 43,  0, 44,  0, 45,  0, 48,  0, 49,  0, 50,  0, 24,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
# expected = set(range(1, 51))       # 要求包含的数字集合
# actual = set(data)                 # 实际列表中的数字集合
#
# missing = sorted(expected - actual)
# extra = sorted(actual - expected - {0})
#
# if not missing:
#     print("✅ 列表中完整包含 1 到 50 的所有数字")
# else:
#     print(f"❌ 缺少以下数字: {missing}")
#
# if extra:
#     print(f"⚠️ 出现了 1~50 范围之外的数字: {extra}")
# import numpy as np
#
# mask1 = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
# mask2 = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1])
#
# mask_fleet = mask1 & mask2
# print(mask_fleet)

# import numpy as np
# import pickle
#
# # 设置随机种子
# np.random.seed(1234)
#
# num_nodes = 92 # 1个车库加上91个节点
# coords = np.random.rand(num_nodes, 2) * 100
#
# dist_matrix = np.zeros((num_nodes, num_nodes))
# for i in range(num_nodes):
#     for j in range(num_nodes):
#         dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
#
# # 转成字典形式 {(i, j): distance}
# distance_info = {}
# for i in range(num_nodes):
#     for j in range(num_nodes):
#         distance_info[(i, j)] = dist_matrix[i, j]
#
# # 保存字典到文件
# with open('distance.pkl', 'wb') as f:
#     pickle.dump(distance_info, f)
#
# with open('distance.pkl', 'rb') as f:
#     distance_info = pickle.load(f)
#     print('distance_info:', distance_info)


import numpy as np
import pickle
import os

np.random.seed(1234)

num_locations = 100  # 100 个患者住址位置
num_nodes = num_locations + 1  # 1 个 depot + 100 个位置

location_coords = np.random.rand(num_locations, 2) * 100

depot_x = np.mean(location_coords[:, 0])
depot_y = np.mean(location_coords[:, 1])
depot_coord = np.array([[depot_x, depot_y]])

coords = np.vstack([depot_coord, location_coords])

dist_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

distance_info = {}
for i in range(num_nodes):
    for j in range(num_nodes):
        distance_info[(i, j)] = dist_matrix[i, j]

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'distance.pkl'), 'wb') as f:
    pickle.dump(distance_info, f)

coords_info = {}
coords_info[0] = [depot_x, depot_y]
for i in range(num_locations):
    coords_info[i + 1] = [location_coords[i, 0], location_coords[i, 1]]

with open(os.path.join(script_dir, 'coordinates.pkl'), 'wb') as f:
    pickle.dump(coords_info, f)

print(f"Generated {num_nodes} nodes ({num_locations} locations + 1 depot)")
print(f"Distance matrix: {num_nodes}x{num_nodes} = {len(distance_info)} entries")
print(f"Depot at ({depot_x:.2f}, {depot_y:.2f})")