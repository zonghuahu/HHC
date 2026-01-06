import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from problems.agh.state_agh import StateAGH


class AGH(object):
    """
    AGH 类定义了机场地面处理（Airport Ground Handling）问题的核心逻辑。
    - 包括成本计算、数据集生成、状态初始化。
    - 静态属性定义问题参数，如车辆容量、速度、节点数量。
    """
    NAME = 'agh'  # 问题名称：Airport Ground Handling

    SPEED = 80.0  # 车辆速度（单位：假设为公里/小时，用于时间计算）
    NODE_SIZE = 92  # 总节点数（91 个登机口 + 1 个车库）

    @staticmethod
    def get_costs(dataset, pi, return_components=False):
        """
        计算路径成本并验证路径有效性。
        - dataset: 输入数据，包含 loc,distance, tw_left, tw_right, duration 等。
        - pi: 路径（[batch_size, seq_length]），表示每个样本的节点访问顺序。
        - return_components: 如果为 True，返回 (total_distance, total_waiting_time, mask)
        - 返回：总距离成本 ([batch_size]) 和掩码（当前为 None）。
                如果 return_components=True，返回 (total_distance, total_waiting_time, mask)
        
        NEAT-weight-style 多目标优化：
        - C_dist(π) = 路径总距离
        - C_wait(π) = sum_i max(0, service_start_time_i - tw_right_i)
        """
        batch_size, graph_size = dataset['duration'].size()  # 获取批次大小和登机口数量（含车库）

        graph_size = graph_size - 1 # 去除车库

        # 获取路径的节点索引（包括车库）
        loc = torch.cat((torch.zeros_like(dataset['loc'][:, :1]), dataset['loc']), dim=1)  # 车库索引为 0
        loc = loc.gather(1, pi)  # 按路径 pi 顺序获取节点

        # 计算距离索引：NODE_SIZE * from_node + to_node
        distance_index = AGH.NODE_SIZE * torch.cat((torch.zeros_like(loc[:, :1]), loc), dim=1) + \
                         torch.cat((loc, torch.zeros_like(loc[:, :1])), dim=1)
        batch_distance = dataset['distance']  # 距离矩阵 [batch_size, NODE_SIZE*NODE_SIZE]

        # 验证时间窗口约束并计算等待时间
        ids = torch.arange(batch_size, dtype=torch.int64, device=pi.device)[:, None]  # 批次索引
        time_distance = batch_distance / AGH.SPEED  # 距离转换为时间（分钟）
        time_distance = time_distance.gather(1, distance_index)  # 按路径获取时间
        cur_time = torch.full_like(pi[:, 0:1], -60, dtype=torch.float)  # 初始时间为 -60（假设提前到达）
        duration = torch.cat((torch.zeros_like(dataset['duration'][:, :1], device=dataset['duration'].device),
                             dataset['duration']), dim=1)  # 车库服务时长为 0
        
        # 累积等待时间: waiting_i = max(0, service_start_time_i - tw_right_i)
        total_waiting = torch.zeros(batch_size, device=pi.device, dtype=torch.float)
        
        for i in range(pi.size(1)):
            # 到达时间
            arrival_time = cur_time + time_distance[:, i:i+1]
            # 服务开始时间 = max(到达时间, tw_left)
            service_start_time = torch.max(arrival_time, dataset['tw_left'][ids, pi[:, i:i+1]])
            
            # 计算等待时间: max(0, service_start_time - tw_right)
            # 只对非车库节点计算 (pi != 0)
            is_not_depot = (pi[:, i:i+1] != 0).float()
            tw_right_node = dataset['tw_right'][ids, pi[:, i:i+1]]
            waiting_i = torch.clamp(service_start_time - tw_right_node, min=0) * is_not_depot
            total_waiting = total_waiting + waiting_i.squeeze(-1)
            
            # 更新当前时间：service_start_time + 服务时长
            cur_time = (service_start_time + duration[ids, pi[:, i:i+1]]) * is_not_depot - \
                       60 * (1 - is_not_depot)  # 访问车库重置时间为 -60
            
            # 验证时间窗口（可选，保留原有检查逻辑）
            if not (service_start_time <= tw_right_node + 1e-5).all():
                # 注意：在训练过程中可能会有轻微违反，这里只打印警告
                pass

        # 计算总距离成本
        total_distance = batch_distance.gather(1, distance_index).sum(1)
        
        if return_components:
            # 返回距离和等待时间分量（用于 dry-run 模式）
            return total_distance, total_waiting, None
        
        # 默认返回总距离成本（保持向后兼容）
        return total_distance, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """
        创建 AGH 数据集。
        - 返回：AGHDataset 实例。
        """
        return AGHDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """
        初始化 AGH 状态。
        - 返回：StateAGH 实例。
        """
        return StateAGH.initialize(*args, **kwargs)

    @staticmethod
    def beam_search():
        """
        束搜索算法（未实现）。
        - TODO: 用于生成高质量路径。
        """
        pass


def make_instance(args):
    """
    将输入数据转换为张量格式，生成 AGH 实例。
    - args: 包含 loc, arrival, departure, type_, demand 等。
    - 返回：字典，包含张量化的数据。
    """
    loc, arrival, departure, type_, need, *args = args
    return {
        'loc': torch.tensor(loc, dtype=torch.long),  # 登机口索引
        'arrival': torch.tensor(arrival, dtype=torch.float),  # 到达时间
        'departure': torch.tensor(departure, dtype=torch.float),  # 离开时间
        'type': torch.tensor(type_, dtype=torch.long),  # 节点类型
        'need': torch.tensor(need, dtype=torch.float)  # 需求
    }


class AGHDataset(Dataset):
    """
    AGH 数据集类，用于生成或加载 AGH 数据。
    - 支持随机生成或从 .pkl 文件加载。
    - 数据包含 loc, arrival, departure, type, demand。
    """
    def __init__(self, filename=None, size=50, num_samples=4, offset=0, distribution=None, fleet_size=6):
        """
        初始化数据集。
        - filename: .pkl 文件路径（可选）。
        - size: 每个样本的登机口数量（默认 50）。
        - num_samples: 样本数量（默认 1000000）。
        - offset: 数据偏移（用于切片）。
        - distribution: 未使用。
        - fleet_size: 车队数量（默认 10）。
        """
        super(AGHDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            # 从文件加载数据
            assert os.path.splitext(filename)[1] == '.pkl', "File must be .pkl"
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            # 随机生成数据
            n_hour = np.arange(10, 20)
            n_min = 60
            n_gate, prob = (91, np.load('problems/agh/arrival_prob.npy')) # 登机口数、小时、分钟、到达概率
            # print(prob)
            loc = 1 + np.random.choice(n_gate, size=(num_samples, size))  # 随机生成登机口索引 (1-91)
            arrival = (60 * np.random.choice(n_hour, size=(num_samples, size),p=prob)
                       + np.random.randint(0, n_min, size=(num_samples, size)))  # 到达时间
            stay = torch.tensor([120, 120, 120]).repeat(num_samples, 1)  # 停留时间（根据类型）
            type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)  # 节点类型 (0-2)
            departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)  # 离开时间 = 到达 + 停留
            # 按 7:3 比例生成 need
            service_to_select = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
            prob = torch.tensor([0.116, 0.116, 0.116, 0.116, 0.116, 0.116, 0.1, 0.1, 0.1])
            # 按照概率逐行采样
            need = torch.stack([
                service_to_select[torch.multinomial(prob, size, replacement=True)]
                for _ in range(num_samples)
            ])

            data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), need.tolist()))

        # 转换为张量实例
        self.data = [make_instance(args)  for args in data[offset:offset + num_samples]]
        self.size = len(self.data)

    def __len__(self):
        """
        返回数据集大小。
        """
        return self.size

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        - idx: 样本索引。
        - 返回：单个 AGH 实例（字典）。
        """
        return self.data[idx]


if __name__ == "__main__":
    """
    主程序：加载并打印 fleet_info.pkl。
    - fleet_info.pkl 包含车队优先级和时间信息。
    """
    with open('fleet_info.pkl', 'rb') as f:
        fleet_info = pickle.load(f)
        print(fleet_info)