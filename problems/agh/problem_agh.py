import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from problems.agh.state_agh import StateAGH


class AGH(object):
    NAME = 'agh'

    SPEED = 80.0
    NODE_SIZE = 101  # 100 个患者位置 + 1 个 depot

    @staticmethod
    def get_costs(dataset, pi):
        """
        计算路径的双目标成本：f1（总行驶距离）和 f2（总等待时间）。
        返回：((f1, f2), mask=None)
        """
        batch_size, graph_size = dataset['duration'].size()
        graph_size = graph_size - 1

        loc = torch.cat((torch.zeros_like(dataset['loc'][:, :1]), dataset['loc']), dim=1)
        loc = loc.gather(1, pi)

        distance_index = AGH.NODE_SIZE * torch.cat((torch.zeros_like(loc[:, :1]), loc), dim=1) + \
                         torch.cat((loc, torch.zeros_like(loc[:, :1])), dim=1)
        batch_distance = dataset['distance']

        ids = torch.arange(batch_size, dtype=torch.int64, device=pi.device)[:, None]
        time_distance = batch_distance / AGH.SPEED
        time_distance = time_distance.gather(1, distance_index)
        cur_time = torch.full_like(pi[:, 0:1], -60, dtype=torch.float)
        duration = torch.cat((torch.zeros_like(dataset['duration'][:, :1], device=dataset['duration'].device),
                             dataset['duration']), dim=1)

        total_wait = torch.zeros(batch_size, device=pi.device)

        for i in range(pi.size(1)):
            arrival_time = cur_time + time_distance[:, i:i+1]
            tw_left_i = dataset['tw_left'][ids, pi[:, i:i+1]]
            wait_i = torch.clamp(arrival_time - tw_left_i, min=0) * (pi[:, i:i+1] != 0).float()
            total_wait += wait_i.squeeze(1)

            cur_time = (torch.max(arrival_time, tw_left_i)
                        + duration[ids, pi[:, i:i+1]]) * (pi[:, i:i+1] != 0).float() - \
                       60 * (pi[:, i:i+1] == 0).float()

        f1 = batch_distance.gather(1, distance_index).sum(1)
        f2 = total_wait

        return (f1, f2), None

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
    def __init__(self, filename=None, size=50, num_samples=4, offset=0, distribution=None, fleet_size=6):
        super(AGHDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', "File must be .pkl"
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            n_hour = np.arange(10, 20)
            n_min = 60
            n_gate = 100
            prob = np.load('problems/agh/arrival_prob.npy')
            loc = 1 + np.random.choice(n_gate, size=(num_samples, size))
            arrival = (60 * np.random.choice(n_hour, size=(num_samples, size), p=prob)
                       + np.random.randint(0, n_min, size=(num_samples, size)))
            stay = torch.tensor([120, 120, 120]).repeat(num_samples, 1)
            type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)
            departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)
            service_to_select = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
            prob_need = torch.tensor([0.116, 0.116, 0.116, 0.116, 0.116, 0.116, 0.1, 0.1, 0.1])
            need = torch.stack([
                service_to_select[torch.multinomial(prob_need, size, replacement=True)]
                for _ in range(num_samples)
            ])

            data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), need.tolist()))

        self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
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