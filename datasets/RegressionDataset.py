import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class RegressDataset(Dataset):
    def __init__(self, mode='train', data_path=None):
        if data_path and os.path.exists(data_path):
            # 从外部文件加载数据（支持CSV格式）
            df = pd.read_csv(data_path)
            self.x = torch.tensor(df[['x₁', 'x₂', 'x₃']].values, dtype=torch.float32)
            self.y = torch.tensor(df['y（标签）'].values.reshape(-1, 1), dtype=torch.float32)
        else:
            # 内置数据（与你提供的数据一致）
            # self.x = torch.tensor([
            #     [1.0, 2.0, 0.5],
            #     [2.0, 3.0, 1.0],
            #     [3.0, 4.0, 1.5],
            #     [4.0, 5.0, 2.0],
            #     [5.0, 6.0, 2.5],
            #     [6.0, 7.0, 3.0],
            #     [7.0, 8.0, 3.5],
            #     [8.0, 9.0, 4.0],
            #     [9.0, 10.0, 4.5],
            #     [10.0, 11.0, 5.0]
            # ], dtype=torch.float32)
            
            # self.y = torch.tensor([
            #     [5.1], [8.0], [10.9], [14.2], [17.0],
            #     [20.1], [22.8], [26.2], [28.0], [31.0]
            # ], dtype=torch.float32)
            self.x = torch.tensor([
                [1.0],  # 注意：保持二维结构 [样本数, 特征数=1]
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [6.0],
                [7.0],
                [8.0],
                [9.0],
                [10.0]
            ], dtype=torch.float32)
            
            self.y = torch.tensor([
                [3.1],
                [5.0],
                [7.2],
                [9.1],
                [11.0],
                [13.1],
                [15.0],
                [16.8],
                [19.2],
                [21.0]
            ], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "input_data": self.x[idx],  # 特征x1, x2, x3
            "label": self.y[idx]        # 标签y
        }
