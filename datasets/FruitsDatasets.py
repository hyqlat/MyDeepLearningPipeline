import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image

class Fruits100Dataset(Dataset):
    def __init__(self, split='train', data_root=None, classname_path="classname.txt", transform=None):
        self.data_dir = os.path.join(data_root, split)
        assert os.path.exists(self.data_dir), f"{self.data_dir}路径不存在"
        assert os.path.exists(classname_path), f"{classname_path}路径不存在"

        with open(classname_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        
        assert len(self.class_names) == 100, "classname.txt应包含100个类别"
        self.class_to_label = {cls: idx for idx, cls in enumerate(self.class_names)}

        self.image_paths = []
        self.labels = []

        for cls_name in self.class_names:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"缺失类别文件夹{cls_dir}")

            img_names = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]
            img_names.sort(key=lambda x: int(x.split(".")[0]))

            for img_name in img_names:
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_label[cls_name])

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"读取图像{img_path}失败：{str(e)}")
        
        image_tensor = self.transform(image)

        label = torch.tensor(self.labels[item], dtype = torch.long)

        return {"input_data": image_tensor, "label": label}
    


class Fruits100TestDataset(Dataset):
    def __init__(self, data_root, classname_path="classname.txt", transform=None):
        self.test_dir = os.path.join(data_root, "test")
        assert os.path.exists(self.test_dir), f"{self.test_dir}路径不存在"
        
        # 读取classname.txt，后续用于将预测标签（0-99）映射为水果名
        with open(classname_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        assert len(self.class_names) == 100, "classname.txt应包含100个类别"

        
        self.image_info = []  #每个元素：(图像路径, 测试文件夹编号0-99, 图像名)
        for folder_idx in range(100):  #test 0-99文件夹
            folder_path = os.path.join(self.test_dir, str(folder_idx))
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"测试文件夹{folder_path}缺失")
            
            img_names = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
            img_names.sort(key=lambda x: int(x.split(".")[0]))
            
            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                self.image_info.append((img_path, folder_idx, img_name))

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path, folder_idx, img_name = self.image_info[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"读取测试图像{img_path}失败：{str(e)}")
        
        image_tensor = self.transform(image)

        return {
            "input_data": image_tensor,
            "img_path": img_path,        #图像完整路径
            "test_folder": folder_idx,   #测试集文件夹编号
            "img_name": img_name         #图像文件名
        }