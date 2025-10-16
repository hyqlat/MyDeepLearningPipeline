import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import DemoModel, RegressModel
# import datasets 
from datasets import DemoDatasets, RegressDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


class DeepLearningPipeline:
    def __init__(self, opt):
        self.model = None
        self.opt = opt
        self.start_epoch = 0

    def dataPrepare(self, is_train=True):
        if is_train:
            if self.opt.dataset_type == "RegressionDataset":
                self.dataset = RegressDataset('train')
                if self.opt.has_valid:
                    self.valid_dataset = RegressDataset('valid')
                self.test_dataset = RegressDataset('test')
            else:
                self.dataset = DemoDatasets()
                if self.opt.has_valid:
                    self.valid_dataset = DemoDatasets()
                self.test_dataset = DemoDatasets()

            #训练
            self.data_loader = DataLoader(self.dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)

            #验证
            if self.opt.has_valid:
                self.valid_data_loader = DataLoader(self.valid_dataset, batch_size=self.opt.valid_batch_size, shuffle=False, num_workers=0, pin_memory=True)
            
            #测试
            self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        else:
            if self.opt.dataset_type == "RegressionDataset":
                self.test_dataset = RegressDataset('test')
            else:
                self.test_dataset = DemoDatasets()
            
            #测试
            self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def modelPrepare(self,):
        if self.opt.model_type == "RegressionModel":
            self.model = RegressModel()
        else:
            self.model = DemoModel()
        self.model.to(self.device)
    
    def optimizerPrepare(self,):
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.opt.lr)
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        
    def lossPrepare(self,):
        if self.opt.loss_type == "MSE":
            self.lossType = nn.MSELoss()
        elif self.opt.loss_type == "CE":
            self.lossType = torch.nn.CrossEntropyLoss()
        else:
            self.lossType = nn.MSELoss()
        
    def deviceDetect(self,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

    def lossCaculate(self, y_pred, y):
        return self.lossType(y_pred, y)
    
    def saveModel(self, epoch, filename=None):
        if filename is None:
            save_path = '{}/{}.pth'.format(self.opt.ckpt_dir, epoch)#'checkpoints/ +' 
        else:
            save_path = '{}/{}.pth'.format(self.opt.ckpt_dir, filename)

        checkpoint = {
            'epoch': epoch,  # 当前训练到第几轮
            'model_state_dict': self.model.state_dict(),  # 模型参数
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态
            'loss': self.train_losses[-1],  # 最新损失值
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        
        # print(f"模型已保存至: {save_path}")
        # print(f"保存信息: 轮数 {epoch}, 损失 {self.train_losses[-1]:.4f}")
    
    def loadModel(self,):
        load_path = '{}/{}.pth'.format(self.opt.ckpt_dir, self.opt.load_filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"no exists: {load_path}")
        
        if self.device is None:
            checkpoint = torch.load(load_path)
        else:
            checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0)
        
        print(f"模型已从 {load_path} 加载")
        print(f"恢复训练轮数: {start_epoch}, 最后记录的损失: {loss:.4f}" 
              if self.loss is not None else f"恢复训练轮数: {start_epoch}")
        
        self.start_epoch = start_epoch       
    
    def train(self,):
        self.train_losses = []
        # 创建tqdm进度条对象，方便后续更新描述
        pbar = tqdm(range(self.start_epoch, self.opt.total_epoch), desc="Training Epochs")
        
        for epoch in pbar:
            # train
            total_loss = self.trainOneEpoch()
            avg_loss = total_loss / len(self.data_loader)
            self.train_losses.append(avg_loss)

            # val
            valid_info = ""
            if self.opt.has_valid:
                valid_loss = self.validation()
                valid_avg_loss = valid_loss / len(self.valid_data_loader)
                valid_info = f", Valid Loss: {valid_avg_loss:.6f}"

            pbar.set_postfix_str(
                f"Epoch [{epoch+1}/{self.opt.total_epoch}], Loss: {avg_loss:.6f}{valid_info}"
            )

            # if (epoch + 1) % 10 == 0:
            #     tqdm.write(f"Epoch [{epoch+1}/{self.opt.total_epoch}], Loss: {avg_loss:.6f}{valid_info}")

            self.saveModel(epoch=epoch, filename='latest')

    def trainOneEpoch(self,):
        self.model.train()
        total_loss = 0.0
        for i, (data_dict) in enumerate(self.data_loader):
            X = data_dict["input_data"].to(self.device)
            y = data_dict["label"].to(self.device)
            y_pred = self.model(X)

            loss = self.lossCaculate(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
    
        return total_loss
        
    def validation(self,):
        self.model.eval()
        total_loss = 0.0
        for i, (data_dict) in enumerate(self.valid_data_loader):
            with torch.no_grad():
                X = data_dict["input_data"].to(self.device)
                y = data_dict["label"].to(self.device)
                y_pred = self.model(X)

                loss = self.lossCaculate(y_pred, y)
                total_loss += loss.item()
    
        return total_loss

    def test(self):
        self.model.eval()
        all_X = []
        all_y_pred = []

        with torch.no_grad():
            for data_dict in self.test_data_loader:
                X = data_dict["input_data"].to(self.device)
                y_pred = self.model(X)

                all_X.append(X.cpu())
                all_y_pred.append(y_pred.cpu())

        all_X = torch.cat(all_X, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        save_dir = self.opt.result_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "X_y_pred.pt")
        torch.save({"X": all_X, "y_pred": all_y_pred}, save_path)

        print(f"已保存至 {save_path}")
        print(f"X类型：{all_X.dtype}，y_pred类型：{all_y_pred.dtype}")
        # 后续加载方式：data = torch.load(save_path); X = data["X"]; y_pred = data["y_pred"]


    def traingPipeline(self,):
        #prepare
        self.deviceDetect()
        self.dataPrepare(is_train=True)
        self.modelPrepare()
        if self.opt.is_load:
            self.loadModel()
        self.optimizerPrepare()
        self.lossPrepare()

        #train
        self.train()

        #test
        self.test()

    def testPipeline(self,):
        #prepare
        self.deviceDetect()
        self.dataPrepare(is_train=False)
        self.modelPrepare()
        self.loadModel()

        #test
        self.test()

