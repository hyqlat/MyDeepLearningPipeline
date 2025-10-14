import numpy as np
import torch
import torch.nn as nn

class RegressModel(nn.Module):
    def __init__(self,):
        super(RegressModel, self).__init__()

        self.layer = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.layer(x)