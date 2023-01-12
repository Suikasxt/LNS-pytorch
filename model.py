import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, n_feature, n_label, n_hidden):
        super().__init__()
        self.linear_1 = nn.Linear(n_feature, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_label)
    
    def forward(self, features):
        hidden = self.linear_1(features)
        output = self.linear_2(hidden)
        logits = nn.Sigmoid()(output)
        return logits
        
