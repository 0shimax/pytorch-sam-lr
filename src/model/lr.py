import torch
import torch.nn as nn
import torch.nn.functional as F


class LrNet(nn.Module):
    def __init__(self, in_dim, n_class=2):
        super(LrNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_class)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))
        h = F.dropout(h)        
        out = self.fc3(h)
        return out.squeeze(1)