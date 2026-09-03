import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self,in_dim,hidden_size,out_dim):
        super().__init__()
        self.projection=nn.Sequential(
            nn.Linear(in_dim,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,out_dim)
        )

    def forward(self,x):
        return self.projection(x)