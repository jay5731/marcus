import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self,temperature=0.07):
        self.temperature=temperature

    def forward(self,image_vectors,text_vectors):
        image_vectors=F.normalize(image_vectors,dim=1)
        text_vectors=F.normalize(text_vectors,dim=1)

        similarity=image_vectors @ text_vectors.T/self.temperature

        labels=torch.arange(similarity.shape[0].to(similarity.device))

        loss_i2t=F.cross_entropy(similarity,labels)
        loss_t2i=F.cross_entropy(similarity.T,labels)

        return (loss_i2t+loss_t2i)/2