import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from projection import ProjectionHead

class CLIP(nn.Module):
    def __init__(self,embed_dim=256,projection_dim=128):
        super().__init__()
        self.image_encoder=ImageEncoder(embed_dim=embed_dim)
        self.text_encoder=TextEncoder(embed_dim=embed_dim)
        self.image_projection=ProjectionHead(embed_dim,embed_dim*2,projection_dim)
        self.text_projection=ProjectionHead(embed_dim,embed_dim*2,projection_dim)

    def forward(self,image,text):
        image_features=self.image_encoder(image)
        text_features=self.text_encoder(text)

        image_vectors=self.image_projection(image_features)
        text_vectors=self.text_projection(text_features)

        return image_vectors,text_vectors