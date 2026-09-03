import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,image_size=224,patch_size=16,in_channels=3,embed_dim=256):
        super().__init__()
        self.num_patches=(image_size//patch_size)**2
        self.projection=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        x=self.projection(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x

class ImageEncoder(nn.Module):
    def __init__(self,image_size=224,patch_size=16,embed_dim=256,num_heads=8,num_layers=6,dropout=0.1):
        super().__init__()
        num_patches=(image_size//patch_size)**2
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=embed_dim,n_head=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,batch_first=True
        )
        self.transformer=nn.Transformer(encoder_layer,num_layers=num_layers)
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self,x):
        B=x.shape[0]
        x=self.patch_embedding(x)
        cls=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls,x],dim=1)
        x=x+self.pos_embedding
        x=self.dropout(x)
        x=self.transformer(x)
        x=self.norm(x[:,0,:])
        return x