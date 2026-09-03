import torch
import torch.nn as nn
class TextEncoder(nn.Module):
    def __init__(self,vocab_size=49408,max_seq_len=77,embed_dim=256,num_heads=8,num_layers=8,dropout=0.1):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,embed_dim)
        self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding=nn.Parameter(torch.randn(1,max_seq_len+1,embed_dim))
        self.dropout=nn.Dropout(dropout)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=embed_dim,nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,batch_first=True
        )
        self.transformer=nn.TransformerEncoder(encoder_layer,num_heads)
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self,x):
        B=x.shape[0]
        x = self.token_embedding(x)  
        cls=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls,x],dim=1)      # (B, 77, 256)
        x = x + self.pos_embedding         # (B, 77, 256)
        x = self.dropout(x)
        x = self.transformer(x)            # (B, 77, 256)
        x = self.norm(x[:, 0, :]) 
        return x