import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import CLIPTokenizer
from model import CLIP
from loss import CLIPLoss

class CLIPDataset(Dataset):
    def __init__(self,csv_path,tokenizer,max_seq_len=77):
        self.df=pd.read_csv(csv_path)
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        image = self.transform(image)                              # (3, 224, 224)

        caption = self.df.iloc[idx]["caption"]
        tokens = self.tokenizer(
            caption, max_length=self.max_seq_len,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )
        text = tokens["input_ids"].squeeze(0)                     # (77,)

        return image, text

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
EMBED_DIM = 256
PROJECTION_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset=CLIPDataset("data.csv",tokenizer)
loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

model=CLIP(embed_dim=EMBED_DIM,projection_dim=PROJECTION_DIM).to(DEVICE)
model=torch.compile(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
loss_fn=CLIPLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss=0

    for image,text in loader:
        image,text=image.to(DEVICE),text.to(DEVICE)
        optimizer.zero_grad()
        image_vectors,text_vectors=model(image,text)
        loss=loss_fn(image_vectors,text_vectors)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "clip.pth")
    
