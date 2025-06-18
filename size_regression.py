import json
# funiture_data 로드
output_file = "furniture_data.json"
with open(output_file, 'r') as f:
    furniture_data = json.load(f)

print(f"총 {len(furniture_data)}개의 유효한 가구 데이터가 추출되었습니다.")

# 결과 예시 출력
for item in furniture_data[:5]:
    print(item)

from collections import Counter

# category 개수 세기
category_counts = Counter(item["category"] for item in furniture_data)

# 출력
for category, count in category_counts.items():
    print(f"{category}: {count}")
from collections import defaultdict

# 고유 카테고리 추출 및 인덱스 매핑
category_set = sorted(set(item["category"] for item in furniture_data))
category_to_idx = {cat: i for i, cat in enumerate(category_set)}

# 클래스 인덱스 리스트 생성
class_indices = [category_to_idx[item["category"]] for item in furniture_data]
size_list = [item["size"] for item in furniture_data]

import os

latent_vectors = []
valid_class_indices = []
valid_sizes = []

import torch
saved_latent_vectors = torch.load("/home/eden/Data/JNU/AI-System/latent_vectors__.pt")
print(f"Loaded latent vectors of shape: {saved_latent_vectors.shape}")

from tqdm import tqdm

for i in range(len(saved_latent_vectors)):
    latent_vectors.append(saved_latent_vectors[i])
    valid_class_indices.append(class_indices[i])
    valid_sizes.append(size_list[i])
    
import torch.nn as nn

class FurnitureSizeRegressor(nn.Module):
    def __init__(self, latent_dim, class_count, output_dim=3):
        super().__init__()
        self.class_embed = nn.Embedding(class_count, 64)
        self.latent_embed = nn.Linear(latent_dim, 512)
        self.mlp = nn.Sequential(
            nn.Linear(512+64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, latent_vec, class_idx):
        class_vec = self.class_embed(class_idx)
        latent_vec = self.latent_embed(latent_vec)
        x = torch.cat([latent_vec, class_vec], dim=-1)
        return self.mlp(x)

from torch.utils.data import Dataset, DataLoader
import numpy as np

class FurnitureDataset(Dataset):
    def __init__(self, latent_vectors, class_indices, sizes):
        self.latents = torch.from_numpy(np.array(latent_vectors)).float()
        self.classes = torch.from_numpy(np.array(class_indices)).int()
        self.sizes = torch.from_numpy(np.array(sizes)).float()

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.classes[idx], self.sizes[idx]

# 예시
dataset = FurnitureDataset(latent_vectors, valid_class_indices, valid_sizes)

# dataset을 test와 train으로 나눔
from sklearn.model_selection import train_test_split
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
class FurnitureSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
train_dataset = FurnitureSubset(dataset, train_indices)
test_dataset = FurnitureSubset(dataset, test_indices)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def train_model(model, train_dataloader, test_dataloader, epochs=10, lr=1e-3):
    device = next(model.parameters()).device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for latents, classes, sizes in train_bar:
            latents, classes, sizes = latents.to(device), classes.to(device), sizes.to(device)

            optimizer.zero_grad()
            outputs = model(latents, classes)
            loss = criterion(outputs, sizes)
            train_bar.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        test_bar = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing", unit="batch")
        with torch.no_grad():
            total_test_loss = 0.0
            for latents, classes, sizes in test_bar:
                latents, classes, sizes = latents.to(device), classes.to(device), sizes.to(device)
                outputs = model(latents, classes)
                loss = criterion(outputs, sizes)
                total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_dataloader)
            print(f"Test Loss: {avg_test_loss:.4f}")



# 모델 초기화 및 학습
latent_dim = latent_vectors[0].shape[0]  # latent vector의 차원
class_count = len(category_set)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FurnitureSizeRegressor(latent_dim, class_count).to(device)
train_model(model, train_dataloader, test_dataloader, epochs=10, lr=1e-3)

# 모델 저장
output_model_path = "furniture_size_regressor.pth"
torch.save(model.state_dict(), output_model_path)