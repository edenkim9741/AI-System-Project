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

# 클래스 인덱스 원-핫 인코딩
import numpy as np
one_hot_indices = np.zeros((len(class_indices), len(category_set)), dtype=int)
for i, idx in enumerate(class_indices):
    one_hot_indices[i, idx] = 1
# 사이즈 리스트 추출
size_list = [item["size"] for item in furniture_data]

# print(one_hot_indices[:5])

import numpy as np
size_np = np.array(size_list, dtype=np.float32)
print("max size:", size_np.max())
print("min size:", size_np.min())

normalized_size = (size_np - size_np.min()) / (size_np.max() - size_np.min())
# 정규화된 사이즈를 리스트로 변환
normalized_size_list = normalized_size.tolist()


import os

latent_vectors = []
valid_class_indices = []
valid_sizes = []

def make_img_path(jid):
    return os.path.join("/home/eden/Data/JNU/AI-System/3D-FUTURE-model", f"{jid}/image.jpg")



img_paths = [make_img_path(item["jid"]) for item in furniture_data]
batch_size = 4
batch_img_paths = [img_paths[i:i + batch_size] for i in range(0, len(img_paths), batch_size)]

import h5py 
latent_vectors_file = "latent_vectors.h5"

with h5py.File(latent_vectors_file, 'r') as f:
    temp_latent_vectors = f['latent_vectors'][:]

from tqdm import tqdm

for i in range(len(temp_latent_vectors)):
    latent_vectors.append(temp_latent_vectors[i])
    valid_class_indices.append(one_hot_indices[i])
    valid_sizes.append(size_list[i])


import torch.nn as nn

class FurnitureSizeRegressor(nn.Module):
    def __init__(self, latent_dim, class_count, hidden_dims=[256, 64], output_dim=3):
        super(FurnitureSizeRegressor, self).__init__()

        self.latent_extractor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
        )

        self.size_estimate_layer = nn.Sequential(
            nn.Linear(hidden_dims[1]+class_count, output_dim),
            nn.Sigmoid()  # 0~1 범위로 정규화된 size 예측
        )

    def forward(self, latent_vec, class_onehot):
        # x = torch.cat([latent_vec, class_onehot], dim=1)
        latent_ext = self.latent_extractor(latent_vec)
        x = torch.cat([latent_ext, class_onehot], dim=1)
        return self.size_estimate_layer(x)


from torch.cuda.amp import autocast, GradScaler

def train_model(model, train_dataloader, test_dataloader, epochs=10, lr=1e-3):
    device = next(model.parameters()).device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for latents, classes, sizes in train_bar:
            latents, classes, sizes = latents.to(device), classes.to(device), sizes.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(latents, classes)
                loss = criterion(outputs, sizes)
            train_bar.set_postfix(loss=loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            total_loss += loss.item() * latents.size(0)
            wandb.log({"train_loss": loss.item()})


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        test_bar = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing", unit="batch")
        with torch.no_grad():
            total_test_loss = 0.0
            for latents, classes, sizes in test_bar:
                latents, classes, sizes = latents.to(device), classes.to(device), sizes.to(device)
                with autocast():
                    outputs = model(latents, classes)
                    loss = criterion(outputs, sizes)
                total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_dataloader)
            wandb.log({"test_loss": avg_test_loss})
            print(f"Test Loss: {avg_test_loss:.4f}")




latent_dim = latent_vectors[0].shape[0]  # latent vector의 차원
class_count = len(category_set)

print(f"Latent dimension: {latent_dim}, Class count: {class_count}")

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import wandb

from sklearn.model_selection import KFold

class FurnitureDataset(Dataset):
    def __init__(self, latent_vectors, class_indices, sizes):
        self.latents = torch.from_numpy(np.array(latent_vectors)).float()
        self.classes = torch.from_numpy(np.array(class_indices)).int()
        self.sizes = torch.from_numpy(np.array(normalized_size)).float()

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.classes[idx], self.sizes[idx]

# 예시
dataset = FurnitureDataset(latent_vectors, valid_class_indices, valid_sizes)

# KFold 설정
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\nFold {fold+1}/{k}")
    print(f"Train indices: {len(train_idx)}, Val indices: {len(val_idx)}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False)
    wandb.init(project="ai-system", name=f"furniture_norm_size_regression_fold_{fold+1}", config={
        "latent_dim": latent_dim,
        "class_count": class_count,
        "batch_size": 16,
        "epochs": 10,
        "learning_rate": 1e-3,
        "fold": fold + 1
        })
    # 모델 초기화 및 학습
    latent_dim = latent_vectors[0].shape[0]  # latent vector의 차원
    class_count = len(category_set)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FurnitureSizeRegressor(latent_dim, class_count).to(device)
    train_model(model, train_dataloader, val_dataloader, epochs=10, lr=1e-3)
    wandb.finish()

    # # 모델 저장
    # output_model_path = "furniture_size_regressor.pth"
    # torch.save(model.state_dict(), output_model_path


# 전체 데이터셋에 대해 학습한 모델 저장
from sklearn.model_selection import train_test_split
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_subset, batch_size=16, shuffle=False)

latent_dim = latent_vectors[0].shape[0]  # latent vector의 차원
class_count = len(category_set)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FurnitureSizeRegressor(latent_dim, class_count).to(device)
train_model(model, train_dataloader, test_dataloader, epochs=10, lr=1e-3)

output_model_path = "furniture_size_regressor.pth"
torch.save(model.state_dict(), output_model_path)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드

loaded_model = FurnitureSizeRegressor(1048576, 8)
loaded_model.load_state_dict(torch.load(output_model_path, map_location=device))
loaded_model.eval()

# 테스트할 가구 이미지 시각화
furniture_data_sample = furniture_data[:5]
hundred_furniture_data_sample = furniture_data[:100]
import matplotlib.pyplot as plt
def visualize_furniture_images(img_paths, titles=None):
    plt.figure(figsize=(16, 16))
    for i, img_path in enumerate(img_paths):
        img = plt.imread(img_path)
        plt.subplot(4, 5, i + 1)
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("furniture_images_sample.pdf")
    plt.show()

sample_paths = [make_img_path(item['jid']) for item in furniture_data_sample]
hundred_sample_paths = [make_img_path(item['jid']) for item in hundred_furniture_data_sample]

visualize_furniture_images(sample_paths, titles=[f"Category: {item['category']} \n Size: {[round(s,3) for s in item['size']]}" for item in furniture_data_sample])

