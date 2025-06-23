import os
import json
from tqdm import tqdm

# def extract_valid_furniture_data(json_dir):
#     result = []

#     bar = tqdm(os.listdir(json_dir), desc="Processing JSON files", unit="file")

#     for filename in bar:
#         if not filename.endswith(".json"):
#             continue

#         file_path = os.path.join(json_dir, filename)
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#         except Exception as e:
#             print(f"Failed to load {filename}: {e}")
#             continue

#         furniture_list = data.get("furniture", [])
#         for item in furniture_list:
#             if not item.get("valid", False):
#                 continue
#             if not all(k in item for k in ("category", "size", "jid")):
#                 continue
#             result.append({
#                 "category": item["category"],
#                 "size": item["size"],
#                 "jid": item["jid"]
#             })

#     return result

# # 사용 예시
# json_directory = "/home/eden/Documents/JNU/2025-1/AI-System/AI-System-Project/3D-FRONT"
# furniture_data = extract_valid_furniture_data(json_directory)

import pickle
# pickle로 저장
output_file = "/home/eden/Data/JNU/AI-System/furniture_data.pkl"
# with open(output_file, 'wb') as f:
#     pickle.dump(furniture_data, f)

# pickle로 저장된 데이터를 불러오기
with open(output_file, 'rb') as f:
    furniture_data = pickle.load(f)

    

# 결과 예시 출력
for item in furniture_data[:5]:
    print(item)


import torch
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.sample import sample_latents
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.image_util import load_image
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 및 설정 로드 (1회만)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
guidance_scale = 3.0

def extract_latent_from_jpg(img_paths):
    # img_load
    images = [load_image(img_path) for img_path in img_paths]
    # latent 추출
    latents = sample_latents(
        batch_size=len(images),
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=images),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    return latents

latent_vectors = []
valid_class_indices = []
valid_sizes = []

def make_img_path(jid):
    return os.path.join("/home/eden/Data/JNU/AI-System/3D-FUTURE-model", f"{jid}/image.jpg")

img_paths = [make_img_path(item["jid"]) for item in furniture_data]
batch_size = 6
batch_img_paths = [img_paths[i:i + batch_size] for i in range(0, len(img_paths), batch_size)]

batch_img_paths = batch_img_paths

for batch in tqdm(batch_img_paths, desc="Processing batches", unit="batch"):
    latent = extract_latent_from_jpg(batch)
    latent_vectors.append(latent.cpu())
    # clear cuda memory
    torch.cuda.empty_cache()

latent_vectors = torch.cat(latent_vectors, dim=0)

# # latent_vector 저장
# output_path = "/home/eden/Data/JNU/AI-System/latent_vectors_.pt"
# torch.save(latent_vectors, output_path)

# h5py로 저장
import h5py
output_path = "/home/eden/Data/JNU/AI-System/latent_vectors.h5"
with h5py.File(output_path, 'w') as f:
    f.create_dataset('latent_vectors', data=latent_vectors.numpy())
