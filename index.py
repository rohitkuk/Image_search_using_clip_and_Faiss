# This modiule is reposnible for creating embeddings and storing it an index
from faiss import write_index, IndexFlatIP
from PIL import Image
from tqdm import tqdm

import argparse
import clip
import json
import numpy as np
import os 
import torch


def index(image_dir_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device = device)
    model.eval()
    
    images = []
    image_paths = []
    
    for dir_name in sorted(os.listdir(image_dir_path)):
        print(dir_name)
        if not os.path.isdir(os.path.join(image_dir_path, dir_name)):
            continue
        for img_file in  tqdm(os.listdir(os.path.join(image_dir_path, dir_name))):
            if not img_file.endswith(".jpg"):
                continue
        print(img_file)
        image  =  Image.open(os.path.join(image_dir_path, dir_name, img_file)).convert("RGB")
        images.append(preprocess(image))
        image_paths.append(os.path.join(image_dir_path, dir_name, img_file))
    
    batch_size = 32
    image_features = []
    for i in range(0,len(images), batch_size):
        batch_images = images[i:i+batch_size]
        images_input = torch.tensor(np.stack(batch_images)).to(device)
        
        with torch.no_grad():
            batch_features = model.encode_image(images_input).float()
            batch_features /= batch_features.norm(dim = -1, keepdim = True)
            image_features.append(batch_features.cpu().numpy())
        
    image_features = np.concatenate(image_features, axis = 0)
    index = IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    write_index(index, "index_assets/index.faiss")
    with open("index_assets/image_paths.json", "w") as f:
        json.dump(image_paths, f)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_path", type=str, default="DATA/animals/animals")
    args = parser.parse_args()
    index(args.image_dir_path)

