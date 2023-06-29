import sys
sys.path.insert(0,  '../')

import torch
from PIL import Image

import open_clip

from tqdm import tqdm
import numpy as np

import os
import argparse
from dataloader import MultiEnvDataset
torch.cuda.set_device(1)

import const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
# model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run CLIP zero shot')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    args = parser.parse_args()
    dataset_name = args.dataset
    assert dataset_name in const.IMAGE_DATA

    batch_size = args.batch_size
    labels_text = MultiEnvDataset().get_labels(dataset_name)

    store_dir = f'../{dataset_name}_features/features_openclip_vitL32'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    filepaths = MultiEnvDataset().get_file_paths(dataset_name)
    metadata_raw = MultiEnvDataset().get_raw_metadata(dataset_name).detach().cpu().numpy()
    y_raw = MultiEnvDataset().get_raw_y(dataset_name).detach().cpu().numpy()
    print(len(filepaths), len(metadata_raw), len(y_raw))
    image_embeddings_all = []
    y_all = []
    metadata_all = []
    for i, filename in tqdm(enumerate(filepaths)):
        try:
            image = preprocess(Image.open(filename)).unsqueeze(0)
            image_features = model.encode_image(image)
            image_embeddings_all.append(image_features.detach().cpu().numpy())
            metadata_all.append(metadata_raw[i, :])
            y_all.append(y_raw[i])
        except Exception as e:
            # raise e
            continue

    image_embeddings_all = np.vstack(image_embeddings_all)
    print('EMBEDDING SHAPE', image_embeddings_all.shape)
    metadata_all = np.vstack(metadata_all)
    y_all = np.array(y_all)
    os.makedirs(os.path.join(store_dir, str(0)), exist_ok=True)
    np.save(os.path.join(store_dir, str(0), 'image_emb.npy'), image_embeddings_all)
    np.save(os.path.join(store_dir, str(0), 'metadata.npy'), metadata_all)
    np.save(os.path.join(store_dir, str(0), 'y.npy'), y_all)
    print(f"features etc saved to {os.path.join(store_dir, str(0))}")



