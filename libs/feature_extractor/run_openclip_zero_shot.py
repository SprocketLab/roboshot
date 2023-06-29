import sys
sys.path.insert(0,  '../../')

import torch
from PIL import Image
import open_clip

from tqdm import tqdm
import numpy as np

import os
from libs.dataloader import MultiEnvDataset

import utils.const as const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    if model_name == const.CLIP_OPEN_VITL14:   
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
    elif model_name == const.CLIP_OPEN_VITB32:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    elif model_name == const.CLIP_OPEN_VITH14: 
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    elif model_name == const.CLIP_OPEN_RN50:
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
    return model, preprocess

def extract_image_features(dataset_name, model_name, batch_size=32):
    assert dataset_name in const.IMAGE_DATA
    labels_text = MultiEnvDataset().get_labels(dataset_name)
    model, preprocess = get_model(model_name)
    store_dir = f'features/{dataset_name}/{model_name}'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    filepaths = MultiEnvDataset().get_file_paths(dataset_name)
    metadata_raw = MultiEnvDataset().get_raw_metadata(dataset_name).detach().cpu().numpy()
    y_raw = MultiEnvDataset().get_raw_y(dataset_name).detach().cpu().numpy()
    
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
    metadata_all = np.vstack(metadata_all)
    y_all = np.array(y_all)
    os.makedirs(os.path.join(store_dir, str(0)), exist_ok=True)
    np.save(os.path.join(store_dir, 'image_emb.npy'), image_embeddings_all)
    np.save(os.path.join(store_dir, 'metadata.npy'), metadata_all)
    np.save(os.path.join(store_dir, 'y.npy'), y_all)
    print(f"features and metadata saved to {os.path.join(store_dir)}")



