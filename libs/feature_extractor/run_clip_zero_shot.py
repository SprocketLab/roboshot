import sys
sys.path.insert(0, '../../')

import torch

from transformers import CLIPProcessor, CLIPModel
from transformers import AlignProcessor, AlignModel
from transformers import BlipProcessor, BlipModel
from transformers import AutoProcessor, AutoModel, FlavaModel
from transformers import AltCLIPModel, AltCLIPProcessor

from tqdm import tqdm
import numpy as np
import os

from libs.dataloader import MultiEnvDataset
import utils.const as const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    if model_name == const.CLIP_BASE_NAME:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif model_name == const.CLIP_ALIGN_NAME:
        clip_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        clip_model = AlignModel.from_pretrained("kakaobrain/align-base")
    elif model_name == const.CLIP_ALT_NAME:
        clip_model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        clip_processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")
    return clip_model, clip_processor

def extract_image_features(dataset_name, model_name, batch_size=32):
    assert dataset_name in const.IMAGE_DATA
    labels_text = MultiEnvDataset().get_labels(dataset_name)
    clip_model, clip_processor = get_model(model_name)

    store_dir = f'features/{dataset_name}/{model_name}'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
    dataloaders = MultiEnvDataset().get_dataloaders(dataset_name, batch_size)

    for i, dataloader in enumerate(dataloaders):
        image_embeddings_all = []
        y_all = []
        metadata_all = []
        for j, labeled_batch in tqdm(enumerate(dataloader)):
            if len(labeled_batch) == 3:
                x, y, metadata = labeled_batch
                metadata = metadata.detach().cpu().numpy()
            else:
                if type(labeled_batch) != dict:
                    x, y = labeled_batch
                else:
                    x = labeled_batch['image']
                    y = labeled_batch['label']
            if torch.cuda.is_available():
                x.to(device)
            try:
                noun_entity_inputs = clip_processor(text=['aaa' for i in range(32)], images=x, return_tensors="pt", max_length=16, padding='max_length')
                clip_outputs = clip_model(**noun_entity_inputs)
                    
                img_embedding = clip_outputs.image_embeds

                image_embeddings_all.append(img_embedding.detach().cpu().numpy())
                metadata_all.append(metadata)
                y = y.detach().cpu().numpy().tolist()
                y_all.extend(y)
            except Exception as e:
                # raise e
                continue
        
        image_embeddings_all = np.vstack(image_embeddings_all)
        if len(labeled_batch) == 3:
            if len(metadata_all[0].shape) == 1:
                metadata_all = np.hstack(metadata_all)
            else:
                metadata_all = np.vstack(metadata_all)
        y_all = np.array(y_all)

        np.save(os.path.join(store_dir, 'image_emb.npy'), image_embeddings_all)
        if len(labeled_batch) == 3:
            np.save(os.path.join(store_dir, 'metadata.npy'), metadata_all)
        np.save(os.path.join(store_dir, 'y.npy'), y_all)
        print(f"features and metadata saved to {store_dir}")



