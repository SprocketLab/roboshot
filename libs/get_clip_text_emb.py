import sys
sys.path.insert(0, '../../')

import torch
import open_clip

from transformers import CLIPProcessor, CLIPModel
from transformers import AlignModel, AlignProcessor
from transformers import AltCLIPModel, AltCLIPProcessor

from tqdm import tqdm
import numpy as np

from libs.dataloader import MultiEnvDataset
import utils.const as const

from transformers.utils import logging
logging.set_verbosity(40)

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP image - text model

SUPPORTED_MODELS = [const.CLIP_BASE_NAME, const.CLIP_ALIGN_NAME, const.CLIP_BIOMED_NAME, const.CLIP_ALT_NAME, const.CLIP_OPEN_VITL14,\
                    const.CLIP_OPEN_VITB32, const.CLIP_OPEN_VITH14, const.CLIP_OPEN_RN50]

def get_models(model_name):
    if model_name == const.CLIP_BASE_NAME:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return clip_processor, clip_model
    elif model_name == const.CLIP_ALIGN_NAME:
        clip_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        clip_model = AlignModel.from_pretrained("kakaobrain/align-base")
        return clip_processor, clip_model
    elif model_name == const.CLIP_BIOMED_NAME:
        clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        return tokenizer, clip_model
    elif model_name == const.CLIP_ALT_NAME:
        clip_model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        clip_processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")
        return clip_processor, clip_model
    elif model_name == const.CLIP_OPEN_VITL14:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        return tokenizer, clip_model
    elif model_name == const.CLIP_OPEN_VITB32:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        return tokenizer, model
    elif model_name == const.CLIP_OPEN_VITH14:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        return tokenizer, model
    elif model_name == const.CLIP_OPEN_RN50:
        model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('RN50')
        return tokenizer, model
    
def get_text_embedding(prompts, model_name='clip'):
    assert model_name in SUPPORTED_MODELS
    clip_processor, clip_model = get_models(model_name)
    text_emb_all = []
    dataloaders = MultiEnvDataset().get_dataloaders('waterbirds', batch_size=1)
    dataloader = dataloaders[0]
    for labeled_batch in dataloader:
        x = labeled_batch[0]
        break
    with torch.no_grad():
        for i, item in enumerate(prompts):
            if model_name  in [const.CLIP_BIOMED_NAME, const.CLIP_OPEN_VITB32, const.CLIP_OPEN_VITL14, \
                               const.CLIP_OPEN_VITH14, const.CLIP_OPEN_RN50] :
                # context_length = 16
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_embedding = clip_model.encode_text(clip_processor([item])).detach().cpu().numpy()
            else:
                noun_entity_inputs = clip_processor(text=item, images=x, return_tensors="pt", max_length=16, padding='max_length')
                clip_outputs = clip_model(**noun_entity_inputs)
                text_embedding = clip_outputs.text_embeds.detach().cpu().numpy()
            text_emb_all.append(text_embedding)
        text_emb_all = np.vstack(text_emb_all)
    return text_emb_all.squeeze()