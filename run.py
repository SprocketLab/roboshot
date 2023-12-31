import sys
sys.path.insert(0, '../')
import os
import argparse

import numpy as np
import torch

from wilds import get_dataset

from libs.dataloader import MultiEnvDataset
from libs.get_clip_text_emb import get_text_embedding
from libs.chatgpt_reprompting import get_z_prompts
from libs.openLM_reprompting import get_z_prompts_openLM
from libs.text_prompts import text_prompts
from libs.cached_concepts import get_cached_concept
from libs.feature_extractor import hf_extractor, openclip_extractor
import utils.const as const

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

def eval_wilds(preds, test_Y):
    if not torch.is_tensor(test_Y):
        test_Y = torch.Tensor(test_Y)
    metadata = np.load(os.path.join(load_dir, 'metadata.npy'))
    dataset = get_dataset(dataset=dataset_name, download=False, root_dir='/home/dyah')
    _, results_str = dataset.eval(preds, test_Y, torch.Tensor(metadata))
    print(results_str)
    return results_str

def eval_domainbed(y_pred, y_true, logits):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    metadata = np.load(os.path.join(load_dir, 'metadata.npy'))
    if len(metadata.shape) > 1:
        metadata = metadata.flatten()
    unique_domains = np.unique(metadata)
    acc_all = []
    for domain in unique_domains:
        for y in np.unique(y_true):
            d_sample_idx = np.argwhere((metadata== domain) & (y_true==y))
            if len(d_sample_idx) == 0:
                continue
            samples_y_pred = y_pred[d_sample_idx]
            samples_y_true = y_true[d_sample_idx]
            domain_acc = accuracy_score(samples_y_true, samples_y_pred)
            acc_all.append(domain_acc)
            print(domain, y, len(d_sample_idx))
    acc_all = np.array(acc_all)
    print(f"AVG acc = {np.mean(acc_all):.3f}")
    print(f"WORST group acc = {np.amin(acc_all):.3f}")
    print('\n')
    
def eval_cxr(y_pred, y_true, logits):
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    acc_all = []
    for y in np.unique(y_true):
        class_sample_idx = np.argwhere(y_true==y)
        group_acc = accuracy_score(y_true[class_sample_idx], y_pred[class_sample_idx])
        acc_all.append(group_acc)
        print(y, len(class_sample_idx))
    acc_all = np.array(acc_all)
    print(f'avg acc = {np.mean(acc_all):.3f}')
    print(f'wg acc = {np.amin(acc_all):.3f}')
    print('\n')

def make_clip_preds(image_features, text_features):
    if not torch.is_tensor(image_features):
        image_features = torch.Tensor(image_features)
    if not torch.is_tensor(text_features):
        text_features = torch.Tensor(text_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return torch.argmax(text_probs, dim=1), text_probs

def group_prompt_preds(raw_preds):
    raw_preds = raw_preds.detach().cpu().numpy()
    raw_preds[np.argwhere((raw_preds == 0) | (raw_preds == 1)).flatten()] = 0
    raw_preds[np.argwhere((raw_preds == 2) | (raw_preds == 3)).flatten()] = 1
    return torch.Tensor(raw_preds)

def group_prompt_preds_multi(raw_preds, n_full_prompt, n_prompt_per_class, n_class):
    c_idx = 0
    for p_idx in range(0, n_full_prompt, n_prompt_per_class):
        idxs = []
        for cp_idx in range(n_prompt_per_class):
            idxs.extend(np.argwhere(raw_preds == p_idx + cp_idx).flatten())
        raw_preds[idxs] = c_idx
        c_idx +=1
    return torch.Tensor(raw_preds)


def evaluate(dataset_name, preds, test_Y, logits):
    eval_func = {
        const.WATERBIRDS_NAME: eval_wilds,
        const.CELEBA_NAME: eval_wilds,
        const.PACS_NAME: eval_domainbed,
        const.VLCS_NAME: eval_domainbed,
        const.CXR_NAME: eval_cxr,
    }
    if dataset_name not in [const.CXR_NAME, const.PACS_NAME, const.VLCS_NAME]:
        eval_func[dataset_name](preds, test_Y)
    else:
        eval_func[dataset_name](preds, test_Y, logits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run CLIP zero shot')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-clip', '--clip_model', type=str, default='openclip_vitl14')
    parser.add_argument('-lm', '--llm', type=str, default='chatgpt')
    parser.add_argument('-reuse', '--reuse', action='store_true')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    clip_model = args.clip_model
    llm_model = args.llm
    reuse_cached_concepts = args.reuse

    assert clip_model in const.SUPPORTED_CLIP
    assert llm_model in const.SUPPORTED_LM

    labels = text_prompts[dataset_name]['labels']
    max_tokens = 100
    n_paraphrases = 0
    if reuse_cached_concepts == True:
        z_reject, z_accept = get_cached_concept(dataset_name)
    else:
        if llm_model == const.CHATGPT_NAME:
            z_reject = get_z_prompts(dataset_name, text_prompts[dataset_name]['question_reject'])
            z_accept = get_z_prompts(dataset_name, text_prompts[dataset_name]['question_accept'])
        elif llm_model == const.LLAMA_NAME:
            z_reject = get_z_prompts_openLM(dataset_name, const.LLAMA_NAME, text_prompts[dataset_name]['question_llama_reject'])
            z_accept = []
        elif llm_model == const.GPT2_NAME:
            z_reject = get_z_prompts_openLM(dataset_name, const.GPT2_NAME, text_prompts[dataset_name]['question_openLM_reject'])
            z_accept = get_z_prompts_openLM(dataset_name, const.GPT2_NAME, text_prompts[dataset_name]['question_openLM_accept'])
        elif llm_model == const.T5_NAME:
            z_reject = get_z_prompts_openLM(dataset_name, const.T5_NAME, text_prompts[dataset_name]['question_openLM_reject'])
            z_accept = get_z_prompts_openLM(dataset_name, const.T5_NAME, text_prompts[dataset_name]['question_openLM_accept'])
    print('z reject', z_reject)
    print('z accept', z_accept)
    labels_text = MultiEnvDataset().dataset_dict[dataset_name]().get_labels()
    load_dir = f'features/{dataset_name}/{clip_model}'

    cached_features = False
    if os.path.isdir(load_dir):
        if len(os.listdir(load_dir)):
            cached_features = True

    if not cached_features:
        print(f"extracting CLIP features...")
        if clip_model in const.HF_CLIP:
            hf_extractor(dataset_name, clip_model)
        elif clip_model in const.OPEN_CLIP:
            openclip_extractor(dataset_name, clip_model)
    
    print(f'CLIP MODEL = {clip_model}')
    test_X = np.load(os.path.join(load_dir, 'image_emb.npy'))
    test_Y = np.load(os.path.join(load_dir, 'y.npy'))

    label_emb = get_text_embedding(labels_text, model_name=clip_model)
    preds, logits = make_clip_preds(test_X, label_emb)
    print("========= Baseline (ZS) =========")
    evaluate(dataset_name, preds, test_Y, logits)
    
    if dataset_name in [const.WATERBIRDS_NAME, const.CELEBA_NAME, const.PACS_NAME]:
        print("========= Baseline (Group Prompt) =========")
        group_prompt = MultiEnvDataset().dataset_dict[dataset_name]().get_group_prompts()
        group_prompt_emb = get_text_embedding(group_prompt, model_name=clip_model)
        preds, logits = make_clip_preds(test_X, group_prompt_emb)
        if dataset_name == const.PACS_NAME:
            group_prompt_preds_multi(preds, len(group_prompt), 4, 7)
        else:
            preds = group_prompt_preds(preds)
        evaluate(dataset_name, preds, test_Y, logits)

    reject_emb = []
    accept_emb = []
    for prompt in tqdm(z_reject):
        emb = get_text_embedding(prompt, model_name=clip_model)
        reject_emb.append(emb)
    
    for prompt in tqdm(z_accept):
        emb = get_text_embedding(prompt, model_name=clip_model)
        accept_emb.append(emb)

    test_Y = torch.Tensor(test_Y)
    reject_emb_all = np.array(reject_emb)
    accept_emb_all = np.array(accept_emb)

    # --------- Rejecting all spurious directions ------------
    spurious_vectors = reject_emb_all[:, 0, :] - reject_emb_all[:, 1, :]
    q_spurious, r = np.linalg.qr(spurious_vectors.T)
    q_spurious = q_spurious.T
    
    # Transform X so that so that it is orthogonal to all spurious directions
    test_proj = np.copy(test_X)
    test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)

    # Reject projections to those orthonormal vectors
    for orthonormal_vector in q_spurious:
        cos = np.squeeze(cosine_similarity(test_proj, orthonormal_vector.reshape(1, -1)))
        rejection_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), cos.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
        test_proj = test_proj - rejection_features
        test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)
    
    test_proj = torch.Tensor(test_proj)
    label_emb = torch.Tensor(label_emb)
    preds, logits = make_clip_preds(test_proj, label_emb)
    print("========= ROBOSHOT W/ QR Rejection =========")
    evaluate(dataset_name, preds, test_Y, logits)

    if len(accept_emb) == 0:
        exit()

    # --------- Accepting all true directions ------------
    true_vectors = accept_emb_all[:, 0, :] - accept_emb_all[:, 1, :]
    q_true, r = np.linalg.qr(true_vectors.T)
    q_true = q_true.T
    
    # Transform X so that so that it is orthogonal to all spurious directions
    test_proj = np.copy(test_X)
    test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)

    # Reject projections to those orthonormal vectors
    for orthonormal_vector in q_true:
        cos = np.squeeze(cosine_similarity(test_proj, orthonormal_vector.reshape(1, -1)))
        rejection_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), cos.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
        test_proj = test_proj + rejection_features
        test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)
    
    test_proj = torch.Tensor(test_proj)
    label_emb = torch.Tensor(label_emb)
    preds, logits = make_clip_preds(test_proj, label_emb)
    print("========= ROBOSHOT W/ QR Accept =========")
    evaluate(dataset_name, preds, test_Y, logits)

    # --------- COMBINED ------------
    test_proj = np.copy(test_X)
    test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)

    for orthonormal_vector in q_spurious:
        cos = np.squeeze(cosine_similarity(test_proj, orthonormal_vector.reshape(1, -1)))
        rejection_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), cos.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
        test_proj = test_proj - rejection_features
        test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)
    
    for orthonormal_vector in q_true:
        cos = np.squeeze(cosine_similarity(test_proj, orthonormal_vector.reshape(1, -1)))
        rejection_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), cos.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
        test_proj = test_proj + rejection_features
        test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)

    test_proj = torch.Tensor(test_proj)
    label_emb = torch.Tensor(label_emb)
    preds, logits = make_clip_preds(test_proj, label_emb)
    print("========= ROBOSHOT W/ BOTH =========")
    evaluate(dataset_name, preds, test_Y, logits)
