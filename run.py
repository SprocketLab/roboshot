import sys
sys.path.insert(0, '../')
import os
import argparse

import numpy as np
import torch

from wilds import get_dataset

from dataloader import MultiEnvDataset
from get_clip_text_emb import get_text_embedding
from chatgpt_reprompting import get_z_prompts
from openLM_reprompting import get_z_prompts_openLM
import const
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from text_prompts import text_prompts

def eval_wilds(preds, test_Y):
    if not torch.is_tensor(test_Y):
        test_Y = torch.Tensor(test_Y)
    metadata = np.load(os.path.join(load_dir, 'metadata.npy'))
    dataset = get_dataset(dataset=dataset_name, download=False, root_dir='/home/dyah')
    _, results_str = dataset.eval(preds, test_Y, torch.Tensor(metadata))
    print(results_str)
    return results_str

def eval_domainbed(y_pred, y_true, logits):
    print('ALL', len(logits))
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

def eval_synthetic(y_pred, y_true, logits):
    dataset = MultiEnvDataset().dataset_dict[dataset_name]()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    metadata = np.load(os.path.join(load_dir, 'metadata.npy'))
    print('per domain acc')
    unique_domains = np.unique(metadata)
    acc_all = []
    idx2class = {v:k for k,v in dataset.class_to_idx.items()}
    for domain in unique_domains:
        for y in np.unique(y_true):
            d_sample_idx = np.argwhere((metadata== domain) & (y_true == y))
            samples_y_pred = y_pred[d_sample_idx]
            samples_y_true = y_true[d_sample_idx]
            domain_acc = accuracy_score(samples_y_true, samples_y_pred)
            print(f'{dataset.metadata_map_reverse[domain]} y = {idx2class[y]} acc: {domain_acc:.3f}')
            acc_all.append(domain_acc)
    print(f'Average acc = {np.mean(acc_all):.3f}')
    print(f'Worst acc = {np.amin(acc_all):.3f}')
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'F1 = {f1:.3f}')
    print('\n')
    
def eval_cxr(y_pred, y_true, logits):
    print('ALL', len(logits))
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
        const.SD_CATDOG_NAME: eval_synthetic,
        const.SD_NURSE_FIREFIGHTER_NAME: eval_synthetic,
        const.CXR_NAME: eval_cxr,
        const.BREEDS17_NAME: eval_domainbed,
        const.BREEDS26_NAME: eval_domainbed,
    }
    if dataset_name not in [const.IMAGENETS_NAME, const.CXR_NAME, const.PACS_NAME, const.BREEDS17_NAME, const.BREEDS26_NAME, const.VLCS_NAME]:
        eval_func[dataset_name](preds, test_Y)
    else:
        eval_func[dataset_name](preds, test_Y, logits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run CLIP zero shot')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-clip', '--clip_model', type=str, default='openclip_vitl14')
    parser.add_argument('-lm', '--llm', stype=str, default='chatgpt')
    parset.add_argument('-reuse', '--reuse_cached_lm_output', type=bool, default=True)
    args = parser.parse_args()
    
    dataset_name = args.dataset
    clip_model = args.clip_model
    llm_model = args.llm
    reuse_lm_output = args.reuse_cached_lm_output

    assert clip_model in const.SUPPORTED_CLIP
    assert llm_model in const.SUPPORTED_LM

    labels = text_prompts[dataset_name]['labels']
    max_tokens = 100
    n_paraphrases = 0


    # z_all = get_z_prompts_openLM(dataset_name, model_name=const.GPT2_NAME)
    # differences = z_all[0]
    # spurious = z_all[1:]
    # spurious = np.unique(np.array(spurious).flatten()).flatten()
    # differences = np.unique(np.array(differences).flatten()).flatten()
    # print(spurious)
    # print(differences)
    # for i, diff in enumerate(differences):
    # # for j, p_ in enumerate(diff):
    #     p_ = differences[i]
    #     p_ = p_.strip().rstrip()
    #     differences[i] = f'this object is {p_}'
    # for i, set_ in enumerate(spurious):
    #     # for j, p_ in enumerate(set_):
    #     p_ = spurious[i]
    #     p_ = p_.strip().rstrip()
    #     for l_ in labels:
    #         p_ = p_.replace(l_, '')
    #     spurious[i]= f'this object is {p_}'
    # z_reject = []
    # visited = set()
    # for i, item1 in enumerate(spurious):
    #     for j, item2 in enumerate(spurious):
    #         if (i, j) in visited:
    #             continue
    #         if item1==item2:
    #             continue
    #         z_reject.append([f'{item1}',f'{item2}'])
    #         visited.add((i,j))
    #         visited.add((j,i))
    # z_accept = []
    # visited = set()
    # for i, item1 in enumerate(differences):
    #     for j, item2 in enumerate(differences):
    #         if (i, j) in visited:
    #             continue
    #         if item1==item2:
    #             continue
    #         z_accept.append([f'{item1}',f'{item2}'])
    #         visited.add((i,j))
    #         visited.add((j,i))
    # z_accept = np.array(z_accept)
    # z_reject = np.array(z_reject)
    # print('accept', z_accept)
    # print('reject', z_reject)
    # exit()
    # z_prompts = get_z_prompts(dataset_name, verbose=True, max_tokens=max_tokens, n_paraphrases=n_paraphrases)
    # exit()
    # exit()
    if dataset_name == const.WATERBIRDS_NAME:
        z_reject = [['a bird with aquatic habitat', 'a bird with terrestrial habitat'], ['a bird with keratin feathers physiology', 'a bird with hydrophobic feathers physiology'], ['a bird with insects diet', 'a bird with fish diet'], ['a bird with longer wingspan flight', 'a bird with shorter wingspan flight'], ['a bird with coastal migration', 'a bird with inland migration'], ['a bird that lives in watery environments', 'a bird that lives on land.'], ['a bird has feathers made of the protein', "a bird's physiology with feathers that rep"], ['a bird that eats bugs.', 'a bird that eats mainly fish.'], ['a bird with wings that span farther when', 'a bird with a smaller wingspan can'], ['a bird that migrates along coastlines', 'a bird that migrates to different areas']]
        z_accept = [['a bird with webbed feet', 'a bird with talons feet'], ['a bird with waterproof feathers', 'a bird with non-waterproof feathers'], ['a bird with larger size', 'a bird with smaller size'], ['a bird with darker color', 'a bird with lighter color'], ['a bird with longer bill', 'a bird with shorter bill'], ['a bird with wide beaks', 'a bird with narrow beaks']]
    # elif dataset_name == const.CELEBA_NAME:
    #     z_reject = [['a person with dark skin tone', 'a person with light skin tone'], ['a person with angular strong facial features', 'a person with soft round facial features'], ['a person with high perceived attractiveness', 'a person with low perceived attractiveness'], ['a person with serious personality traits', 'a person with loving personality traits'], ['a person with high intelligence', 'a person with low intelligence'], ['a person with high confidence level', 'a person with low confidence level'], ['a person with a deep complexion.', 'a person with a fair complexion.'], ['a person with sharp, prominent facial features', 'a person with a gentle, rounded face'], ['an individual with deep-seated character', 'a person who is caring and kind.'], ['a highly intelligent individual.', 'a person of limited mental capacity.'], ['a person who is sure of themselves.', 'a person who lacks self-assurance']]
    #     z_accept = [['a person with dark hair', 'a person with blond hair'],['a person with coarse hair texture', 'a person with smooth hair texture'], ['a person with lighter eye color', 'a person with darker eye color']]
    # elif dataset_name == const.PACS_NAME:
    #     spurious = [['overly loyal'], ['aggressive'], ['messy'], ['judgmental'], ['expensive'], ['difficult to play']]
    #     differences = [['this object is loyal'], ['this object has long trunk'], ['this object has long neck'], ['this object has self-awareness'], ['this object is shelter'], ['this object has strings']]
    #     spurious = np.unique(np.array(spurious).flatten()).flatten()
    #     differences = np.unique(np.array(differences).flatten()).flatten()
    #     z_reject = []
    #     visited = set()
    #     for i, item1 in enumerate(spurious):
    #         for j, item2 in enumerate(spurious):
    #             if (i, j) in visited:
    #                 continue
    #             if item1==item2:
    #                 continue
    #             z_reject.append([f'this object is {item1}',f'this object is {item2}'])
    #             visited.add((i,j))
    #             visited.add((j,i))
    #     z_accept = []
    #     visited = set()
    #     for i, item1 in enumerate(differences):
    #         for j, item2 in enumerate(differences):
    #             if (i, j) in visited:
    #                 continue
    #             if item1==item2:
    #                 continue
    #             z_accept.append([f'{item1}',f'{item2}'])
    #             visited.add((i,j))
    #             visited.add((j,i))
    #     z_accept = np.array(z_accept)
    #     z_reject = np.array(z_reject)
    # elif dataset_name == const.CXR_NAME:
    #     z_reject = [['distorted lung contour', 'normal lung contour'], ['normal lung volume', 'decreased lung volume'], ['increased lung opacity', 'normal lung opacity'], ['present mediastinal shift', 'absent mediastinal shift']]
    #     z_accept = [['consolidation opacity', 'airspace opacity'], ['increased size', 'normal size'], ['symmetrical shape', 'uneven shape'], ['smooth border', 'irregular border']]
    # elif dataset_name == const.BREEDS17_NAME:
    #     differences = [['this has moist smooth skin'], ['this has hard protective shell'], ['this has scales'], ['this has cylindrical long body'], ['this has 8 legs'], ['this has plumage'], ['this has beak'], ['this has pincers'], ['this has 4 legs'], ['this has large ears'], ['this has pointed muzzle'], ['this has whiskers'], ['this has claws'], ['this has hard exoskeleton'], ['this has wings'], ['this has long arms'], ['this has prehensile tail']]
    #     spurious = [['slimy skin'], ['hard shell'], ['scaly skin'], ['cylindrical body'], ['eight legs'], ['stocky body'], ['brightly colored feathers'], ['hard exoskeleton'], ['fur'], ['long muzzle'], ['long snout'], ['soft fur'], ['heavy body'], ['hard exoskeleton'], ['wings'], ['opposable thumbs'], ['tail']]
    #     # [['slow moving and aquatic'], ['slow moving and shelled'], ['tailed and blooded'], ['slithery and carnivorous'], ['legged and spinning'], ['flying and dwelling'], ['colorful and intelligent'], ['scuttling and shellfish'], ['sociable and loyal'], ['oriented and intelligent'], ['swift and cunning'], ['affectionate and agile'], ['strong and powerful'], ['flying and shelled'], ['delicate and colored'], ['intelligent and bipedal'], ['curious and playful']]
    #     # spurious = [['does not have lungs'], ['does not have fur'], ['cannot fly'], ['cannot climb trees'], ['does not have feathers'], ['cannot swim'], ['does not have claws'], ['does not have a tail'], ['cannot camouflage'], ['cannot hunt in packs'], ['cannot bark'], ['cannot climb trees'], ['cannot climb trees'], ['cannot fly'], ['cannot spin webs'], ['cannot bark'], ['cannot fly']]
    #     # spurious = [['this has slimy skin'], ['hard shell'], ['has scales'], ['long and slender'], ['8 legs'], ['mottled feathers'], ['colorful plumage'], ['pincers'], ['this is loyal'], ['this has pack mentality'], ['this is cunning'], ['this is independent'], ['this is powerful'], ['this has hard exoskeleton'], ['this has wings'], ['this has opposable thumbs'], ['this has prehensile tail']]
    #     spurious = np.unique(np.array(spurious).flatten()).flatten()
    #     differences = np.unique(np.array(differences).flatten()).flatten()
    #     z_reject = []
    #     visited = set()
    #     for i, item1 in enumerate(spurious):
    #         for j, item2 in enumerate(spurious):
    #             if (i, j) in visited:
    #                 continue
    #             if item1==item2:
    #                 continue
    #             z_reject.append([f'this has {item1}',f'this has {item2}'])
    #             visited.add((i,j))
    #             visited.add((j,i))
    #     z_accept = []
    #     visited = set()
    #     for i, item1 in enumerate(differences):
    #         for j, item2 in enumerate(differences):
    #             if (i, j) in visited:
    #                 continue
    #             if item1==item2:
    #                 continue
    #             z_accept.append([f'{item1}',f'{item2}'])
    #             visited.add((i,j))
    #             visited.add((j,i))
    #     z_accept = np.array(z_accept)
    #     z_reject = np.array(z_reject)
    #     z_accept = z_accept[np.random.choice(len(z_accept), 30, replace=False)]
    #     z_reject = z_reject[np.random.choice(len(z_reject), 30, replace=False)]
    # elif dataset_name == const.VLCS_NAME:
    #     differences = [['feathers', 'wings'], ['engine', 'wheels'], ['legs', 'seat'], ['tail', 'fur'], ['arms', 'legs']]
    #     spurious = [['has four wheels'], ['has feathers'], ['barks'], ['has legs'], ['flies']]
    #     # [['object with flight', 'object with feathers'], ['object with wheels', 'object with motor'], ['object with seat', 'object with legs'], ['object with fur', 'object with loyalty'], ['object with emotion', 'object with speech']]
    #     spurious = np.unique(np.array(spurious).flatten()).flatten()
    #     differences = np.unique(np.array(differences).flatten()).flatten()
    #     z_reject = []
    #     visited = set()
    #     for i, item1 in enumerate(spurious):
    #         for j, item2 in enumerate(spurious):
    #             if (i, j) in visited:
    #                 continue
    #             if item1==item2:
    #                 continue
    #             z_reject.append([f'this object {item1}',f'this object {item2}'])
    #             visited.add((i,j))
    #             visited.add((j,i))
    #     z_accept = []
    #     visited = set()
    #     for i, item1 in enumerate(differences):
    #         for j, item2 in enumerate(differences):
    #             if (i, j) in visited:
    #                 continue
    #             if item1==item2:
    #                 continue
    #             z_accept.append([f'this object has {item1}',f'this object has {item2}'])
    #             visited.add((i,j))
    #             visited.add((j,i))
    labels_text = MultiEnvDataset().dataset_dict[dataset_name]().get_labels()
    dir_dict = {
        # const.CLIP_ALIGN_NAME: f'../{dataset_name}_features/features_gt_ALIGN/0',
        # const.CLIP_BASE_NAME: f'../{dataset_name}_features/features_gt/0',
        # const.CLIP_ALT_NAME: f'../{dataset_name}_features/features_gt_alt/0',
        # const.CLIP_BIOMED_NAME: f'../{dataset_name}_features/features_gt/0',
        # const.CLIP_OPEN_VITL14: f'../{dataset_name}_features/features_openclip_vitL14/0',
        const.CLIP_OPEN_VITB32: f'../{dataset_name}_features/features_openclip_vitB32/0',
        # const.CLIP_OPEN_VITH14: f'../{dataset_name}_feature÷s/features_openclip_vitH14/0',
        # const.CLIP_OPEN_RN50: f'../{dataset_name}_features/features_openclip_rn50/0',
    }

    for key in dir_dict:
        load_dir = dir_dict[key]
        reject_emb = []
        accept_emb = []

        print(f'CLIP MODEL = {key}')
        test_X = np.load(os.path.join(load_dir, 'image_emb.npy'))
        test_Y = np.load(os.path.join(load_dir, 'y.npy'))

        label_emb = get_text_embedding(labels_text, model_name=key)
        preds, logits = make_clip_preds(test_X, label_emb)
        print("========= Baseline (ZS) =========")
        evaluate(dataset_name, preds, test_Y, logits)
        # exit()
        if dataset_name in [const.WATERBIRDS_NAME, const.CELEBA_NAME, const.PACS_NAME]:
            print("========= Baseline (Group Prompt) =========")
            group_prompt = MultiEnvDataset().dataset_dict[dataset_name]().get_group_prompts()
            group_prompt_emb = get_text_embedding(group_prompt, model_name=key)
            preds, logits = make_clip_preds(test_X, group_prompt_emb)
            if dataset_name == const.PACS_NAME:
                group_prompt_preds_multi(preds, len(group_prompt), 4, 7)
            else:
                preds = group_prompt_preds(preds)
            evaluate(dataset_name, preds, test_Y, logits)
        # exit()
        for prompt in tqdm(z_reject):
            emb = get_text_embedding(prompt, model_name=key)
            reject_emb.append(emb)
        
        for prompt in tqdm(z_accept):
            emb = get_text_embedding(prompt, model_name=key)
            accept_emb.append(emb)

        test_Y = torch.Tensor(test_Y)
        reject_emb_all = np.array(reject_emb)
        accept_emb_all = np.array(accept_emb)
        print(reject_emb_all.shape)
        print(accept_emb_all.shape)


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
        print("========= OURS W/ QR Rejection =========")
        evaluate(dataset_name, preds, test_Y, logits)

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
        print("========= OURS W/ QR Accept =========")
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
        print("========= OURS W/ BOTH =========")
        evaluate(dataset_name, preds, test_Y, logits)
   