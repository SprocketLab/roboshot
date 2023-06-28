import sys
sys.path.insert(0, '../')
from sys_const import DATA_DIR

from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
import torchvision.transforms as transforms

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

import os

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import comnivore.const as const

from datasets import load_dataset

import pandas as pd

class WILDSDataset:
    def get_daloaders(self, dataset_name, batch_size, transform, return_test=True):
        root_dir=DATA_DIR
        
        dataset = get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)
        test_dataset = dataset.get_subset("test", transform=transform)
        test_dataloader = get_eval_loader("standard", test_dataset, batch_size=batch_size)
        if return_test:
            return [test_dataloader]
        
        train_dataset = dataset.get_subset("train", transform=transform)
        train_dataloader = get_train_loader("standard", train_dataset, batch_size=batch_size)
        val_dataset = dataset.get_subset("val", transform=transform)
        val_dataloader = get_eval_loader("standard", val_dataset, batch_size=batch_size)
        return [train_dataloader, test_dataloader, val_dataloader]

    def get_file_paths(self, dataset_name, split='test'):
        root_dir=DATA_DIR
        dataset = get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)

        test_dataset = dataset.get_subset("test", transform=None)
        if split=='test':
            test_idxs = np.argwhere(np.array(dataset.split_array)==2).flatten()
        else:
            test_idxs = np.argwhere(np.array(dataset.split_array)==0).flatten()
        data_dir = test_dataset.dataset.data_dir
        if dataset_name == const.WATERBIRDS_NAME:
            file_paths = np.array([os.path.join(data_dir, f_) for f_ in test_dataset.dataset._input_array])
        else:
            file_paths = np.array([os.path.join(data_dir, 'img_align_celeba',f_) for f_ in test_dataset.dataset._input_array])
        file_paths = file_paths[test_idxs].flatten()
        return file_paths

    def get_raw_metadata(self, dataset_name, split='test'):
        root_dir=DATA_DIR
        dataset = get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)
        if split=='test':
            test_dataset = dataset.get_subset("test", transform=None)
        else:
            test_dataset = dataset.get_subset("train", transform=None)
        return test_dataset.metadata_array
    
    def get_raw_y(self, dataset_name, split='test'):
        root_dir=DATA_DIR
        dataset = get_dataset(dataset=dataset_name, download=True, root_dir=root_dir)
        if split=='test':
            test_dataset = dataset.get_subset("test", transform=None)
        else:
            test_dataset = dataset.get_subset("train", transform=None)
        return test_dataset.y_array

    
class WaterbirdsDataset:
    def get_dataloaders(self, batch_size, transform, return_test=True):
        return WILDSDataset().get_daloaders(const.WATERBIRDS_NAME, batch_size, transform, return_test)
    
    def get_labels(self,):
        self.labels = ['a landbird', 'a waterbird']
        return self.labels

    def get_file_paths(self, split):
        return WILDSDataset().get_file_paths(const.WATERBIRDS_NAME, split)
    
    def get_raw_metadata(self, split):
        return WILDSDataset().get_raw_metadata(const.WATERBIRDS_NAME, split)
    
    def get_raw_y(self, split):
        return WILDSDataset().get_raw_y(const.WATERBIRDS_NAME, split)
    
    def get_group_prompts(self):
        return ["landbird on land background", "landbird on water background", "waterbird on land background", "landbird on water background"]
    
class CelebADataset:
    def get_dataloaders(self, batch_size, transform, return_test=True):
        return WILDSDataset().get_daloaders(const.CELEBA_NAME, batch_size, transform, return_test)
    
    def get_labels(self,):
        self.labels = ['person with dark hair', 'person with blond hair']
        return self.labels

    def get_file_paths(self, split):
        return WILDSDataset().get_file_paths(const.CELEBA_NAME, split)
    
    def get_raw_metadata(self, split):
        return WILDSDataset().get_raw_metadata(const.CELEBA_NAME, split)
    
    def get_raw_y(self, split):
        return WILDSDataset().get_raw_y(const.CELEBA_NAME, split)
    
    def get_group_prompts(self):
        return ["female with dark hair", "male with dark hair", "female with blond hair", "male with blond hair"]

class CivilCommentsDataset:
    def get_dataloaders(self, batch_size, transform=None, return_test=True):
        return WILDSDataset().get_daloaders(const.CIVILCOMMENTS_NAME, batch_size, None, return_test)

class AmazonDataset:
    def get_dataloaders(self, batch_size, transform=None, return_test=True):
        return WILDSDataset().get_daloaders(const.AMAZON_NAME, batch_size, None, return_test)

class GenderBiasNLPDataset(Dataset):
    def __init__(self):
        self.texts, self.labels, self.metadata = self.process_dataset()
    
    def process_dataset(self,):
        dataset = load_dataset("md_gender_bias", "funpedia", split='train')
        texts = np.array([obj['text'] for obj in dataset])
        labels = np.array([obj['gender'] for obj in dataset])
        persona = np.array([obj['persona'].lower() for obj in dataset])
        filtered_idxs = np.argwhere(labels>0).flatten()
        texts = texts[filtered_idxs]
        labels = labels[filtered_idxs]
        persona = persona[filtered_idxs]
        labels = labels-1
        persona_unique = np.unique(persona)
        persona_id = [np.argwhere(persona_unique == p).flatten()[0] for p in persona]
        # print(len(labels), len(persona_unique))
        return texts, labels, persona_id
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.labels[idx]
        meta = self.metadata[idx]
        return text, y, meta
    
    def __len__(self):
        return len(self.labels)

class GenderBiasDataset:
    def get_dataloaders(self, batch_size, transform=None, return_test=True):
        dataset = GenderBiasNLPDataset()
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]

class CSVDataset(Dataset):
    def __init__(self):
        csv_path = f'{DATA_DIR}/hateXplain/hateXplain_test.csv'
        self.df = pd.read_csv(csv_path)
        self.texts = self.df['text']
        self.metadata = self.df[['Hindu','Islam','Minority','Refugee','Indian','Caucasian','Hispanic','Women','Disability','Homosexual','Arab','Christian','Jewish','Men','African','Nonreligious','Asian','Indigenous','Heterosexual','Buddhism','Bisexual','Asexual']].to_numpy()
        self.label = self.df['label'].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.label[idx]
        metadata = self.metadata[idx, :]
        return text, y, metadata

class HateXplainDataset:
    def get_dataloaders(self, batch_size, transform=None, return_test=True):
        dataset = CSVDataset()
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]

class FolderDataset(Dataset):
    def __init__(self, root_dir, transform, class2idx, metadata_map):
        self.root_dir = root_dir
        subdirs = [os.path.join(root_dir, subdir) for subdir in os.listdir(root_dir)]
        self.image_paths = []
        for subdir in subdirs:
            self.image_paths.extend([os.path.join(subdir, img_p) for img_p in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, img_p))])
        self.transform = transform
        self.class2idx = class2idx
        self.metadata_map = metadata_map
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath).convert('RGB')
        image = self.transform(image)
        label = self.class2idx[image_filepath.split('/')[5]]
        domain = self.metadata_map[image_filepath.split('/')[4]]
        return image, label, domain

class BreedsDataset(Dataset):
    def get_labels(self,):
        self.labels = [f'this is a {c_}' for c_ in list(self.class_to_idx.keys())]
        return self.labels

    def get_dataloaders(self, batch_size, transform, return_test=True):
        dataset_all = []
        for env in self.envs:
            dataset = FolderDataset(os.path.join(self.root_dir, env), transform=transform, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            dataset_all.append(dataset)
        dataset = ConcatDataset(dataset_all)
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]
    
    def get_file_paths(self, split='test'):
        file_paths_all = []
        for env in self.envs:
            dataset = FolderDataset(os.path.join(self.root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            file_paths_all.extend(dataset.image_paths)
        return file_paths_all
    
    def get_raw_metadata(self, split='test'):
        metadata_all = []
        for env in self.envs:
            dataset = FolderDataset(os.path.join(self.root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            metadata_all.extend([self.metadata_map[env] for i in range(len(dataset))])
        return torch.Tensor(metadata_all).reshape(-1,1)
    
    def get_group_prompts(self):
        raise NotImplementedError
    
    def get_raw_y(self, split='test'):
        y_all = []
        for env in self.envs:
            dataset = FolderDataset(os.path.join(self.root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            y_all.extend([self.class_to_idx[p.split('/')[5]] for p in dataset.image_paths])
        return torch.Tensor(y_all)

class BreedsNonliving26Dataset(BreedsDataset):
    def __init__(self):
        super().__init__()
        label_map = {0: 'bag', 1: 'ball', 2: 'boat', 3: 'body armor, body armour, suit of armor, suit of armour, coat of mail, cataphract',\
                    4: 'bottle', \
                    5: 'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', \
                    6: 'car, auto, automobile, machine, motorcar', 7: 'chair', 8: 'coat', 9: 'digital computer', \
                    10: 'dwelling, home, domicile, abode, habitation, dwelling house', 11: 'fence, fencing', 12: 'hat, chapeau, lid', 
                    13: 'keyboard instrument', 14: 'mercantile establishment, retail store, sales outlet, outlet', 15: 'outbuilding', 
                    16: 'percussion instrument, percussive instrument', 17: 'pot', 18: 'roof', 19: 'ship', 20: 'skirt', 
                    21: 'stringed instrument', 22: 'timepiece, timekeeper, horologe', 23: 'truck, motortruck', 
                    24: 'wind instrument, wind', 25: 'squash'}
        self.class_to_idx = {v:k for k,v in label_map.items()}
        self.metadata_map = {
            'source': 0,
            'target': 1
        }
        self.envs = ['source', 'target']
        self.metadata_map_reverse = {v:k for k,v in self.metadata_map.items()}
        self.root_dir = f'{DATA_DIR}/breeds_nonliving26'

class BreedsLiving17Dataset(BreedsDataset):
    def __init__(self):
        super().__init__()
        label_map = {0: 'salamander', 1: 'turtle', 2: 'lizard', \
                     3: 'snake, serpent, ophidian', 4: 'spider', \
                    5: 'grouse', 6: 'parrot', 7: 'crab', \
                    8: 'dog', \
                    9: 'wolf', 10: 'fox', \
                    11: 'domestic cat',\
                    12: 'bear', 13: 'beetle', 14: 'butterfly', 15: 'ape', \
                    16: 'monkey'}
        self.class_to_idx = {v:k for k,v in label_map.items()}
        self.metadata_map = {
            'source': 0,
            'target': 1
        }
        self.envs = ['source', 'target']
        self.metadata_map_reverse = {v:k for k,v in self.metadata_map.items()}
        self.root_dir = f'{DATA_DIR}/breeds_living17'
    
class CatDogDataset:
    def __init__(self):
        self.class_to_idx = {
            'cat': 0,
            'dog': 1,
        }
        self.metadata_map = {
            'indoor': 0,
            'outdoor': 1,
        }
        self.metadata_map_reverse = {v:k for k,v in self.metadata_map.items()}
    
    def get_dataloaders(self, batch_size, transform, return_test=True):
        root_dir=f'{DATA_DIR}/stablediffusion_cat_dog'
        dataset_all = []
        for env in self.metadata_map.keys():
            dataset = FolderDataset(os.path.join(root_dir, env), transform=transform, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            dataset_all.append(dataset)
        dataset = ConcatDataset(dataset_all)
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]
    
    def get_labels(self,):
        self.labels = [f'{c_}' for c_ in list(self.class_to_idx.keys())]
        return self.labels

class NurseFirefighterDataset:
    def __init__(self):
        self.class_to_idx = {
            'nurse': 0,
            'firefighter': 1,
        }
        self.metadata_map = {
            'female': 0,
            'male': 1,
        }
        self.metadata_map_reverse = {v:k for k,v in self.metadata_map.items()}
    
    def get_dataloaders(self, batch_size, transform, return_test=True):
        root_dir=f'{DATA_DIR}/stablediffusion_nurse_firefighter'
        dataset_all = []
        for env in self.metadata_map.keys():
            dataset = FolderDataset(os.path.join(root_dir, env), transform=transform, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            dataset_all.append(dataset)
        dataset = ConcatDataset(dataset_all)
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]
    
    def get_labels(self,):
        self.labels = [f'{c_}' for c_ in list(self.class_to_idx.keys())]
        return self.labels

class CXR14Dataset:
    def get_labels(self):
        return ['non-pneumothorax','pneumothorax']

class VLCSDataset:
    def __init__(self):
        self.class_to_idx = {
            'bird': 0,
            'car': 1,
            'chair': 2,
            'dog': 3,
            'person': 4,
        }
        self.metadata_map = {
            'Caltech101': 0,
            'LabelMe': 1,
            'SUN09': 2,
            'VOC2007': 3,
        }
        self.metadata_map_reverse = {v:k for k,v in self.metadata_map.items()}

    def get_dataloaders(self, batch_size, transform, return_test=True):
        root_dir=f'{DATA_DIR}/VLCS'
        envs = list(self.metadata_map.keys())
        dataset_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=transform, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            dataset_all.append(dataset)
        dataset = ConcatDataset(dataset_all)
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]
    
    def get_labels(self,):
        self.labels = [f'this object is {c_}' for c_ in list(self.class_to_idx.keys())]
        return self.labels
    
    def get_file_paths(self, split='test'):
        root_dir=f'{DATA_DIR}/VLCS'
        envs = list(self.metadata_map.keys())
        file_paths_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            file_paths_all.extend(dataset.image_paths)
        return file_paths_all
    
    def get_raw_metadata(self, split='test'):
        root_dir=f'{DATA_DIR}/VLCS'
        envs = list(self.metadata_map.keys())
        metadata_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            metadata_all.extend([self.metadata_map[env] for i in range(len(dataset))])
        return torch.Tensor(metadata_all).reshape(-1,1)
    
    def get_group_prompts(self):
        group_prompts = []
        for class_ in list(self.class_to_idx.keys()):
            for domain in list(self.metadata_map.keys()):
                if '_' in domain:
                    domain = domain.replace('_', '')
                group_prompts.append(f'{class_} {domain}')
        return group_prompts
    
    def get_raw_y(self, split='test'):
        root_dir=f'{DATA_DIR}/VLCS'
        envs = list(self.metadata_map.keys())
        y_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            y_all.extend([self.class_to_idx[p.split('/')[5]] for p in dataset.image_paths])
        return torch.Tensor(y_all)
    
class PACSDataset:
    def __init__(self):
        self.class_to_idx = {
            'dog': 0,
            'elephant': 1,
            'giraffe': 2,
            'guitar': 3,
            'horse': 4,
            'house': 5,
            'person': 6,
        }
        self.metadata_map = {
            'art_painting': 0,
            'cartoon': 1,
            'photo': 2,
            'sketch': 3,
        }
        self.metadata_map_reverse = {v:k for k,v in self.metadata_map.items()}

    def get_dataloaders(self, batch_size, transform, return_test=True):
        root_dir=f'{DATA_DIR}/PACS'
        envs = ['art_painting', 'cartoon', 'photo', 'sketch']
        dataset_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=transform, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            dataset_all.append(dataset)
        dataset = ConcatDataset(dataset_all)
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]
    
    def get_labels(self,):
        self.labels = [f'an image of {c_}' for c_ in list(self.class_to_idx.keys())]
        return self.labels
    
    def get_file_paths(self, split='test'):
        root_dir=f'{DATA_DIR}/PACS'
        envs = ['art_painting', 'cartoon', 'photo', 'sketch']
        file_paths_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            file_paths_all.extend(dataset.image_paths)
        return file_paths_all
    
    def get_raw_metadata(self, split='test'):
        root_dir=f'{DATA_DIR}/PACS'
        envs = ['art_painting', 'cartoon', 'photo', 'sketch']
        metadata_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            metadata_all.extend([self.metadata_map[env] for i in range(len(dataset))])
        return torch.Tensor(metadata_all).reshape(-1,1)
    
    def get_group_prompts(self):
        group_prompts = []
        for class_ in list(self.class_to_idx.keys()):
            for domain in list(self.metadata_map.keys()):
                if '_' in domain:
                    domain = domain.replace('_', '')
                group_prompts.append(f'{class_} {domain}')
        return group_prompts
    
    def get_raw_y(self, split='test'):
        root_dir=f'{DATA_DIR}/PACS'
        envs = ['art_painting', 'cartoon', 'photo', 'sketch']
        y_all = []
        for env in envs:
            dataset = FolderDataset(os.path.join(root_dir, env), transform=None, class2idx=self.class_to_idx, metadata_map=self.metadata_map)
            y_all.extend([self.class_to_idx[p.split('/')[5]] for p in dataset.image_paths])
        return torch.Tensor(y_all)


class ColoredMNISTDataset:
    def get_dataloaders(self, batch_size, transform, return_test=True):
        root_dir=f'{DATA_DIR}/ColoredMNIST'

        envs = ["0","1","2"]
        train_env = np.random.choice(envs)
        test_envs = list(set(envs) - set([train_env]))
        if not return_test:
            print(f"train env = {train_env} test envs = {test_envs}")
            envs = [train_env]
            envs.extend(test_envs)
            dataloaders = []
            for env in envs:
                dataset = FolderDataset(os.path.join(root_dir, env), transform=transform)
                dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
            return dataloaders
        test_env = np.random.choice(envs)
        dataset = FolderDataset(os.path.join(root_dir, test_env), transform=transform)
        return [DataLoader(dataset, batch_size=batch_size, shuffle=True)]
    
class MultiEnvDataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.Resize((448,448)), transforms.ToTensor()])
        self.dataset_dict = {
            const.WATERBIRDS_NAME: WaterbirdsDataset,
            const.CMNIST_NAME: ColoredMNISTDataset,
            const.CELEBA_NAME: CelebADataset,
            const.PACS_NAME: PACSDataset,
            const.SD_CATDOG_NAME: CatDogDataset,
            const.SD_NURSE_FIREFIGHTER_NAME: NurseFirefighterDataset,
            const.CXR_NAME: CXR14Dataset,
            const.BREEDS17_NAME: BreedsLiving17Dataset,
            const.BREEDS26_NAME: BreedsNonliving26Dataset,
            const.CIVILCOMMENTS_NAME: CivilCommentsDataset,
            const.HATEXPLAIN_NAME: HateXplainDataset,
            const.AMAZON_NAME: AmazonDataset,
            const.GENDER_BIAS_NAME: GenderBiasDataset,
            const.VLCS_NAME: VLCSDataset,
        }
    
    def get_dataloaders(self, dataset_name, batch_size, return_test=True):
        assert dataset_name.lower() in [k.lower() for k in list(self.dataset_dict.keys())]
        return self.dataset_dict[dataset_name]().get_dataloaders(batch_size, self.transform, return_test)

    def get_labels(self, dataset_name):
        return self.dataset_dict[dataset_name]().get_labels()

    def get_file_paths(self, dataset_name, split='test'):
        return self.dataset_dict[dataset_name]().get_file_paths(split)
    
    def get_raw_metadata(self, dataset_name, split='test'):
        return self.dataset_dict[dataset_name]().get_raw_metadata(split)
    
    def get_raw_y(self, dataset_name, split='test'):
        return self.dataset_dict[dataset_name]().get_raw_y(split)
    
    def get_group_prompts(self, dataset_name):
        return self.dataset_dict[dataset_name]().get_group_prompts()
    
if __name__ == '__main__':
    loader = MultiEnvDataset().get_dataloaders(const.AMAZON_NAME, 16)