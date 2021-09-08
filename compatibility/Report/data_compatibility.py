import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import glob
import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image
import itertools
import csv
import pandas as pd

from utils import Config

import matplotlib.pyplot as plt


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()

    def get_data_transforms(self):

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
        #     std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.Resize(244),
        #         transforms.ToTensor(),
        #         normalize
        #     ]),
        #     'test': transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize
        #     ]),
        # }
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomRotation(30),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }
        return data_transforms

    def get_json_filtype(self, file_):
        return 'train' if 'train' in file_.name else 'valid' if 'valid' in file_.name else 'test'

    def json_to_dict(self, json_f):

        json_dict = {}
        meta_json = json.load(json_f)
        for obj in meta_json:
            item_dict = {}
            for item in obj["items"]:
                item_dict[int(item["index"])] = item["item_id"]
            json_dict[obj["set_id"]] = item_dict
        return json_dict

    def get_item_id(self, outfits, json_dict):

        item_ids = list()
        for outfit in outfits:
            outfit = outfit.split("_")
            outfit_id = outfit[0]
            index = int(outfit[1])
            item_ids.append(str(json_dict[outfit_id][index]+'.jpg'))
        return item_ids

    def create_pairs(self, ids, class_):

        pair_list = []
        for x in itertools.combinations(ids, 2):
            lst = list(x)
            lst.append(class_)
            pair_list.append(lst)
        return pair_list

    def create_dataset(self):

        paths = osp.join(self.root_dir, 'compatibility_*.txt')
        for filename in glob.glob(paths):

            with open(filename, 'r') as file_:
                f_type = self.get_json_filtype(file_)
                f_name = f_type + '.json'
                csv_f_name = f_type + '_compatibility.csv'
                print("-----------------------------------------")
                print("     Processing text file:", file_.name)

                with open(osp.join(self.root_dir, csv_f_name), 'w') as file:
                    writer = csv.writer(file)
                    print("     Writing to csv:", csv_f_name)

                    with open(glob.glob(osp.join(self.root_dir, f_name))[0], 'r') as json_f:
                        print(
                            "     Referencing JSON meta data and converting. Please wait ...")
                        meta_dict = self.json_to_dict(json_f)

                        for line in tqdm(file_):
                            if f_type in ['train', 'valid']:
                                line = line.split(" ", 1)
                                class_ = int(line[0])
                                outfits = line[1].split()
                            else:
                                outfits = line.split()
                                class_ = 0
                            ids = self.get_item_id(outfits, meta_dict)
                            pairs_list = self.create_pairs(ids, class_)
                            writer.writerows(pairs_list)

                    print("     Completed writing to file:", csv_f_name)
        print("-----------------------------------------")

        df = pd.read_csv(osp.join(self.root_dir, 'train_compatibility.csv'))
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # This will split into 80% for train, 20% for test
        msk = np.random.rand(len(df)) < 0.81
        train_df = df[msk]
        test_df = df[~msk]
        
        train_df.to_csv(
            osp.join(self.root_dir, 'train_compatibility.csv'), index=False, header=False)
        test_df.to_csv(
            osp.join(self.root_dir, 'test_compatibility.csv'), index=False, header=False)
        return True


class polyvore_pairwise_train(Dataset):

    def __init__(self, transform=None):
        self.training_df = pd.read_csv(
            osp.join(Config['root_path'], 'train_compatibility.csv'))
        self.training_df.columns = ["image1", "image2", "label"]
        self.training_dir = osp.join(Config['root_path'], 'images')
        self.transform = transform
        self.debug = Config['debug']
        if self.debug:
            self.training_df = self.training_df.sample(
                frac=1).reset_index(drop=True)[:10000]

    def __len__(self):
        return len(self.training_df)

    def __getitem__(self, index):

        # getting the image path
        image1_path = os.path.join(
            self.training_dir, self.training_df.iat[index, 0])
        image2_path = os.path.join(
            self.training_dir, self.training_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.training_df.iat[index, 2])], dtype=np.float32))


class polyvore_pairwise_valid(Dataset):

    def __init__(self, transform=None):
        self.valid_df = pd.read_csv(
            osp.join(Config['root_path'], 'valid_compatibility.csv'))
        self.valid_df.columns = ["image1", "image2", "label"]
        self.valid_dir = osp.join(Config['root_path'], 'images')
        self.transform = transform
        self.debug = Config['debug']
        if self.debug:
            self.valid_df = self.valid_df.sample(
                frac=1).reset_index(drop=True)[:10000]

    def __len__(self):
        return len(self.valid_df)

    def __getitem__(self, index):

        # getting the image path

        image1_path = os.path.join(self.valid_dir, self.valid_df.iat[index, 0])
        image2_path = os.path.join(self.valid_dir, self.valid_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.valid_df.iat[index, 2])], dtype=np.float32))


class polyvore_pairwise_test(Dataset):

    def __init__(self, transform=None):
        self.test_df = pd.read_csv(
            osp.join(Config['root_path'], 'test_compatibility.csv'))
        self.test_df.columns = ["image1", "image2", "label"]
        self.test_dir = osp.join(Config['root_path'], 'images')
        self.transform = transform
        self.debug = Config['debug']
        if self.debug:
            self.test_df = self.test_df.sample(
                frac=1).reset_index(drop=True)[:10000]

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, index):

        # getting the image path

        image1_path = os.path.join(self.test_dir, self.test_df.iat[index, 0])
        image2_path = os.path.join(self.test_dir, self.test_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.test_df.iat[index, 2])], dtype=np.float32))


def get_dataloader():
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    if(dataset.create_dataset()):
        print("Datasets created successfully")

    train_set = polyvore_pairwise_train(transform=transforms['train'])
    valid_set = polyvore_pairwise_valid(transform=transforms['valid'])
    test_set = polyvore_pairwise_test(transform=transforms['test'])

    datasets = {'train': train_set, 'test': test_set, 'valid': valid_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x == 'train' else False,
                                 batch_size=Config['batch_size'],
                                 num_workers=Config['num_workers'])
                   for x in ['train', 'test', 'valid']}
    return dataloaders
