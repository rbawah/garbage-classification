import os
import numpy as np 
import pandas as pd
from PIL import Image
import re
from collections import Counter
import time
import torchvision
from torchvision import datasets, transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import random
from transformers import BertTokenizer, BertForSequenceClassification

from torch.utils.data import (TensorDataset, 
                              Dataset, 
                              Subset,
                              random_split, 
                              DataLoader, 
                              RandomSampler, 
                              SequentialSampler, 
                              DataLoader)
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

# initialize BertTokenizer    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class GarbageRandomSplit(Dataset):
    def __init__(self, dataset, test_size = 0.2, val_size = None, transforms = None):
        self.dataset = dataset
        self.data_size = len(self.dataset)
        self.tranforms = transforms
        self.test_size = test_size
        self.val_size = test_size if val_size is None else val_size
        self.train_size = 1 - (self.test_size + self.val_size)

        self.train_set = self.dataset[0 : self.data_size * self.train_size]
        self.val_set = self.dataset[self.data_size * self.train_size : self.data_size * (self.train_size+self.val_size)]
        self.test_set = self.dataset[self.data_size * (self.train_size+self.val_size): ]

    def __len__(self):
        return {
            "train": len(self.train_set),
            "val": len(self.val_set),
            "test": len(self.test_set),
            "total": len(self.dataset)
            }

    def __getitem__(self, idx):
        # if self.tranforms:
        return self.dataset[idx]

    def get_datasets(self):
        return self.train_set, self.val_set, self.test_set

    def get_train_set(self):
        return self.train_set
    
    def get_test_set(self):
        return self.test_set

    def get_val_set(self):
        return self.val_set

# use random_split to split data into train, val, and test sets
def split_dataset(dataset, test_size = 0.2, val_size = None):
    random.shuffle(dataset)
    
    data_size = len(dataset)
    val_size = test_size if val_size is None else val_size
    train_size = 1 - (test_size + val_size)

    # 0-------a---b----data_size
    a = int(data_size*train_size)
    b = int(data_size*(train_size + val_size))
    train_split = dataset[slice(0, a)]
    val_split = dataset[slice(a, b)]
    test_split = dataset[slice(b, -1)]

    return train_split, val_split, test_split


def validate_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


class GarbageDataset(Dataset):
    def __init__(self, garbage_data, loader=default_loader, is_subset=False, transform=None, target_transform=None):
        self.garbage_data = garbage_data
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_subset = is_subset
        if not is_subset:
            self.filenames = self.get_filenames()
            self.input_ids, self.attention_mask = self.tokenize_filenames()
            self.labels = None
        
    def __len__(self):
        return len(self.garbage_data)
    
    def __setitem__(self, idx, element):
        self.garbage_data[idx] = element

    def __getitem__(self, index):
        if self.is_subset:
            path, label, input_ids, attention_mask = self.garbage_data[index]
            image_file = self.loader(path)
            if self.transform is not None:
                image_file = self.transform(image_file)
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image_file, label, input_ids, attention_mask
        else:
            path, label = self.garbage_data[index]
            image_file = self.loader(path)
            input_ids = self.input_ids[index]
            attention_mask = self.attention_mask[index]
            if self.transform:
                image_file = self.transform(image_file)
            return image_file, label, input_ids, attention_mask

    def preprocess_filename(self, input_str):
        input_str = input_str.split("\\")[-1].split(".")[0]
        if not re.match(r".*[A-Za-z]", input_str):
            return None
        input_str = input_str.replace(r"#U0640", "_")
        input_str = re.sub(r"\d", "", input_str)
        input_str = re.sub(r"_", " ", input_str)
        input_str = re.sub(r"-", " ", input_str)
        input_str = re.sub(r"\(\)", "", input_str)
        input_str = input_str.strip()
        input_str = input_str.lower()
        return input_str

    def get_filenames(self):
        filenames = []
        for image_name, _ in self.garbage_data: #.imgs:
            image_name = self.preprocess_filename(image_name)
            filenames.append(image_name)
        return filenames

    def tokenize_filenames(self):
        tok_dict = tokenizer(self.filenames, padding=True, truncation=True, return_tensors='pt')
        return tok_dict["input_ids"], tok_dict["attention_mask"]



def validate_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


class GarbageImageFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __setitem__(self, idx, image_element):
        self.imgs[idx] = image_element

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
        """
        if isinstance(index, slice):
            sample = None
            sample_list = []
            # step = 1 if index.step is None else index.step
            # start = index.start
            # stop = index.stop
            images = self.samples[index]
            for path, target in images:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                sample_list.append((sample, target))
            return sample_list

        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target

