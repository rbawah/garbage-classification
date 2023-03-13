import os
import numpy as np 
import pandas as pd
from PIL import Image
import re
from collections import Counter
import time
import torchvision
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import random
from torch.utils.data import (TensorDataset, 
                              Dataset, 
                              Subset,
                              random_split, 
                              DataLoader, 
                              RandomSampler, 
                              SequentialSampler, 
                              DataLoader)



def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class GarbageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, 
                    #transform: Optional[Callable] = None, 
                    #target_transform: Optional[Callable] = None, 
                    loader: Callable[[str], Any] = default_loader,
                    is_valid_file: Optional[Callable[[str], bool]]= None,
                    ):

        super().__init__(root, 
                        loader, 
                        IMG_EXTENSIONS if is_valid_file is None else None,
                        #transform=transform, 
                        #target_transform=target_transform,
                        is_valid_file=is_valid_file,)  

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            sample = self.target_transform(sample)
        return sample, target, sample

        self.imgs = self.samples

    def get_imgs(self):
        path, target = None, None
        img = list()
        for sample in self.samples:
            path, target = sample
            img.append((path, target, path.split("\\")[-1].split(".")[0]))
        return img

# use random_split to split data into train, val, and test sets
def split_dataset(image_dataset, test_size=0.2):

    val_size = test_size
    train_size = 1 - (test_size + val_size)
    lengths = [train_size, val_size, test_size]
    
    train_split, val_split, test_split = random_split(image_dataset, lengths=lengths)

    print(f'Train size: {len(train_split)}')
    print(f'Val size: {len(val_split)}')
    print(f'Test size: {len(test_split)}')

    return train_split, val_split, test_split


class GarbageDataset(Dataset):
    def __init__(self, garbage_data, transform=None, text_data=False):
        self.garbage_data = garbage_data
        self.transform = transform
        self.filenames = None
        self.labels = None
        self.text_data = text_data
        if text_data:
            self.filenames = self.get_text_lines()
            self.labels = self.get_text_labels()
        
    def __len__(self):
        return len(self.garbage_data)
        
    def __getitem__(self, index):
        image_file, label = self.garbage_data[index]
        if self.transform:
            image_file = self.transform(image_file)
        if self.text_data:
            return image_file, label, self.filenames[index]
        return image_file, label
    
    def get_text_lines(self):
        return [self.preprocess_filename(file_name) for file_name, label in self.garbage_data.imgs]
    
    def get_text_labels(self):
        return [label for file_name, label in self.garbage_data.imgs]

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



# class FileNames():
#     def __init__(self, transform=None, target_transform=None):
#         self.img_labels, self.label_classes = self.create_df("data")
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         file_path = self.file_paths[idx]
#         image = Image.open(file_path)
#         if self.transform:
#             image = self.transform(image)
#         return image, label
    
#     def create_df(self, path):
#         data_dict = dict()
#         label_classes = list()
#         label_list = os.listdir(path)
#         print("The labels and corresponding indexes are:")
#         for idx, label in enumerate(label_list):
#             print(f'{int(idx)+1, label}')
#             label_classes.append(label)
#             filenames = os.listdir(path+ "/" + label)
#             for f in filenames:
#                 self.append_value(data_dict, "_label", int(idx)+1)
#                 self.append_value(data_dict, "_filename", self.preprocess_filename(f).split(".")[-2])
#         return pd.DataFrame.from_dict(data_dict), label_classes
    
#     def append_value(self, dict_obj, key, value):
#         if key in dict_obj:
#             if not isinstance(dict_obj[key], list):
#                 dict_obj[key] = [dict_obj[key]]
#             dict_obj[key].append(value)
#         else:
#             dict_obj[key] = value
            
#     def preprocess_filename(self, input_str):
#         if not re.match(r".*[A-Za-z]", input_str):
#             return None
#         input_str = input_str.replace(r"#U0640", "_")
#         input_str = re.sub(r"\d", "", input_str)
#         input_str = re.sub(r"_", " ", input_str)
#         input_str = re.sub(r"-", " ", input_str)
#         input_str = re.sub(r"\(\)", "", input_str)
#         input_str = input_str.strip()
#         input_str = input_str.lower()
#         return input_str
            
#     def get_df(self):
#         return self.img_labels
    
#     def get_tuples(self, the_labels):
#         return [tuple(x) for x in the_labels.values]
    
#     def get_label_classes(self):
#         return self.label_classes
    
#     def get_iter(self):
#         return iter(get_tuples())
    
