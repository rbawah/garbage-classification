import os
import numpy as np 
import pandas as pd
from PIL import Image
import re
import random
from collections import Counter
import time



class FileNames():
    def __init__(self, transform=None, target_transform=None):
        self.img_labels, self.label_classes = self.create_df("data")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def create_df(self, path):
        data_dict = dict()
        label_classes = list()
        label_list = os.listdir(path)
        print("The labels and corresponding indexes are:")
        for idx, label in enumerate(label_list):
            print(f'{int(idx)+1, label}')
            label_classes.append(label)
            filenames = os.listdir(path+ "/" + label)
            for f in filenames:
                self.append_value(data_dict, "_label", int(idx)+1)
                self.append_value(data_dict, "_filename", self.preprocess_filename(f).split(".")[-2])
        return pd.DataFrame.from_dict(data_dict), label_classes
    
    def append_value(self, dict_obj, key, value):
        if key in dict_obj:
            if not isinstance(dict_obj[key], list):
                dict_obj[key] = [dict_obj[key]]
            dict_obj[key].append(value)
        else:
            dict_obj[key] = value
            
    def preprocess_filename(self, input_str):
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
            
    def get_df(self):
        return self.img_labels
    
    def get_tuples(self, the_labels):
        return [tuple(x) for x in the_labels.values]
    
    def get_label_classes(self):
        return self.label_classes
    
    def get_iter(self):
        return iter(get_tuples())
    
