import torch
from torch.utils.data import Dataset
import json
from torch.utils.data.dataloader import *
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, spec_list='./dataset/spec_list.json'):
        with open(spec_list, 'r') as f:
             self.list = json.load(f)

    def __getitem__(self, index):
        spec_npy = self.list[index]
        spec = np.load(spec_npy)
        # print(spec.shape)
        return spec

    def __len__(self):
        return len(self.list)
