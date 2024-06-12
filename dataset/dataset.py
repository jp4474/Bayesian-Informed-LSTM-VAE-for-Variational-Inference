import os
import glob
import pickle
from torch.utils.data import Dataset
import torch
import logging
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

class MultiSeriesODEDataset(Dataset):
    def __init__(self, root, suffix = 'train'):
        self.root = root
        self.folder_suffix = suffix
        self.data_files = glob.glob(os.path.join(self.root, self.folder_suffix, '*.pkl'))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)['x']

        tensor_data = torch.tensor(data, dtype=torch.float32)
        return tensor_data[:, 1:]
    
class SinusoidalDataset(Dataset):
    def __init__(self, root, suffix = 'train'):
        self.root = root
        self.folder_suffix = suffix
        self.data_files = glob.glob(os.path.join(self.root, self.folder_suffix, '*.pkl'))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            parameters = data['x']
            observations = data['y']
        parameters = np.array(parameters)
        observations = np.array(observations)

        parameters_tensor = torch.tensor(parameters, dtype=torch.float32)
        observations_tensor = torch.tensor(observations, dtype=torch.float32)
        return parameters_tensor, observations_tensor

def pad_collate(batch):
    data = [item[1].unsqueeze(1) for item in batch]
    data = pad_sequence(data, batch_first=True)
    parameters = [item[0] for item in batch]
    parameters = pad_sequence(parameters, batch_first=True)
    return parameters, data
