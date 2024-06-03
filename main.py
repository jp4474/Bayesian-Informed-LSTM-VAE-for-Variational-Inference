from model import LSTMVAE
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os
import glob
import pickle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
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
            data = pickle.load(f)['y']

        tensor_data = torch.tensor(data.values/10.0, dtype=torch.float32)
        tensor_data = tensor_data.unsqueeze(1)
        return tensor_data
    
def train_vae(model, train_loader, test_loader, learning_rate=0.001, epochs=1000, device = 'cpu'):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    ## interation setup
    batch_train_loss = []
    batch_val_loss = []

    ## training
    for epoch in range(epochs):
        model.train()
        #model.encoder.init_hidden(64)

        optimizer.zero_grad()

        train_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            x = batch_data.to(device)
            ## reshape
            mloss, recon_x, info = model(x)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            train_loss += mloss.mean().item()

        batch_train_loss.append(np.mean(train_loss))
        print("Train Loss : {}".format(np.mean(train_loss)))

        model.eval()
        eval_loss = 0

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                x = batch_data.to(device)
                mloss, recon_x, info = model(x)
                eval_loss += mloss.mean().item()

            batch_val_loss.append(np.mean(eval_loss))
            print("Validation Loss : {}".format(np.mean(eval_loss)))

    return model, batch_train_loss, batch_val_loss

def train_vi(model, train_loader, test_loader, learning_rate=0.001, epochs=1000, device = 'cpu'):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    ## interation setup
    batch_train_loss = []
    batch_val_loss = []

    ## training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            x = batch_data.to(device)
            ## reshape
            beta = model(x)

            # Backward and optimize
            optimizer.zero_grad()
            optimizer.step()

            train_loss += mloss.mean().item()

if __name__ == "__main__":
    print('Data loading initialized.')
    train_ds = SinusoidalDataset(root = 'data', suffix = 'train/processed')
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    val_ds = SinusoidalDataset(root = 'data', suffix = 'val/processed')
    val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    test_ds = SinusoidalDataset(root = 'data', suffix = 'test/processed')
    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    print('Data loading completed.')
    # print('Model loading initialized.')
    # input_size = 1
    device = 'cuda' 
    # model = LSTMVAE(input_size=input_size, hidden_size=64, latent_size=32).to(device)
    # print('Model loading completed.')
    # print('Training initialized.')
    # model, train_loss, val_loss = train_vae(model, train_dataloader, train_dataloader, learning_rate=0.001, epochs = 100, device = device)
    # print('Training completed.')
    # torch.save(model.state_dict(), 'model.pth')

    # if not os.path.exists('reconstruction'):
    #     os.makedirs('reconstruction')

    # model = LSTMVAE(input_size=input_size, hidden_size=64, latent_size=32)
    # model.load_state_dict(torch.load('model.pth', map_location='cpu'))

    # for i in range(10):
    #     x = next(iter(test_dataloader))
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(x[0].squeeze().numpy())
    #     plt.plot(model(x)[1][0].detach().numpy())
    #     plt.savefig(f'reconstruction/reconstruction_{i}.png')

    model = VI(input_size = 32, output_size = 2)

    