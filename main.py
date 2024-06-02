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

        tensor_data = torch.tensor(data.values, dtype=torch.float32)
        tensor_data = tensor_data.unsqueeze(1)
        return tensor_data
    
def train(model, train_loader, test_loader, learning_rate=0.001, epochs=1000, device = 'cpu'):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    ## interation setup
    epochs = tqdm(range(epochs))

    batch_train_loss = []
    batch_val_loss = []

    ## training
    for epoch in epochs:
        model.train()
        #model.encoder.init_hidden(64)

        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        train_loss = 0.0

        for i, batch_data in train_iterator:
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
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, batch_data in test_iterator:
                x = batch_data.to(device)
                mloss, recon_x, info = model(x)
                eval_loss += mloss.mean().item()

            batch_val_loss.append(np.mean(eval_loss))
            print("Validation Loss : {}".format(np.mean(eval_loss)))

    return model, batch_train_loss, batch_val_loss


if __name__ == "__main__":
    train_ds = SinusoidalDataset(root = 'data', suffix = 'train/processed')
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    model = LSTMVAE(input_size=input_size, hidden_size=64, latent_size=32).to(device)
    model, train_loss, val_loss = train(model, train_dataloader, train_dataloader, learning_rate=0.005, epochs = 30, device = device)
    
    x = next(iter(train_dataloader))
    plt.plot(x[0].squeeze().numpy())
    plt.plot(model(x.to(device))[1][0].squeeze().detach().cpu().numpy())
    plt.show()
    # val_ds = MultiSeriesODEDataset(root = 'data', suffix = 'val/processed')
    # val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=True)

    # input_size = 2
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = LSTMVAE(input_size=input_size, hidden_size=12, latent_size=6).to(device)
    # model, train_loss, val_loss = train(model, train_dataloader, val_dataloader, learning_rate=0.001, device = device)

    # torch.save(model.state_dict(), 'model.pth')
    # pd.DataFrame(train_loss).to_csv('train_loss.csv')
    # pd.DataFrame(val_loss).to_csv('val_loss.csv')