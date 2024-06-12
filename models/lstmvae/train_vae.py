from models/lstmvae/model import LSTMVAE
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
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

def train_vae(model, train_loader, test_loader, learning_rate=0.001, epochs=1000, device = 'cpu'):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    ## interation setup
    batch_train_loss = []
    batch_val_loss = []
    kld_weight = 0.00025
    recon_weight = 1.0

    ## training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_hat, z, mu, logvar = model(y)

            #x_hat = pad_packed_sequence(x_hat, batch_first=True)

            # Compute reconstruction loss
            recon_loss = F.mse_loss(y_hat, y)

            # Compute parameter reconstruction loss
            # param_loss = F.mse_loss(x_hat, x)

            # Compute KL divergence
            kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            total_loss = recon_weight*recon_loss + kld_weight*kl_div_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        print("Train Loss : {}".format(train_loss/len(train_loader)))

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                y_hat, z, mu, logvar = model(y)

                recon_loss = F.mse_loss(y_hat, y)

                # Compute KL divergence
                kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Total loss
                total_loss = recon_weight*recon_loss + kld_weight*kl_div_loss
                eval_loss += total_loss.item()

            print("Validation Loss : {}".format(eval_loss/len(test_loader)))

    return model

if __name__ == "__main__":

    print('Data loading initialized.')
    train_ds = SinusoidalDataset(root = 'data', suffix = 'train/processed')
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)

    val_ds = SinusoidalDataset(root = 'data', suffix = 'val/processed')
    val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)

    test_ds = SinusoidalDataset(root = 'data', suffix = 'test/processed')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=pad_collate)

    print('Data loading completed.')
    print('Model loading initialized.')
    input_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMVAE(input_size=input_size, hidden_size=64, latent_size=32).to(device)
    print('Model loading completed.')
    print('Training initialized.')
    model, train_loss, val_loss = train_vae(model, train_dataloader, val_dataloader, learning_rate=0.001, epochs = 100, device = device)
    print('Losses : ', train_loss, val_loss)
    print('Training completed.')
    torch.save(model.state_dict(), 'model.pth')

    if not os.path.exists('reconstruction'):
        os.makedirs('reconstruction')

    model = LSTMVAE(input_size=1, hidden_size=64, latent_size=32)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))

    pred_loss = []

    model.eval()
    for i, batch in enumerate(test_dataloader):
        x, y = batch
        #y = y.squeeze(0)
        y_hat, z, mu, logvar = model(y)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(y_hat, y)
        pred_loss.append(recon_loss.item())

        plt.figure(figsize=(10, 5))
        plt.plot(y[0].squeeze().numpy())
        plt.plot(y_hat[0].squeeze().detach().numpy())
        plt.savefig(f'reconstruction/reconstruction_{i}_{recon_loss}.png')

    print("Prediction Loss : {}".format(np.mean(pred_loss)))