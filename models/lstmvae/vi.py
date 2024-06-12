from model import VI, LSTMVAE
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
from main import MultiSeriesODEDataset, SinusoidalDataset, pad_collate
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.squeeze(0)
        return x


if __name__ == "__main__":
    print('Data loading initialized.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = SinusoidalDataset(root = 'data', suffix = 'train/processed')
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)

    val_ds = SinusoidalDataset(root = 'data', suffix = 'val/processed')
    val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=True, collate_fn=pad_collate)

    test_ds = SinusoidalDataset(root = 'data', suffix = 'test/processed')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=pad_collate)

    model = LSTMVAE(input_size = 1, hidden_size = 64, latent_size = 32, num_layers = 1)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    reg_model = RegressionModel()
    reg_model.to(device)
    optimizer = optim.Adam(reg_model.parameters(), lr=0.0001)

    for epoch in range(30):
        train_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_hat, z, mu, logvar = model(y)

            x_hat = reg_model(z)

            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        print(f'Epoch {epoch}: Train Loss :{train_loss / len(train_dataloader)}')

        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat, z, mu, logvar = model(y)
                x_hat = reg_model(z)

                loss = F.mse_loss(x_hat, x)
                
                val_loss += loss.item()

        print(f'Epoch {epoch}: Val Loss :{val_loss / len(val_dataloader)}')

    torch.save(reg_model.state_dict(), 'reg_model.pth')

    print('Training completed.')
    
    reg_model.load_state_dict(torch.load('reg_model.pth', map_location=device))

    test_loss = 0.0
    a_pred, b_pred = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat, z, mu, logvar = model(y)
            x_hat = reg_model(z)
            
            a_pred.extend(x_hat[:, 0].detach().cpu().numpy().tolist())
            b_pred.extend(x_hat[:, 1].detach().cpu().numpy().tolist())

            loss = F.mse_loss(x_hat, x)
            test_loss += loss.item()

    print(f'Test Loss :{test_loss / len(test_dataloader)}')

    a_true = x[0][0]
    b_true = x[0][1]

    # # Subplot for a_pred
    plt.subplot(1, 2, 1)
    plt.hist(np.array(a_pred).flatten(), label='a_pred')
    plt.axvline(x=a_true, color='r', linestyle='--')

    # Subplot for b_pred
    plt.subplot(1, 2, 2)
    plt.hist(np.array(b_pred).flatten(), label='b_pred')
    plt.axvline(x=b_true, color='r', linestyle='--')

    a_pred_mean = round(np.mean(np.array(a_pred).flatten()), 2)
    b_pred_mean = round(np.mean(np.array(b_pred).flatten()), 2)
    plt.suptitle(f'a_pred: {a_pred_mean}, b_pred: {b_pred_mean}, a_true: {a_true}, b_true: {b_true}')
    plt.savefig(f'reconstruction/posterior_distribution/example.png')

    print('Relative Error a: ', (a_pred_mean - a_true) / a_true)
    print('Relative Error b: ', (b_pred_mean - b_true) / b_true)
    