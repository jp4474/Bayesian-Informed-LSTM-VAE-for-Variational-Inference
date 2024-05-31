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
# def collate_fn(data):
#    data = pack_padded_sequence(torch.tensor(data), 10, batch_first=True)
#    return data

def train(model, train_loader, test_loader, learning_rate=0.001, epochs=1000, device = 'cpu'):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    ## interation setup
    epochs = tqdm(range(epochs))

    ## training
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        for i, batch_data in train_iterator:
            x = batch_data.to(device)
            ## reshape
            mloss, recon_x, info = model(x)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})

        model.eval()
        eval_loss = 0
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, batch_data in test_iterator:
                x = batch_data.to(device)
                ## reshape
                mloss, recon_x, info = model(x)

                eval_loss += mloss.mean().item()

                test_iterator.set_postfix({"eval_loss": float(mloss.mean())})

            eval_loss = eval_loss / len(test_loader)
            #writer.add_scalar("eval_loss", float(eval_loss), epoch)
            print("Evaluation Score : [{}]".format(eval_loss))

    return model


if __name__ == "__main__":
    train_ds = MultiSeriesODEDataset(root = 'data', suffix = 'processed')

    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    input_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMVAE(input_size=input_size, hidden_size=12, latent_size=6).to(device)
    model = train(model, train_dataloader, train_dataloader, device)