from model import LSTMVAE
import os
import torch
import torch.nn.functional as F
import numpy as np
import glob
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import argparse

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
            parameters = data['parameters']
            observations = data['y']
        
        parameters = list(parameters.values())
        parameters = np.array(parameters)
        observations = np.array(observations)

        parameters_tensor = torch.tensor(parameters, dtype=torch.float32)
        observations_tensor = torch.tensor(observations, dtype=torch.float32)
        # observations_tensor = (observations_tensor + 5.825352944438144e-09)/(670.4232307413174+5.825352944438144e-09)
        observations_tensor = (observations_tensor - torch.mean(observations_tensor))/(torch.std(observations_tensor))
        return parameters_tensor, observations_tensor

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    
    return xx_pad, yy_pad

def train_vae(model, train_loader, test_loader, save_path = 'LV_EQUATION_LSTM.pt', learning_rate=0.001, epochs=1000, device = 'cpu'):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    ## interation setup
    batch_train_loss = []
    batch_val_loss = []
    kld_weight = 0.00025
    recon_weight = 1.0
    
    best_loss = 999999
    ## training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y = y.mT
            
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

        print(f"Epoch: {epoch} Train Loss : {train_loss/len(train_loader)}")
        print(f"Epoch: {epoch} Reconstruction Loss : {recon_loss}")
        print(f"Epoch: {epoch} KL divergence Loss : {kl_div_loss}")
        print(f"Epoch: {epoch} Wegihted KL divergence Loss : {kld_weight*kl_div_loss}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                x, y = batch
                x, y = x.to(device), y.to(device)

                y = y.mT
                
                y_hat, z, mu, logvar = model(y)

                recon_loss = F.mse_loss(y_hat, y)

                # Compute KL divergence
                kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Total loss
                total_loss = recon_weight*recon_loss + kld_weight*kl_div_loss
                eval_loss += total_loss.item()

            print(f"Epoch: {epoch} Validation Loss : {eval_loss/len(test_loader)}")
            print(f"Epoch: {epoch} Reconstruction Loss : {recon_loss}")
            print(f"Epoch: {epoch} KL Divergence Loss : {kl_div_loss}")
            print(f"Epoch: {epoch} Wegihted KL divergence Loss : {kld_weight*kl_div_loss}")
        if eval_loss/len(test_loader) < best_loss:
            best_loss = eval_loss/len(test_loader)
            torch.save(model.state_dict(), f'{save_path}')
            print(f"Model saved at epoch {epoch}.")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of the hidden layer')
    parser.add_argument('--latent_size', type=int, default=4, help='Size of the latent layer')
    parser.add_argument('--model_name', type=str, default='LV_EQUATION_LSTM', help='Name of the model file')
    args = parser.parse_args()

    BATCH_SIZE = 128
    INPUT_SIZE = 2
    HIDDEN_SIZE = args.hidden_size
    LATENT_SIZE = args.latent_size
    MODEL_NAME = str(args.model_name) + '_' + str(HIDDEN_SIZE) + '_' + str(LATENT_SIZE) + '.pt'

    print('Data loading initialized.')
    train_ds = SinusoidalDataset(root = 'data', suffix = 'train/processed')
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    val_ds = SinusoidalDataset(root = 'data', suffix = 'val/processed')
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    test_ds = SinusoidalDataset(root = 'data', suffix = 'test/processed')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_collate)

    print('Data loading completed.')
    print('Model loading initialized.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMVAE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE).to(device)
    print('Model loading completed.')
    print('Training initialized.')
    model = train_vae(model, train_dataloader, val_dataloader, save_path = MODEL_NAME, learning_rate=0.001, epochs = 1000, device = device)
    print('Training completed.')

    if not os.path.exists('reconstruction'):
        os.makedirs('reconstruction')

    if not os.path.exists(f'reconstruction/{MODEL_NAME}'):
        os.makedirs(f'reconstruction/{MODEL_NAME}')

    pred_loss = []
    
    model = LSTMVAE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))

    model.eval()
    for i, batch in enumerate(test_dataloader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        y = y.mT
        y_hat, z, mu, logvar = model(y)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(y_hat, y)
        pred_loss.append(recon_loss.item())

        plt.figure(figsize=(10, 5))
        plt.plot(y[0].squeeze().cpu().numpy())
        plt.plot(y_hat[0].squeeze().detach().cpu().numpy())
        plt.savefig(f'reconstruction/{MODEL_NAME}/reconstruction_{i}_{recon_loss}.png')

    print("Prediction Loss : {}".format(np.mean(pred_loss)))