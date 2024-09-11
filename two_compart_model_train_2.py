import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from tqdm.notebook import trange
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from Models import lstm_encoder, lstm_decoder
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0
    for data in dataloader:
        x = data['x'].to(device)
        y = data['y'].to(device)
        optimizer.zero_grad()
        x_recon, mean, logvar = model(y)
        loss = criterion(x_recon, x, mean, logvar)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(data)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for data in dataloader:
            x = data['x'].to(device)
            y = data['y'].to(device)
            x_recon, mean, logvar = model(y)
            loss = criterion(x_recon, x, mean, logvar)
            running_loss += loss.item() * len(data)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def visualize_reconstruction(model, x, y, epoch):
    with torch.no_grad():
        input = y.to(device)
        x_recon, mean, logvar = model(input)

    x_recon = x_recon.detach().cpu().numpy()
    fig, axs = plt.subplots(1,3, figsize = (30,10))

    for i in range(3):
        ax = axs[i]
        ax.hist(x[:,i], bins = 50, color = 'orange', alpha = 0.5)
        ax.hist(x_recon[:,i], bins = 50, color = 'blue', alpha = 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(f'TWO_COMPARTMENT_{HIDDEN_DIM}_{LATENT_DIM}_{ALPHA}_{epoch}_NEW_HIST.png')
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class loss_fn(nn.Module):
    def __init__(self, alpha):
        super(loss_fn, self).__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true, mean, log_var):
        # RECON = torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean')
        RECON = torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean'))
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return (RECON + self.alpha*KLD)

class LSTMVAE_INSTANCE(nn.Module):
    """
    Instance level entangled representation learning model for timeseries
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(LSTMVAE_INSTANCE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = lstm_encoder(input_dim, hidden_size=hidden_dim)
        self.mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)
        # self.decoder = lstm_decoder(latent_dim, hidden_size=hidden_dim, output_size=input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        self.multihead_attn = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = 8, dropout=0.1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (h, c) = self.encoder(x)
        att_output, _ = self.multihead_attn(h, h, h)
        att_output = att_output.permute(1, 0, 2).reshape(batch_size, -1)
        mu = self.mu(att_output)
        logvar = self.logvar(att_output)
        z = self.reparameterize(mu, logvar)
        theta_hat = self.decoder(z)
        return theta_hat, mu, logvar

    
class npyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.x[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a two compartment model.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--input_dim', type=int, default=2, help='Dimension of the input data')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of the hidden layers')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of the latent space')
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs for training')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Alpha parameter for loss function')

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    INPUT_DIM = 3 # args.input_dim
    HIDDEN_DIM = args.hidden_dim
    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    ALPHA = args.alpha

    print(f'Running LSTM-VAE with hidden dim: {HIDDEN_DIM}, latent dim: {LATENT_DIM}, alpha: {ALPHA}')
    print(f'Configurations: Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Epochs: {EPOCHS}')

    Y_train = np.load('data_bcell/two_compt_sims_noise_train.npy')
    X_train = np.load('data_bcell/two_compt_params_train.npy')
    Y_val = np.load('data_bcell/two_compt_sims_noise_val.npy')
    X_val = np.load('data_bcell/two_compt_params_val.npy')
    Y_test = np.load('data_bcell/two_compt_sims_noise_test.npy')
    X_test = np.load('data_bcell/two_compt_params_test.npy')

    y_scaler = MinMaxScaler()
    
    Y_train_norm = np.log(Y_train)
    s = np.abs(Y_train_norm.reshape(-1, Y_train_norm.shape[-1])).mean(axis=0)
    Y_val_norm = np.log(Y_val)
    Y_test_norm = np.log(Y_test)

    # Y_train_norm = Y_train_norm / s
    # Y_val_norm = Y_val_norm / s
    # Y_test_norm = Y_test_norm / s

    Y_train_norm = torch.tensor(Y_train_norm, dtype=torch.float32)
    Y_val_norm = torch.tensor(Y_val_norm, dtype=torch.float32)
    Y_test_norm = torch.tensor(Y_test_norm, dtype=torch.float32)

    x_scaler = StandardScaler() #MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    train_dataset = npyDataset(X_train_scaled, Y_train_norm)
    val_dataset = npyDataset(X_val_scaled, Y_val_norm)
    test_dataset = npyDataset(X_test_scaled, Y_test_norm)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}')

    model = LSTMVAE_INSTANCE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, 7)
    model.to(device)

    print(device)
    print(count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = loss_fn(alpha=ALPHA)

    train_loss_values = []
    test_loss_values = []

    best_test_loss = np.inf
    best_epoch = 0
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, val_loader, criterion)
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)

        interval = 2 if epoch < 10 else 100
        if (epoch + 1) % interval == 0:
            print('Epoch: {} Train: {:.10f}, Test: {:.10f}'.format(epoch + 1, train_loss_values[epoch], test_loss_values[epoch]))

        if epoch % 10 == 0 and epoch > 0:
            visualize_reconstruction(model, X_test_scaled, Y_test_norm, epoch)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            SAVE_DIR = f'TWO_COMPARTMENT_{HIDDEN_DIM}_{LATENT_DIM}_{ALPHA}_NEW.pth'
            torch.save(model.state_dict(), SAVE_DIR)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.semilogy(train_loss_values, label='Train Loss')
    ax.semilogy(test_loss_values, label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f'TWO_COMPARTMENT_{HIDDEN_DIM}_{LATENT_DIM}_{ALPHA}_NEW_LOSS.png')

    # plt.show()

    model = LSTMVAE_INSTANCE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, 7)
    model.load_state_dict(torch.load(SAVE_DIR, map_location=device))
    model.to(device)

    noisy_input = torch.tensor(Y_test_norm, dtype=torch.float32).to(device)
    y_recon, mean, logvar = model(noisy_input)
    Z_recon = y_recon.detach().cpu().numpy()

    fig, axs = plt.subplots(1,3, figsize = (30,10))

    for i in range(3):
        ax = axs[i]
        ax.hist(Z_recon[:,i], bins = 50, color = 'blue', alpha = 0.5)
        ax.hist(X_test_scaled[:,i], bins = 50, color = 'orange', alpha = 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(f'TWO_COMPARTMENT_{HIDDEN_DIM}_{LATENT_DIM}_{ALPHA}_NEW_HIST.png')
    