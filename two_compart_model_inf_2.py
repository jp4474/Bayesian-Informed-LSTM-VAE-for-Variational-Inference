import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import trange
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from ATT_BILSTM.model.ATT_LSTMVAE import lstm_encoder, lstm_decoder
from kan import KAN
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        # print(h.shape) [2, 1000, 512] [num_layers, batch_size, hidden_dim]
        att_output, _ = self.multihead_attn(h, h, h)
        # print(att_output.shape) [2, 1000, 512] [num_layers, batch_size, hidden_dim]
        att_output = att_output.permute(1, 0, 2).reshape(batch_size, -1)
        mu = self.mu(att_output)
        logvar = self.logvar(att_output)
        z = self.reparameterize(mu, logvar)
        theta_hat = self.decoder(z)
        return theta_hat, mu, logvar
    
    def get_embedding(self, x):
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            batch_size, seq_len, _ = x.size()
            _, (h, c) = self.encoder(x)
            att_output, _ = self.multihead_attn(h, h, h)
            att_output = att_output.permute(1, 0, 2).reshape(batch_size, -1)
            mu = self.mu(att_output)
            return mu
    
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

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
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
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Configure the parameters for the model.")

    # Add the arguments
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use args.input_dim, args.hidden_dim, args.latent_dim, and args.alpha in your code
    INPUT_DIM = args.input_dim
    HIDDEN_DIM = args.hidden_dim
    LATENT_DIM = args.latent_dim
    ALPHA = 0.0005 #args.alpha
    
    model = LSTMVAE_INSTANCE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, output_dim=3).to(device)
    model.load_state_dict(torch.load(f'TWO_COMPARTMENT_{HIDDEN_DIM}_{LATENT_DIM}_{ALPHA}_NEW.pth', map_location=device))
    model.eval()

    data = np.load('data_1/data.npy')
    obs = np.log(data[:,5:])
    true_obs = np.log(data[:,0:2])
    
    time = data[:,2]
    print(data.shape)

    obs = obs.reshape(1, -1, 2)
    true_obs = true_obs.reshape(1, -1, 2)

    scaler = StandardScaler() #MinMaxScaler()
    X_train = np.load('data_1/two_compt_params_train.npy')[:,0:3]
    scaler.fit(X_train)
    x_pred_list = []
    for i in range(1000):
        recon, mu, logvar = model(torch.tensor(obs, dtype=torch.float32).to(device))
        x_pred = recon.cpu().detach().numpy()
        x_pred_list.append(scaler.inverse_transform(x_pred.reshape(-1, 3)))

    x_pred_list = np.array(x_pred_list)
    x_pred_org = np.mean(x_pred_list, axis=0)
    # x_pred_org = scaler.inverse_transform(x_pred.reshape(-1, 3))
    print(x_pred_list.shape)
    print(x_pred_org.shape)

    x_test = np.array([0.0024, 0.05, 0.0005])
    mape = (np.abs((x_pred_org-x_test)/x_test)) * 100
    mape = mape[0]
    print(x_pred_org)
    print(f'MAPE: {mape}')

    x_pred_list = np.array(x_pred_list).reshape(-1, 3)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(x_pred_list[:,0])
    ax[0].axvline(x=x_test[0], color='r')    
    ax[0].set_title("{:.2f}".format(mape[0]))

    ax[1].hist(x_pred_list[:,1])
    ax[1].axvline(x=x_test[1], color='r')    
    ax[1].set_title("{:.2f}".format(mape[1]))

    ax[2].hist(x_pred_list[:,2])
    ax[2].axvline(x=x_test[2], color='r')  # Corrected line       
    ax[2].set_title("{:.2f}".format(mape[2]))

    plt.savefig(f'TWO_COMPARTMENT_{HIDDEN_DIM}_{LATENT_DIM}_{ALPHA}_PARAM_EST.png')