from model import LSTMVAE
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from train_vae import MultiSeriesODEDataset, SinusoidalDataset, pad_collate
import matplotlib.pyplot as plt
import argparse

    
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.input_size = input_size

        self.reg_model = nn.Sequential(
            nn.Linear(self.input_size, 256),
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
            nn.Linear(8, 4)
        )

    def forward(self, x):
        reg_model_output = self.reg_model(x)
        reg_model_output = reg_model_output.squeeze(0)
        return reg_model_output

def train_vi(model, reg_model, epochs, train_dataloader, val_dataloader, save_path = 'LV_EQUATION_REG.pt', device = 'cpu'):
    model.to(device)
    reg_model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    optimizer = optim.Adam(reg_model.parameters(), lr=1e-4)    
    criterion = nn.MSELoss()
    best_loss = 999999.

    for epoch in range(epochs):
        train_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y = y.mT

            y_hat, z, mu, logvar = model(y)
            x_hat = reg_model(z)
            loss = criterion(x_hat, x)

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
                y = y.mT
                y_hat, z, mu, logvar = model(y)
                x_hat = reg_model(z)
                loss = criterion(x_hat, x)

                val_loss += loss.item()

        print(f'Epoch {epoch}: Val Loss :{val_loss / len(val_dataloader)}')

        if val_loss/len(val_dataloader) < best_loss:
            best_loss = val_loss/len(val_dataloader)
            torch.save(reg_model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}.")

    print('Training completed.')

    return reg_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters for the model.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for the model')
    parser.add_argument('--input_size', type=int, default=2, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for the model')
    parser.add_argument('--latent_size', type=int, default=4, help='Latent size for the model')
    parser.add_argument('--model_name', type=str, default='LV_EQUATION_LSTM', help='Name of the model file')

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    INPUT_SIZE = args.input_size
    HIDDEN_SIZE = args.hidden_size
    LATENT_SIZE = args.latent_size
    MODEL_NAME = str(args.model_name) + '_' + str(HIDDEN_SIZE) + '_' + str(LATENT_SIZE) + '.pt'
    REG_MODEL_NAME = 'LV_EQUATION_BESTREG_' + str(HIDDEN_SIZE) + '_' + str(LATENT_SIZE) + '.pt'

    print('Data loading initialized.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    train_ds = SinusoidalDataset(root = 'data', suffix = 'train/processed')
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    val_ds = SinusoidalDataset(root = 'data', suffix = 'val/processed')
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    test_ds = SinusoidalDataset(root = 'data', suffix = 'test/processed')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_collate)

    model = LSTMVAE(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, latent_size = LATENT_SIZE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
    model.to(device)

    # Uncomment the following lines to train the regression model
    # reg_model = RidgeRegression(input_dim=LATENT_SIZE, output_dim=4)
    # reg_model = RegressionModel(input_size=LATENT_SIZE)
    # reg_model.to(device)
    # reg_model = train_vi(model, reg_model, epochs = 1000, save_path = REG_MODEL_NAME, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device)
    
    reg_model = RegressionModel(input_size=LATENT_SIZE)
    reg_model.to(device)
    reg_model.load_state_dict(torch.load(REG_MODEL_NAME, map_location=device))

    test_loss = []
    test_mae_a = []
    test_mae_b = []
    test_mae_c = []
    test_mae_d = [] 
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            a_pred, b_pred, c_pred, d_pred = [], [], [], []
            a_true, b_true, c_true, d_true = [], [], [], []
            x, y = batch
            x, y = x.to(device), y.to(device)
            y = y.mT

            batch_a_pred, batch_b_pred, batch_c_pred, batch_d_pred = [], [], [], []

            for j in range(2000):
                y_hat, z, mu, logvar = model(y)
                x_hat = reg_model(z)
            
                batch_a_pred.extend(x_hat[:, 0].detach().cpu().numpy().tolist())
                batch_b_pred.extend(x_hat[:, 1].detach().cpu().numpy().tolist())
                batch_c_pred.extend(x_hat[:, 2].detach().cpu().numpy().tolist())
                batch_d_pred.extend(x_hat[:, 3].detach().cpu().numpy().tolist())

            a_pred.append(np.array(batch_a_pred))
            b_pred.append(np.array(batch_b_pred))
            c_pred.append(np.array(batch_c_pred))
            d_pred.append(np.array(batch_d_pred))

            a_true.extend(x[:, 0].detach().cpu().numpy().tolist())
            b_true.extend(x[:, 1].detach().cpu().numpy().tolist())
            c_true.extend(x[:, 2].detach().cpu().numpy().tolist())
            d_true.extend(x[:, 3].detach().cpu().numpy().tolist())

            loss = F.mse_loss(x_hat, x)
            test_loss.append(loss.item())

            fig, axs = plt.subplots(2, 2)

            # Subplot for a_pred
            axs[0, 0].hist(np.array(a_pred).flatten(), alpha=0.5, label='a_pred')
            axs[0, 0].axvline(np.array(a_true).flatten(), linestyle='--', label='a_true', color = 'r')
            # Subplot for b_pred
            axs[0, 1].hist(np.array(b_pred).flatten(), alpha=0.5, label='b_pred')
            axs[0, 1].axvline(np.array(b_true).flatten(), linestyle='--', label='b_true', color = 'r')
            # Subplot for c_pred
            # Assuming c_pred is defined somewhere above this code
            axs[1, 0].hist(np.array(c_pred).flatten(), alpha=0.5, label='c_pred')
            axs[1, 0].axvline(np.array(c_true).flatten(), linestyle='--', label='c_true', color = 'r')

            # Subplot for d_pred
            # Assuming d_pred is defined somewhere above this code
            axs[1, 1].hist(np.array(d_pred).flatten(), alpha=0.5, label='d_pred')
            axs[1, 1].axvline(np.array(d_true).flatten(), linestyle='--', label='d_true', color = 'r')

            plt.savefig(f'reconstruction/{MODEL_NAME}/BESTREG_example_{i}.png')

            print('Mean Estimated Values: ', np.mean(a_pred), np.mean(b_pred), np.mean(c_pred), np.mean(d_pred))
            print('True Values: ', np.mean(a_true), np.mean(b_true), np.mean(c_true), np.mean(d_true))

            relative_error_a = (np.mean(a_pred) - a_true)/a_true * 100
            relative_error_b = (np.mean(b_pred) - b_true)/b_true * 100
            relative_error_c = (np.mean(c_pred) - c_true)/c_true * 100
            relative_error_d = (np.mean(d_pred) - d_true)/d_true * 100

            print('Relative Error a: ', np.mean(relative_error_a))
            print('Relative Error b: ', np.mean(relative_error_b))
            print('Relative Error c: ', np.mean(relative_error_c))
            print('Relative Error d: ', np.mean(relative_error_d))
            test_mae_a.append(relative_error_a)
            test_mae_b.append(relative_error_b)
            test_mae_c.append(relative_error_c)
            test_mae_d.append(relative_error_d)
        print(f'Test Loss :{np.mean(test_loss)}')
        print(f'Test MAE a :{np.mean(np.abs(test_mae_a))}')
        print(f'Test MAE b :{np.mean(np.abs(test_mae_b))}')
        print(f'Test MAE c :{np.mean(np.abs(test_mae_c))}')
        print(f'Test MAE d :{np.mean(np.abs(test_mae_d))}')
        print(f'Test MAE :{np.mean(np.abs(test_mae_a) + np.abs(test_mae_b) + np.abs(test_mae_c) + np.abs(test_mae_d))}')
