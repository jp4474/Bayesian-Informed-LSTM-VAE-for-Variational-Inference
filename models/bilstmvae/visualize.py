from model import LSTMVAE
import torch
from torch.utils.data import DataLoader, Dataset
from train_vae import MultiSeriesODEDataset, SinusoidalDataset, pad_collate
import matplotlib.pyplot as plt
import umap
import pandas as pd
import numpy as np
import argparse
import os


def save_latent_space(model, model_name, suffix, batch_size, latent_size, device):
    
    if not os.path.exists(f'data/{suffix}/{model_name}'):
        os.makedirs(f'data/{suffix}/{model_name}')
    
    ds = SinusoidalDataset(root = 'data', suffix = f'{suffix}/processed')
    ds_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    # Uncomment to save latent vectors
    for i, batch in enumerate(ds_dataloader):
        x, y = batch
        y = y.to(device)
        y = y.mT

        y_hat, z, mu, logvar = model(y)

        z = z.squeeze(0).detach().cpu().numpy()
        mu = mu.squeeze(0).detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        
        with open(f'data/{suffix}/{model_name}/latent_{i}.npy', 'wb') as f:
            np.save(f, z)

    print('Done!')

    # Load latent vectors
    z = []

    MAX = len(ds_dataloader)
    for i in range(MAX):
        if i != MAX - 1:
            with open(f'data/{suffix}/{model_name}/latent_{i}.npy', 'rb') as f:
                loaded_np = np.load(f)
                z.append(loaded_np)

    z = np.array(z).reshape(-1, latent_size)

    with open(f'data/{suffix}/{model_name}/latent_{MAX-1}.npy', 'rb') as f:
        loaded_np = np.load(f).reshape(-1, latent_size)

    z = np.concatenate((z, loaded_np), axis=0)

    df = pd.DataFrame(z)
    df.to_csv(f'data/{suffix}/{model_name}/latent_{suffix}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of the hidden layer')
    parser.add_argument('--latent_size', type=int, default=4, help='Size of the latent layer')
    parser.add_argument('--model_name', type=str, default='LV_EQUATION_LSTM', help='Name of the model file')
    args = parser.parse_args()

    BATCH_SIZE = 1024
    INPUT_SIZE = 2
    HIDDEN_SIZE = args.hidden_size
    LATENT_SIZE = args.latent_size
    MODEL_NAME = str(args.model_name) + '_' + str(HIDDEN_SIZE) + '_' + str(LATENT_SIZE) + '.pt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LSTMVAE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    model.eval()

    save_latent_space(model, MODEL_NAME, 'train', BATCH_SIZE, LATENT_SIZE, DEVICE)
    save_latent_space(model, MODEL_NAME, 'val', BATCH_SIZE, LATENT_SIZE, DEVICE)
    save_latent_space(model, MODEL_NAME, 'test', BATCH_SIZE, LATENT_SIZE, DEVICE)

    train = pd.read_csv(f'data/train/{MODEL_NAME}/latent_train.csv')
    val = pd.read_csv(f'data/val/{MODEL_NAME}/latent_val.csv')
    test = pd.read_csv(f'data/test/{MODEL_NAME}/latent_test.csv')

    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(train)
    val_embedding = trans.transform(val)
    test_embedding = trans.transform(test)

    # Define the colors of the rainbow
    #colors = ['red', 'red', 'red' , 'red' ,'red', 'blue', 'red', 'red', 'blue', 'green']

    plt.figure(figsize=(10, 10))
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s=2, c='yellow', label='train')
    plt.scatter(val_embedding[:, 0], val_embedding[:, 1], s=5, c='blue', label='val')
    plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s=10, c='red', label='test')
    plt.legend()
    plt.title('Embedding of Latent Vectors', fontsize=24)
    plt.savefig(f'reconstruction/{MODEL_NAME}/latent_space.png')