import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            bidirectional=False)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        return lstm_out, (hidden, cell)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.num_layers, self.hidden_size),
                torch.zeros(batch_size, self.num_layers, self.hidden_size))
    
class lstm_decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = True,
                            bidirectional=False)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)
    
class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers = 1):
        super(LSTMVAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = lstm_encoder(input_size=input_size, hidden_size = hidden_size, num_layers=num_layers)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder = lstm_decoder(input_size=latent_size, output_size = input_size, hidden_size=hidden_size, num_layers=num_layers)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        _, (hidden, cell) = self.encoder(x)
        return hidden, cell
    
    def decode(self, z, hidden):
        x_hat, (hidden, cell) = self.decoder(z, hidden)
        return x_hat, (hidden, cell)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden, cell = self.encode(x)

        mu = self.mean(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(mu, logvar)

        h_ = self.decoder_input(z)

        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size)

        # initialize hidden state
        hidden = (h_.contiguous(), h_.contiguous())
        x_hat, _ = self.decode(z, hidden)

        # calculate vae loss
        losses = self.loss_function(x_hat, x, mu, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return m_loss, x_hat, (recon_loss, kld_loss)
    
    def loss_function(self, recon_x, x, mu, logvar):
        reconstruction_loss = F.mse_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_weight = 0.00025 # Account for the minibatch samples from the dataset
        #kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = reconstruction_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss.detach(), 'KLD':-kld_loss.detach()}


class FFNN(nn.module):
    def __init__(self, input_size = 32, output_size = 2):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.module = nn.Sequential(
            nn.Linear(self.input_size, self.input_size//2),
            nn.ReLU(),
            nn.Linear(self.input_size//2, self.input_size//4),
            nn.ReLU(),
            nn.Linear(self.input_size//4, self.input_size//8),
            nn.ReLU(),
            nn.Linear(self.input_size//8, self.output_size),
        )
    
    def forward(self, x):
        return self.module(x)

class VI(nn.module):
    def __init__(self, input_size = 32, output_size = 2):
        super(VI, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.vae = LSTMVAE(input_size = self.input_size, output_size = self.output_size)
        self.vae.load_state_dict(torch.load('model.pth'))

        for param in self.vae.parameters():
            param.requires_grad = False

        self.ffnn = FFNN(input_size = self.input_size, output_size = self.output_size)

    def forward(self, x):
        hidden, cell = self.vae.encode(x)
        mu = self.vae.mean(hidden)
        logvar = self.vae.logvar(hidden)
        z = self.vae.reparameterize(mu, logvar)

        beta = self.ffnn(z)

        return beta

