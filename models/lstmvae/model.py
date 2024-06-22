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
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hidden, cell = self.encode(x)

        mu = self.mean(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(mu, logvar)
        h_ = self.decoder_input(z)
        
        z_copy = z
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size)

        # initialize hidden state
        hidden = (h_.contiguous(), h_.contiguous())
        x_hat, _ = self.decode(z, hidden)

        return x_hat, z_copy, mu, logvar
    