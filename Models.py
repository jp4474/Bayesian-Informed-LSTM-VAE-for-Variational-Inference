import torch
import torch.nn as nn

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
                            batch_first = True, # (batch_dim, seq_dim, input_dim)
                            bidirectional=True)
        
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
        
    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x)
        prediction = self.fc(output)
        return prediction, (hidden, cell)
    