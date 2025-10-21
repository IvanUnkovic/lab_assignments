import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, input_size=300, hidden_size=150, output_size=1, num_layers=2, rnn_type='LSTM', dropout=0.0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        self.rnn_type = rnn_type
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        self.fc1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.to(torch.long)
        x = self.embedding(x).to(torch.float32)

        if self.rnn_type=='LSTM':
            out, _ = self.lstm(x)
        elif self.rnn_type=='GRU':
            out, _ = self.gru(x)
        else:
            out, _ = self.rnn(x)
        
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        
        return out