import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=8):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
   # * 2 because it's bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)  # * 2 for bidirectional
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)  # * 2 for bidirectional

        x = x.squeeze()
        x = torch.tensor(x)
        # x = x.permute(0,2,1)
        out, _ = self.lstm(x, (h0, c0))
        out  = out[:, -1, :]


        return out

