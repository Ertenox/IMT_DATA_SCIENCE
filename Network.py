from torch import nn

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=13, num_layers=2, dropout_rate=0.5):
        super(Network, self).__init__()
        
        self.bi_lstm = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.intermediate = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        lstm_out, _ = self.bi_lstm(input)
        last_hidden = lstm_out[:, -1, :]  # Dernier pas de temps
        x = self.dropout(last_hidden)
        x = self.relu(self.intermediate(x))
        logits = self.fc(x)
        return logits  # PAS de softmax ici !
