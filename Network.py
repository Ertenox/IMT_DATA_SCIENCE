from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    """Réseau neuronal de The Classification of Multiple Interacting Gait Abnormalities Using Insole Sensors and Machine Learning"""

    def __init__(self, input_size, hidden_size, num_classes=13, num_layers=1, dropout_rate=0.5):
        super(Network, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)
        

        
        self.dropout = nn.Dropout(dropout_rate)
        self.intermediate_size = nn.Linear(hidden_size , hidden_size *2)  
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input):
        x = input.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x) 
        last_hidden = lstm_out[:, -1, :] 
        dropped = self.dropout(last_hidden)
        x = self.relu(self.intermediate_size(dropped))
        logits = self.fc(x)
        return logits