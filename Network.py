from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    """Réseau neuronal de The Classification of Multiple Interacting Gait Abnormalities Using Insole Sensors and Machine Learning"""

    def __init__(self, input_size, hidden_size, num_classes=12, num_layers=1, dropout_rate=0.5):
        super(Network, self).__init__()
        
        self.bi_lstm = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input):
        lstm_out, _ = self.bi_lstm(input)  
        last_hidden = lstm_out[:, -1, :] 
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)         # (batch_size, num_classes)
        probs = F.softmax(logits, dim=1)  # Convertir en probabilités sur les 12 classes
        
        return probs