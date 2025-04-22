import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SurvivalGNN(nn.Module):
    def __init__(self, in_channels, out_channels_time, out_channels_event, dropout_rate=0.3):
        super(SurvivalGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_time = nn.Linear(64, out_channels_time)
        self.fc_event = nn.Linear(64, out_channels_event)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        survival_time_output = self.fc_time(x)
        event_status_output = self.fc_event(x)

        return survival_time_output, event_status_output
