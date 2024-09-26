import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# import numpy as np


# Custom Dataset to load CSV files
class VideoDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths  # List of CSV file paths
        self.labels = labels  # List of labels (1 for anomalous, 0 for normal)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load features from CSV file
        features = pd.read_csv(self.file_paths[idx]).values[
            :, 1:
        ]  # Skipping the Frame Number
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


# Attention mechanism (optional)
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(
            hidden_size, 1
        )  # Linear layer to compute attention scores

    def forward(self, lstm_output):
        attn_weights = torch.softmax(
            self.attn(lstm_output), dim=1
        )  # Softmax over the time dimension
        context = torch.sum(
            attn_weights * lstm_output, dim=1
        )  # Weighted sum of the LSTM output
        return context


# RNN-based Model with LSTM/GRU and optional attention
class AnomalyDetectionModel(nn.Module):
    def __init__(self, num_features, hidden_size, use_attention=False, rnn_type="LSTM"):
        super(AnomalyDetectionModel, self).__init__()
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.hidden_size = hidden_size

        # Define LSTM or GRU layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(num_features, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(num_features, hidden_size, batch_first=True)

        # Attention layer (optional)
        if use_attention:
            self.attention = Attention(hidden_size)
        else:
            self.global_pool = nn.AdaptiveMaxPool1d(
                1
            )  # Max pooling over time dimension

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        if self.rnn_type == "LSTM":
            rnn_out, (hn, cn) = self.rnn(x)  # LSTM
        else:
            rnn_out, hn = self.rnn(x)  # GRU

        if self.use_attention:
            # Use attention to get context vector
            context = self.attention(rnn_out)
        else:
            # Apply max pooling over time dimension
            context = self.global_pool(rnn_out.permute(0, 2, 1)).squeeze(2)

        # Fully connected layer for classification
        out = self.fc(context)
        out = self.sigmoid(out)  # Sigmoid for binary classification
        return out


# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.BCELoss()  # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy}%")


# Load your CSV files and labels
train_files = ["path_to_train_csv_1.csv", "path_to_train_csv_2.csv", ...]
train_labels = [
    0,
    1,
    ...,
]  # Corresponding labels for normal (0) and anomalous (1) videos

val_files = ["path_to_val_csv_1.csv", "path_to_val_csv_2.csv", ...]
val_labels = [0, 1, ...]

# Create DataLoader
batch_size = 32
train_dataset = VideoDataset(train_files, train_labels)
val_dataset = VideoDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model Parameters
num_features = 16  # Number of features in your CSVs
hidden_size = 128
use_attention = True  # Set to False if you don't want attention
rnn_type = "LSTM"  # Can switch to "GRU" if desired

# Initialize and train the model
model = AnomalyDetectionModel(
    num_features=num_features,
    hidden_size=hidden_size,
    use_attention=use_attention,
    rnn_type=rnn_type,
)
train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001)
