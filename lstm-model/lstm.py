import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import glob

# Load Normal and Anomalous Data
normal_features_path = r'F:\VideoAnomalyDetection\Data\extracted-features\Normal\*.csv'
anomalous_features_path = r'F:\VideoAnomalyDetection\Data\extracted-features\Anomaly\*.csv'

# Load normal features
normal_files = glob.glob(normal_features_path)
normal_dfs = [pd.read_csv(file) for file in normal_files]
normal_data = pd.concat(normal_dfs, ignore_index=True)

# Load anomalous features
anomalous_files = glob.glob(anomalous_features_path)
anomalous_dfs = [pd.read_csv(file) for file in anomalous_files]
anomalous_data = pd.concat(anomalous_dfs, ignore_index=True)

# Combine data and create labels
normal_data['label'] = 0
anomalous_data['label'] = 1
df = pd.concat([normal_data, anomalous_data], ignore_index=True)

# Select features (excluding label)
numerical_cols = df.select_dtypes(include=[np.number]).drop(columns=['label']).columns
X = df[numerical_cols].values
y = df['label'].values

# 2. Prepare Data for LSTM (Split and Reshape into Sequences)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to have the shape [samples, time steps, features]
# Here we'll assume a sliding window approach with a window size of 10 frames per sequence
sequence_length = 10

def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
        labels.append(y[i + seq_length])  # The label corresponds to the last frame in the sequence
    return np.array(sequences), np.array(labels)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# 3. Create Dataset and DataLoader
class VideoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

train_dataset = VideoDataset(X_train_seq, y_train_seq)
test_dataset = VideoDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = X_train_seq.shape[2]  # Number of features
hidden_size = 64  # Number of LSTM units
num_layers = 1  # Single LSTM layer
output_size = 2  # Binary classification

# 5. Model, Loss Function, and Optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 7. Evaluate the Model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 8. Print Accuracy and Classification Report
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomalous']))
