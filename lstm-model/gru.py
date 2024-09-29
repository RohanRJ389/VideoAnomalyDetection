import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
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

# Preprocess data
features = [
    'Number of Objects', 'Spatial Density', 'Class Distribution',
    'Mean Velocity', 'Max Velocity', 'Variance in Velocity',
    'Mean Acceleration', 'Max Acceleration', 'Variance in Acceleration',
    'Mean Direction', 'Direction Variance', 'Mean Displacement',
    'Interaction Count', 'Mean IoU'
]

normal_data = normal_data[features]
anomalous_data = anomalous_data[features]

# Process the 'Class Distribution' feature
normal_data['Class Distribution'] = normal_data['Class Distribution'].apply(
    lambda x: np.array(list(map(float, x.split(','))))
)
anomalous_data['Class Distribution'] = anomalous_data['Class Distribution'].apply(
    lambda x: np.array(list(map(float, x.split(','))))
)

# Preprocess numerical features
scaler = MinMaxScaler()

numerical_cols = [
    'Number of Objects', 'Spatial Density', 'Mean Velocity',
    'Max Velocity', 'Variance in Velocity', 'Mean Acceleration',
    'Max Acceleration', 'Variance in Acceleration',
    'Mean Direction', 'Direction Variance', 'Mean Displacement',
    'Interaction Count', 'Mean IoU'
]

normal_data[numerical_cols] = scaler.fit_transform(normal_data[numerical_cols])
anomalous_data[numerical_cols] = scaler.transform(anomalous_data[numerical_cols])

# Combine data and create labels
normal_data['label'] = 0
anomalous_data['label'] = 1
df = pd.concat([normal_data, anomalous_data], ignore_index=True)

# Select features (excluding label)
numerical_cols = df.select_dtypes(include=[np.number]).drop(columns=['label']).columns
X = df[numerical_cols].values
y = df['label'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Create an instance of the model
input_size = len(numerical_cols)
hidden_size = 128
num_layers = 2
num_classes = 2
model = GRUModel(input_size, hidden_size, num_layers, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Test the model
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test, predicted.cpu().numpy())
    print(f'Test Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, predicted.cpu().numpy()))