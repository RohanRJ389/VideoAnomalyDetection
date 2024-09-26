import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

# Preprocess Data: You can skip calculating deltas and use raw features
# Separate features (X) and labels (y)
numerical_cols = df.select_dtypes(include=[np.number]).drop(columns=['label']).columns
X = df[numerical_cols].values  # Raw features
y = df['label'].values  # Labels

# Reshape X for LSTM: (samples, timesteps, features)
# Assuming each video has multiple frames, we need to segment the data into sequences (e.g., 10 frames at a time)
sequence_length = 10  # Choose a sequence length (10 timesteps here)
n_features = X.shape[1]  # Number of features

def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length-1])  # Use the last frame in the sequence to predict the label
    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(X, y, sequence_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Define LSTM model architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),  # First LSTM layer
    tf.keras.layers.LSTM(32),  # Second LSTM layer
    tf.keras.layers.Dense(16, activation='relu'),  # Dense layer
    tf.keras.layers.Dropout(0.2),  # Dropout to avoid overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Early stopping
])

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred_label = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred_label))
print(classification_report(y_test, y_pred_label))

# Save the trained LSTM model
model_save_path = r'C:\My_Stuff\College\Capstone Project\CAPSTONE\Capstone_2\VideoAnomalyDetection\LSTM-model\lstm_model.h5'
model.save(model_save_path)
print(f'Model saved to {model_save_path}')
