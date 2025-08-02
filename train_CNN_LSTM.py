"""
train_sota_model_pytorch.py

This script implements a State-of-the-Art (SOTA) CNN-LSTM model in PyTorch
for driving behavior classification.

Workflow:
1.  Loads 'train_motion_data.csv'.
2.  Prepares the data:
    - Renames columns.
    - Scales the raw sensor data.
    - Creates windows of raw signal data.
3.  Defines a custom PyTorch Dataset and DataLoader.
4.  Builds a hybrid CNN-LSTM model using nn.Module.
5.  Implements a full training and evaluation loop.
6.  Trains the model, evaluating on a test set.
7.  Saves the trained PyTorch model (.pth) and the pre-processing objects.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# --- 1. Data Preparation ---

def create_windows(scaled_data, labels, window_size, step_size):
    """
    Creates windows from the continuous time-series data.
    Returns data in the shape (num_windows, window_size, num_features)
    and corresponding labels.
    """
    X, y = [], []
    for i in range(0, len(scaled_data) - window_size, step_size):
        window_features = scaled_data[i: i + window_size]
        window_label = mode(labels[i: i + window_size])[0][0]
        X.append(window_features)
        y.append(window_label)
    return np.array(X), np.array(y)

class DrivingDataset(Dataset):
    """Custom PyTorch Dataset for driving data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. Model Architecture ---

class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM Model."""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.5)
        
        # LSTM layer
        # The input to the LSTM will be the output of the CNN layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=0.5)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # PyTorch Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        
        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # We only need the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        x = self.fc1(last_time_step_out)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

# --- 3. Training and Evaluation ---

def train_sota_model(csv_path='train_motion_data.csv'):
    """Main function to load data, build, train, and save the SOTA model."""
    print("--- Starting SOTA Driver Classification Model Training (PyTorch) ---")

    # Load and prepare data
    df = pd.read_csv(csv_path)
    column_mapping = {
        'AccX': 'acceleration_x', 'AccY': 'acceleration_y', 'AccZ': 'acceleration_z',
        'GyroX': 'gyro_x', 'GyroY': 'gyro_y', 'GyroZ': 'gyro_z',
        'Class': 'label', 'Timestamp': 'timestamp'
    }
    df.rename(columns=column_mapping, inplace=True)
    feature_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Loaded, renamed, and scaled data.")

    # Create windows
    window_size = 20
    step_size = 10
    X, y = create_windows(df[feature_cols].values, df['label'].values, window_size, step_size)
    print(f"Created {len(X)} windows. Data shape: {X.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Create DataLoaders
    train_dataset = DrivingDataset(X_train, y_train)
    test_dataset = DrivingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"DataLoaders created. Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CNNLSTM(input_dim=X_train.shape[2], hidden_dim=100, num_layers=1, num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 100
    print("\nStarting model training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    print("\n--- Model Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=le.classes_, columns=le.classes_))

    # Save artifacts
    print("\nSaving PyTorch model and supporting artifacts...")
    artifacts = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': le,
        'model_params': {
            'input_dim': X_train.shape[2],
            'hidden_dim': 100,
            'num_layers': 1,
            'num_classes': len(le.classes_)
        }
    }
    joblib.dump(artifacts, 'sota_pytorch_artifacts.pkl')
    
    print("\n--- Training Complete ---")
    print("Saved artifacts to 'sota_pytorch_artifacts.pkl'.")

if __name__ == '__main__':
    train_sota_model()
