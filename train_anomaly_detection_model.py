"""
train_anomaly_detection_model.py

This script implements the definitive, production-oriented solution by re-framing 
the problem as anomaly detection (Safe vs. Aggressive).

Workflow:
1.  Loads and prepares data.
2.  Merges 'SLOW' and 'NORMAL' labels into a single 'SAFE' class (0), 
    and keeps 'AGGRESSIVE' as the anomaly class (1).
3.  Calculates class weights to handle the new class imbalance.
4.  Uses K-Fold Cross-Validation and Optuna to find the best hyperparameters 
    for this binary classification task.
5.  Trains a final, optimized CNN-LSTM-Attention model.
6.  Evaluates the model's ability to distinguish 'SAFE' from 'AGGRESSIVE' driving.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from scipy.stats import mode
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# --- 1. Data Preparation & Augmentation ---

def create_windows(scaled_data, labels, window_size, step_size):
    """Creates windows from the continuous time-series data."""
    X, y = [], []
    for i in range(0, len(scaled_data) - window_size, step_size):
        window_features = scaled_data[i: i + window_size]
        window_label = mode(labels[i: i + window_size])[0][0]
        X.append(window_features)
        y.append(window_label)
    return np.array(X), np.array(y)

class DrivingDataset(Dataset):
    """Custom PyTorch Dataset with on-the-fly data augmentation."""
    def __init__(self, X, y, is_train=False, augmentation_prob=0.5):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_item = self.X[idx]
        if self.is_train and np.random.rand() < self.augmentation_prob:
            jitter = torch.randn_like(x_item) * 0.05
            x_item += jitter
            scaling_factor = np.random.uniform(0.9, 1.1)
            x_item *= scaling_factor
        return x_item, self.y[idx]

# --- 2. SOTA Model Architecture with Attention ---

class Attention(nn.Module):
    """Attention mechanism layer."""
    def __init__(self, feature_dim, step_dim):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.zeros(feature_dim, 1))
        nn.init.xavier_uniform_(self.weight)
        self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x):
        eij = torch.mm(x.contiguous().view(-1, x.size(-1)), self.weight).view(-1, x.size(1)) + self.b
        a = torch.softmax(eij, dim=1)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class CNNLSTMAttention(nn.Module):
    """The definitive SOTA model combining CNN, LSTM, and Attention."""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, cnn_filters, dropout_rate):
        super(CNNLSTMAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # The sequence length changes after pooling
        seq_len_after_pooling = 10 
        self.attention = Attention(hidden_dim*2, seq_len_after_pooling)
        
        self.fc1 = nn.Linear(hidden_dim*2, 100)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        
        x = self.relu2(self.fc1(attn_out))
        x = self.dropout(x)
        # For binary classification with CrossEntropyLoss, output size should be 2
        x = self.fc2(x)
        return x

# --- 3. Optuna Objective Function ---

def objective(trial, X, y, class_weights):
    """Optuna objective function for the binary classification task."""
    params = {
        'cnn_filters': trial.suggest_categorical('cnn_filters', [64, 128]),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 128),
        'num_lstm_layers': trial.suggest_int('num_lstm_layers', 1, 2),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.7),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['AdamW', 'RMSprop'])
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = DrivingDataset(X_train, y_train, is_train=True)
        val_dataset = DrivingDataset(X_val, y_val, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = CNNLSTMAttention(
            input_dim=X.shape[2], hidden_dim=params['hidden_dim'],
            num_layers=params['num_lstm_layers'], num_classes=2, # Binary classification
            cnn_filters=params['cnn_filters'], dropout_rate=params['dropout_rate']
        ).to(device)
        
        optimizer = getattr(optim, params['optimizer_name'])(model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=False)

        for epoch in range(20):
            model.train()
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss / len(train_loader))

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                fold_preds.extend(predicted.cpu().numpy())
        scores.append(accuracy_score(y_val, fold_preds))

    return np.mean(scores)

# --- 4. Main Execution ---

if __name__ == '__main__':
    print("--- Starting Anomaly Detection Training Pipeline ---")
    
    # Load and prepare data
    df = pd.read_csv('train_motion_data.csv')
    column_mapping = {
        'AccX': 'acceleration_x', 'AccY': 'acceleration_y', 'AccZ': 'acceleration_z',
        'GyroX': 'gyro_x', 'GyroY': 'gyro_y', 'GyroZ': 'gyro_z',
        'Class': 'label', 'Timestamp': 'timestamp'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # *** RE-FRAME THE PROBLEM: Binary Classification ***
    df['label'] = df['label'].replace({'NORMAL': 'SAFE', 'SLOW': 'SAFE'})
    print("Re-framed problem to binary classification: SAFE vs. AGGRESSIVE")

    feature_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    window_size, step_size = 20, 10
    X, y = create_windows(df[feature_cols].values, df['label'].values, window_size, step_size)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Calculate class weights for the new binary problem
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    print(f"Calculated Class Weights: {le.classes_} -> {class_weights}")
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Run Optuna study
    print("\nStarting Optuna search for anomaly detection model...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full, class_weights), n_trials=30) # Fewer trials needed for simpler problem
    
    print("\nOptuna search finished.")
    print(f"Best average cross-validation accuracy: {study.best_value:.4f}")
    best_params = study.best_params
    print("Best hyperparameters: ", best_params)
        
    # Train final model
    print("\n--- Training Final Anomaly Detection Model ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_train_dataset = DrivingDataset(X_train_full, y_train_full, is_train=True)
    final_train_loader = DataLoader(final_train_dataset, batch_size=32, shuffle=True)
    
    final_model = CNNLSTMAttention(
        input_dim=X.shape[2], hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_lstm_layers'], num_classes=2, # Binary
        cnn_filters=best_params['cnn_filters'], dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    optimizer = getattr(optim, best_params['optimizer_name'])(final_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    
    for epoch in range(50):
        final_model.train()
        for inputs, labels in final_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Final Model Training - Epoch [{epoch+1}/50]")

    # Final Evaluation
    print("\n--- Final Model Evaluation on Unseen Test Set ---")
    test_dataset = DrivingDataset(X_test, y_test, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    final_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = final_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Final Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Save final artifacts
    print("\nSaving final optimized model and artifacts...")
    artifacts = {
        'model_state_dict': final_model.state_dict(), 'scaler': scaler, 'label_encoder': le,
        'model_params': {
            'input_dim': X.shape[2], 'hidden_dim': best_params['hidden_dim'],
            'num_layers': best_params['num_lstm_layers'], 'num_classes': 2,
            'cnn_filters': best_params['cnn_filters'], 'dropout_rate': best_params['dropout_rate']
        }
    }
    joblib.dump(artifacts, 'sota_pytorch_anomaly_artifacts.pkl')
    print("--- Anomaly Detection Training Pipeline Complete ---")
