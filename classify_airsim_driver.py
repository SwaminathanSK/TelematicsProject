"""
classify_anomaly_airsim.py

This script uses the trained SOTA anomaly detection model to classify live 
driving from AirSim as either 'SAFE' or 'AGGRESSIVE'.

Workflow:
1.  Loads the trained PyTorch model and pre-processing artifacts.
2.  Connects to AirSim and collects sensor data for a short duration.
3.  Processes the collected data:
    - Renames columns to match training data.
    - Scales the data using the loaded scaler.
    - Formats the data into the correct shape for the model.
4.  Feeds the data into the model to get a prediction.
5.  Displays the final classification result and confidence scores.
"""
import pandas as pd
import numpy as np
import time
import joblib
import warnings

import torch
import torch.nn as nn

from enhanced_collector import EnhancedTelematicsCollector

warnings.filterwarnings('ignore')

# --- 1. Model Architecture Definition ---
# This MUST be identical to the architecture in the training script.

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
        x = self.fc2(x)
        return x

# --- 2. Main Classification Function ---

def classify_live_driving(collection_duration=20):
    """Main function to collect data, process it, and classify it."""
    print("--- Live Anomaly Detection using AirSim ---")
    
    # 1. Load Trained Artifacts
    try:
        # NOTE: Ensure you are loading the correct artifacts file
        artifacts = joblib.load('sota_pytorch_anomaly_artifacts.pkl')
        model_state_dict = artifacts['model_state_dict']
        model_params = artifacts['model_params']
        scaler = artifacts['scaler']
        le = artifacts['label_encoder']
        print("Successfully loaded trained anomaly detection model and artifacts.")
    except FileNotFoundError:
        print("Error: 'sota_pytorch_anomaly_artifacts.pkl' not found.")
        print("Please run the final anomaly detection training script first.")
        return

    # 2. Reconstruct the Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMAttention(**model_params).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Model reconstructed successfully.")

    # 3. Collect Data from AirSim
    print(f"\nStarting data collection from AirSim for {collection_duration} seconds.")
    
    collector = EnhancedTelematicsCollector()
    collector.client.enableApiControl(False)
    print("\nManual control enabled! Please drive in the AirSim window.")
    
    collector.start_enhanced_collection(interval=0.05) # Higher frequency collection
    
    for i in range(collection_duration, 0, -1):
        print(f"  Collecting data... {i} seconds remaining.", end='\r')
        time.sleep(1)
        
    collector.stop_collection()
    collector.restore_manual_control()
    
    airsim_df = collector.get_enhanced_dataframe()
    
    # We need at least 20 data points for a window
    if len(airsim_df) < 20:
        print(f"\nNot enough data collected ({len(airsim_df)} points). Need at least 20. Please try again.")
        return
    
    # Use the last 20 points for the window. Use .copy() to avoid pandas warnings.
    airsim_window_df = airsim_df.tail(20).copy()
    print(f"\nCollected {len(airsim_df)} points. Using the last 20 for classification.")

    # 4. Process the AirSim Data Window
    # *** FIX: Rename AirSim columns to match the training data columns ***
    airsim_window_df.rename(columns={
        'angular_velocity_x': 'gyro_x',
        'angular_velocity_y': 'gyro_y',
        'angular_velocity_z': 'gyro_z'
    }, inplace=True)
    
    feature_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Scale the data using the loaded scaler
    airsim_window_df[feature_cols] = scaler.transform(airsim_window_df[feature_cols])
    
    # Convert to tensor and add batch dimension
    X_live = torch.tensor(airsim_window_df[feature_cols].values, dtype=torch.float32).unsqueeze(0)
    X_live = X_live.to(device)

    # 5. Predict
    with torch.no_grad():
        outputs = model(X_live)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs.data, 1)
        
    predicted_label = le.inverse_transform(predicted_idx.cpu().numpy())[0]
    
    # 6. Display Results
    print("\n" + "="*35)
    print("--- ANOMALY DETECTION RESULT ---")
    print(f"      Detected Driving Style: {predicted_label.upper()}")
    print("="*35 + "\n")
    
    print("Prediction Confidence:")
    for i, class_name in enumerate(le.classes_):
        print(f"  - {class_name}: {probabilities[0][i]:.2%}")

if __name__ == '__main__':
    classify_live_driving()
