"""
train_driver_model.py (Improved)

This script trains a more advanced model by engineering sophisticated features
to better distinguish between driving styles.

Workflow:
1.  Loads the 'train_motion_data.csv'.
2.  Renames columns to a standard format.
3.  Segments the data into windows.
4.  Engineers advanced features for each window, including:
    - Basic statistics (mean, std, max)
    - Frequency domain features (FFT)
    - Zero-crossing rate
5.  Trains a RandomForestClassifier and evaluates its performance.
6.  Saves the improved model and artifacts.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
from scipy.stats import mode

warnings.filterwarnings('ignore')

def create_windows(df, window_size_seconds=10, step_size_seconds=5, sampling_rate_hz=2):
    """Creates windows from the continuous time-series data."""
    print(f"Creating {window_size_seconds}-second windows from continuous data...")
    window_size = window_size_seconds * sampling_rate_hz
    step_size = step_size_seconds * sampling_rate_hz
    windows, labels = [], []
    
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i: i + window_size]
        window_label = mode(window['label'])[0][0]
        windows.append(window)
        labels.append(window_label)
        
    print(f"Created {len(windows)} windows (data samples).")
    return windows, labels

def engineer_features_from_windows(windows):
    """Engineers a rich set of features for each window."""
    print("Engineering advanced features for each window...")
    feature_list = []
    
    for window in windows:
        features = {}
        
        # Calculate magnitudes
        window['accel_mag'] = np.sqrt(window['acceleration_x']**2 + window['acceleration_y']**2 + window['acceleration_z']**2)
        window['gyro_mag'] = np.sqrt(window['gyro_x']**2 + window['gyro_y']**2 + window['gyro_z']**2)

        # Process each signal type (accel_mag, gyro_mag, etc.)
        for signal in ['accel_mag', 'gyro_mag', 'acceleration_x', 'acceleration_y', 'acceleration_z']:
            signal_data = window[signal]
            
            # Basic Stats
            features[f'{signal}_mean'] = signal_data.mean()
            features[f'{signal}_std'] = signal_data.std()
            features[f'{signal}_max'] = signal_data.max()
            features[f'{signal}_min'] = signal_data.min()
            features[f'{signal}_range'] = features[f'{signal}_max'] - features[f'{signal}_min']
            features[f'{signal}_quantile'] = signal_data.quantile(0.5)

            # Zero Crossing Rate
            features[f'{signal}_zcr'] = np.sum(np.diff(np.sign(signal_data - signal_data.mean())) != 0) / len(signal_data)
            
            # Frequency Domain Features (FFT)
            fft_vals = np.fft.rfft(signal_data.values)
            fft_abs = np.abs(fft_vals)
            features[f'{signal}_fft_mean'] = np.mean(fft_abs)
            features[f'{signal}_fft_std'] = np.std(fft_abs)
            features[f'{signal}_fft_max'] = np.max(fft_abs)
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)


def train_model(csv_path='train_motion_data.csv'):
    """Main function to load data, train the model, and save artifacts."""
    print("--- Starting Improved Driver Classification Model Training ---")
    
    # Load Data
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded '{csv_path}' with {len(df)} rows.")

    # Rename Columns
    column_mapping = {
        'AccX': 'acceleration_x', 'AccY': 'acceleration_y', 'AccZ': 'acceleration_z',
        'GyroX': 'gyro_x', 'GyroY': 'gyro_y', 'GyroZ': 'gyro_z',
        'Class': 'label', 'Timestamp': 'timestamp'
    }
    df.rename(columns=column_mapping, inplace=True)
    print("Renamed columns to standard format.")

    # Create Windows
    windows, labels = create_windows(df)
    
    # Engineer Features
    features_df = engineer_features_from_windows(windows)
    print(f"Engineered {features_df.shape[1]} features for each sample.")
    
    # Prepare Data for Modeling
    X = features_df
    y = pd.Series(labels)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nData prepared for training:")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train RandomForest Model
    print("\nTraining RandomForestClassifier with improved features...")
    # Increased n_estimators for more complex features
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=le.classes_, columns=le.classes_))
    
    # Save Artifacts
    print("\nSaving improved model and supporting artifacts...")
    artifacts = {
        'model': model, 'scaler': scaler, 'label_encoder': le,
        'feature_columns': list(X.columns)
    }
    joblib.dump(artifacts, 'driver_classification_artifacts.pkl')
    
    print("\n--- Training Complete ---")
    print("Saved improved artifacts to 'driver_classification_artifacts.pkl'.")

if __name__ == '__main__':
    train_model()
