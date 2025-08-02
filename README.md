# Context-Aware Telematics Risk Engine

## 1. Summary of Problem & Solution Strategy

### The Problem
Traditional auto insurance relies on static, demographic data (age, location, vehicle type), which often fails to reflect an individual's actual driving habits. This leads to unfair premiums and provides no incentive for safer driving. The objective of this project is to build a proof-of-concept for a Usage-Based Insurance (UBI) model that addresses these shortcomings.

### Our Solution
This project implements an end-to-end telematics solution that calculates a dynamic insurance premium based on three core pillars:

**Driver Behavior Analysis**: At the heart of the system is a state-of-the-art deep learning model that analyzes real-time sensor data (accelerometer and gyroscope) to classify driving style as either SAFE or AGGRESSIVE. This moves beyond simple metrics to understand the quality of driving.

**Contextual Risk Assessment**: We recognize that risk is not solely about behavior. Our engine incorporates external factors, allowing for the simulation of:
- Weather Conditions: (e.g., Rain, Fog, Snow)
- Traffic Density: (e.g., Light, Moderate, Heavy)
- Route Risk: Analyzing if the trip passes through historically accident-prone zones.

**Dynamic Premium Calculation**: A weighted algorithm combines the Behavior Score, Context Score, and Route Score into a final, transparent, and fair insurance premium. This provides direct feedback to the driver, incentivizing safer choices.

This strategy moves from a generalized model to a personalized one, creating a fairer system for both the insurer and the customer.

## 2. Data Model & Schema

The system is built upon two primary types of data: telematics data for behavior analysis and contextual data for situational risk.

### 2.1. Telematics Sensor Data

This data is collected from the vehicle's (or smartphone's) Inertial Measurement Unit (IMU) and forms the input for our deep learning model.

| Feature | Data Type | Unit | Description | Source |
|---------|-----------|------|-------------|---------|
| timestamp | float | seconds | The Unix timestamp of the data point. | IMU / Phone Sensor |
| acceleration_x | float | m/s² | Forward/backward acceleration. | IMU / Phone Sensor |
| acceleration_y | float | m/s² | Left/right (lateral) acceleration. | IMU / Phone Sensor |
| acceleration_z | float | m/s² | Up/down acceleration. | IMU / Phone Sensor |
| gyro_x | float | rad/s | Angular velocity around the X-axis (pitch). | IMU / Phone Sensor |
| gyro_y | float | rad/s | Angular velocity around the Y-axis (roll). | IMU / Phone Sensor |
| gyro_z | float | rad/s | Angular velocity around the Z-axis (yaw). | IMU / Phone Sensor |
| label | string | - | The ground-truth label (SAFE or AGGRESSIVE). | Training Data Only |

### 2.2. Contextual & Pricing Data

This data is used by the pricing engine to adjust the risk score calculated from the telematics data.

| Feature | Data Type | Description |
|---------|-----------|-------------|
| Behavior Score | float | A multiplier based on the model's prediction (e.g., 1.0 for SAFE, 1.5 for AGGRESSIVE). |
| Context Score | float | A combined multiplier from weather and traffic conditions. |
| Route Score | float | A multiplier based on the percentage of the trip in high-risk zones. |
| Base Premium | float | The policyholder's initial premium before dynamic adjustments. |
| Final Premium | float | The final calculated premium after all multipliers are applied. |

## 3. Model Architecture: CNN-LSTM with Attention

To accurately classify driving behavior, we chose a sophisticated deep learning architecture that captures both short-term patterns and long-term temporal dependencies in the sensor data.

**Convolutional Neural Network (CNN) Layer**: The initial Conv1d layer acts as a feature extractor. It scans for significant low-level patterns in the time-series data, such as sharp braking (a spike in acceleration) or sudden swerving (a pattern across multiple axes).

**Long Short-Term Memory (LSTM) Layer**: The output from the CNN is fed into a bidirectional LSTM. This layer is crucial for understanding the sequence of events. It can learn the temporal relationships between patterns, distinguishing between a single, isolated hard brake and a sustained pattern of erratic acceleration and deceleration that indicates aggressive driving.

**Attention Mechanism**: The final layer of analysis is an Attention mechanism. It allows the model to dynamically focus on the most critical parts of the driving sequence when making its final prediction. For example, if a 20-second window contains 18 seconds of safe driving and 2 seconds of a dangerous maneuver, the attention layer learns to assign a higher weight to those 2 critical seconds, leading to a more accurate classification.

This hybrid model was trained on labeled data to distinguish between SAFE and AGGRESSIVE driving styles, achieving high accuracy on the test set. The trained artifacts (`sota_pytorch_anomaly_artifacts.pkl`) contain the model weights, the data scaler, and the label encoder needed for inference.