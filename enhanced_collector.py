"""
Enhanced AirSim Telematics Data Collector
Captures 25+ features from all available sensors for comprehensive risk assessment
"""

import airsim
import numpy as np
import pandas as pd
import time
from datetime import datetime
import threading
import json
import os

class EnhancedTelematicsCollector:
    """Enhanced collector using all available AirSim sensors"""
    
    def __init__(self):
        print("Initializing Enhanced Telematics Collector...")
        
        # Connect to AirSim
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
        self.data_buffer = []
        self.collecting = False
        self.collection_thread = None
        
        # Initialize tracking variables
        self.prev_acceleration = None
        self.prev_speed = None
        self.prev_timestamp = None
        self.trip_start_time = datetime.now()
        
        # Weather and environmental data
        self.weather_conditions = {
            'rain': 0.0,
            'snow': 0.0,
            'fog': 0.0,
            'dust': 0.0
        }
        
        print("Enhanced collector initialized successfully!")
        
    def get_sensor_availability(self):
        """Check which sensors are available and working"""
        sensor_status = {}
        
        try:
            # Test basic car state
            car_state = self.client.getCarState()
            sensor_status['car_state'] = True
        except Exception as e:
            sensor_status['car_state'] = False
            print(f"Car state sensor error: {e}")
        
        try:
            # Test IMU
            imu_data = self.client.getImuData()
            sensor_status['imu'] = True
        except Exception as e:
            sensor_status['imu'] = False
            print(f"IMU sensor error: {e}")
        
        try:
            # Test GPS
            gps_data = self.client.getGpsData()
            sensor_status['gps'] = True
        except Exception as e:
            sensor_status['gps'] = False
            print(f"GPS sensor error: {e}")
        
        try:
            # Test collision detection
            collision_info = self.client.simGetCollisionInfo()
            sensor_status['collision'] = True
        except Exception as e:
            sensor_status['collision'] = False
            print(f"Collision sensor error: {e}")
        
        return sensor_status
    
    def collect_comprehensive_sensor_data(self, vehicle_name=""):
        """Collect data from all available sensors"""
        
        try:
            current_timestamp = datetime.now()
            
            # 1. CORE KINEMATICS DATA
            car_state = self.client.getCarState(vehicle_name)
            kinematics = car_state.kinematics_estimated
            
            # Position and GPS conversion
            position = kinematics.position
            lat, lon = self.ue4_to_gps(position.x_val, position.y_val)
            
            # Velocity components
            velocity = kinematics.linear_velocity
            angular_velocity = kinematics.angular_velocity
            
            # Acceleration components
            acceleration = kinematics.linear_acceleration
            angular_acceleration = kinematics.angular_acceleration
            
            # Calculate derived motion metrics
            speed_ms = np.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
            speed_kmh = speed_ms * 3.6
            
            # Linear acceleration magnitude
            linear_accel_mag = np.sqrt(
                acceleration.x_val**2 + 
                acceleration.y_val**2 + 
                acceleration.z_val**2
            )
            
            # Angular acceleration magnitude
            angular_accel_mag = np.sqrt(
                angular_acceleration.x_val**2 + 
                angular_acceleration.y_val**2 + 
                angular_acceleration.z_val**2
            )
            
            # 2. IMU DATA (High precision sensors)
            try:
                imu_data = self.client.getImuData(vehicle_name=vehicle_name)
                imu_accel = imu_data.linear_acceleration
                imu_angular = imu_data.angular_velocity
                imu_available = True
            except:
                # Fallback to kinematic data if IMU unavailable
                imu_accel = acceleration
                imu_angular = angular_velocity
                imu_available = False
            
            # 3. GPS DATA (Enhanced positioning)
            try:
                gps_data = self.client.getGpsData(vehicle_name=vehicle_name)
                gps_available = True
                # Use GPS coordinates if available, otherwise use converted UE4 coordinates
                if hasattr(gps_data, 'gnss_report'):
                    gps_lat = gps_data.gnss_report.geo_point.latitude if hasattr(gps_data.gnss_report.geo_point, 'latitude') else lat
                    gps_lon = gps_data.gnss_report.geo_point.longitude if hasattr(gps_data.gnss_report.geo_point, 'longitude') else lon
                    gps_alt = gps_data.gnss_report.geo_point.altitude if hasattr(gps_data.gnss_report.geo_point, 'altitude') else -position.z_val
                else:
                    gps_lat, gps_lon, gps_alt = lat, lon, -position.z_val
            except:
                gps_available = False
                gps_lat, gps_lon, gps_alt = lat, lon, -position.z_val
            
            # 4. COLLISION DETECTION
            try:
                collision_info = self.client.simGetCollisionInfo(vehicle_name)
                has_collision = collision_info.has_collided
                collision_impact = np.sqrt(
                    collision_info.impact_point.x_val**2 + 
                    collision_info.impact_point.y_val**2 + 
                    collision_info.impact_point.z_val**2
                ) if has_collision else 0.0
                collision_available = True
            except:
                has_collision = False
                collision_impact = 0.0
                collision_available = False
            
            # 5. ADVANCED DRIVING BEHAVIOR CALCULATIONS
            
            # Jerk calculation (rate of change of acceleration)
            if self.prev_acceleration is not None and self.prev_timestamp is not None:
                time_delta = (current_timestamp - self.prev_timestamp).total_seconds()
                if time_delta > 0:
                    jerk = (linear_accel_mag - self.prev_acceleration) / time_delta
                else:
                    jerk = 0.0
            else:
                jerk = 0.0
            
            # Update previous values
            self.prev_acceleration = linear_accel_mag
            self.prev_timestamp = current_timestamp
            
            # Lateral acceleration (cornering force)
            lateral_accel = speed_ms * angular_velocity.z_val if speed_ms > 0 else 0
            
            # Longitudinal vs lateral acceleration analysis
            longitudinal_accel = acceleration.x_val  # Forward/backward
            lateral_component = acceleration.y_val   # Side-to-side
            
            # Speed change rate
            if self.prev_speed is not None:
                speed_change_rate = speed_kmh - self.prev_speed
            else:
                speed_change_rate = 0.0
            self.prev_speed = speed_kmh
            
            # 6. ENVIRONMENTAL CONDITIONS
            time_of_day = current_timestamp.hour + current_timestamp.minute / 60.0
            day_of_week = current_timestamp.weekday()  # 0=Monday, 6=Sunday
            
            # Trip duration
            trip_duration = (current_timestamp - self.trip_start_time).total_seconds()
            
            # 7. ENHANCED DRIVING EVENT DETECTION
            
            # Harsh braking (more sensitive detection)
            harsh_braking = 1 if (longitudinal_accel < -3.5 or 
                                (longitudinal_accel < -2.5 and speed_change_rate < -5)) else 0
            
            # Harsh acceleration (context-aware)
            harsh_acceleration = 1 if (longitudinal_accel > 3.0 or 
                                     (longitudinal_accel > 2.0 and speed_change_rate > 8)) else 0
            
            # Harsh cornering (lateral force detection)
            harsh_cornering = 1 if (abs(lateral_accel) > 4.0 or 
                                  abs(lateral_component) > 3.5) else 0
            
            # Speed violations (context-aware)
            speed_limit = 60  # Base speed limit
            if time_of_day < 6 or time_of_day > 22:  # Night time
                speed_limit = 50
            speeding = 1 if speed_kmh > speed_limit else 0
            
            # Rapid lane changes
            rapid_lane_change = 1 if (abs(angular_velocity.z_val) > 0.5 and 
                                    speed_kmh > 30 and 
                                    abs(lateral_component) > 2.0) else 0
            
            # Tailgating indicator (simplified - would need LiDAR for accuracy)
            # Using jerk and speed as proxy for following too closely
            potential_tailgating = 1 if (jerk > 5.0 and speed_kmh > 40) else 0
            
            # Distracted driving indicator (erratic steering)
            steering_inconsistency = 1 if (abs(angular_velocity.z_val) > 0.3 and 
                                         angular_accel_mag > 2.0) else 0
            
            # 8. DRIVING QUALITY METRICS
            
            # Smoothness score (inverse of jerk and angular acceleration)
            smoothness_score = max(0, 100 - (abs(jerk) * 5 + angular_accel_mag * 3))
            
            # Efficiency score (consistent speed and acceleration)
            speed_efficiency = max(0, 100 - abs(speed_change_rate) * 2)
            acceleration_efficiency = max(0, 100 - linear_accel_mag * 10)
            
            # Overall driving score
            overall_driving_score = (smoothness_score * 0.4 + 
                                   speed_efficiency * 0.3 + 
                                   acceleration_efficiency * 0.3)
            
            # 9. COMPREHENSIVE DATA POINT ASSEMBLY
            data_point = {
                # === METADATA ===
                'timestamp': current_timestamp,
                'vehicle_id': vehicle_name or 'default_car',
                'trip_duration_seconds': trip_duration,
                'sensor_status': json.dumps({
                    'imu': imu_available,
                    'gps': gps_available,
                    'collision': collision_available
                }),
                
                # === POSITION & NAVIGATION ===
                'lat': gps_lat,
                'lon': gps_lon,
                'altitude': gps_alt,
                'position_x': position.x_val,
                'position_y': position.y_val,
                'position_z': position.z_val,
                'heading': self.calculate_heading(kinematics.orientation),
                
                # === SPEED & VELOCITY ===
                'speed_kmh': speed_kmh,
                'speed_ms': speed_ms,
                'velocity_x': velocity.x_val,
                'velocity_y': velocity.y_val,
                'velocity_z': velocity.z_val,
                'speed_change_rate': speed_change_rate,
                
                # === LINEAR MOTION ===
                'acceleration_x': acceleration.x_val,
                'acceleration_y': acceleration.y_val,
                'acceleration_z': acceleration.z_val,
                'longitudinal_accel': longitudinal_accel,
                'lateral_accel': lateral_component,
                'accel_magnitude': linear_accel_mag,
                'jerk': jerk,
                
                # === ANGULAR MOTION ===
                'angular_velocity_x': angular_velocity.x_val,
                'angular_velocity_y': angular_velocity.y_val,
                'angular_velocity_z': angular_velocity.z_val,
                'angular_acceleration_x': angular_acceleration.x_val,
                'angular_acceleration_y': angular_acceleration.y_val,
                'angular_acceleration_z': angular_acceleration.z_val,
                'angular_accel_mag': angular_accel_mag,
                'lateral_force': lateral_accel,
                
                # === IMU DATA (High Precision) ===
                'imu_accel_x': imu_accel.x_val,
                'imu_accel_y': imu_accel.y_val,
                'imu_accel_z': imu_accel.z_val,
                'imu_angular_x': imu_angular.x_val,
                'imu_angular_y': imu_angular.y_val,
                'imu_angular_z': imu_angular.z_val,
                
                # === VEHICLE STATE ===
                'gear': car_state.gear,
                'rpm': car_state.rpm,
                'handbrake': car_state.handbrake,
                'speed_limit': speed_limit,
                
                # === COLLISION & SAFETY ===
                'collision': has_collision,
                'collision_impact': collision_impact,
                
                # === ENVIRONMENTAL CONTEXT ===
                'time_of_day': time_of_day,
                'day_of_week': day_of_week,
                'weather_rain': self.weather_conditions['rain'],
                'weather_snow': self.weather_conditions['snow'],
                'weather_fog': self.weather_conditions['fog'],
                'weather_dust': self.weather_conditions['dust'],
                
                # === DRIVING EVENTS (Insurance Critical) ===
                'harsh_braking': harsh_braking,
                'harsh_acceleration': harsh_acceleration,
                'harsh_cornering': harsh_cornering,
                'speeding': speeding,
                'rapid_lane_change': rapid_lane_change,
                'potential_tailgating': potential_tailgating,
                'steering_inconsistency': steering_inconsistency,
                
                # === DRIVING QUALITY SCORES ===
                'smoothness_score': smoothness_score,
                'speed_efficiency': speed_efficiency,
                'acceleration_efficiency': acceleration_efficiency,
                'overall_driving_score': overall_driving_score,
                
                # === RISK AGGREGATION ===
                'total_risk_events': (harsh_braking + harsh_acceleration + harsh_cornering + 
                                    speeding + rapid_lane_change),
                'total_behavior_flags': (potential_tailgating + steering_inconsistency),
                
                # === DERIVED METRICS ===
                'g_force': linear_accel_mag / 9.81,  # G-force experienced
                'cornering_severity': abs(lateral_accel) / max(speed_ms, 0.1),
                'acceleration_variance': abs(jerk),
                'driving_aggressiveness_score': min(100, (harsh_braking + harsh_acceleration + 
                                                        harsh_cornering) * 20 + abs(jerk) * 5)
            }
            
            return data_point
            
        except Exception as e:
            print(f"Enhanced sensor data collection error: {e}")
            return None
    
    def calculate_heading(self, orientation):
        """Calculate heading from quaternion orientation"""
        return np.degrees(np.arctan2(
            2 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
            1 - 2 * (orientation.y_val**2 + orientation.z_val**2)
        ))
    
    def ue4_to_gps(self, x, y, origin_lat=47.641468, origin_lon=-122.140165):
        """Convert UE4 coordinates to GPS coordinates"""
        lat = origin_lat + (x / 111000.0)
        lon = origin_lon + (y / (111000.0 * np.cos(np.radians(origin_lat))))
        return lat, lon
    
    def start_enhanced_collection(self, interval=0.1):
        """Start enhanced continuous data collection"""
        self.collecting = True
        self.trip_start_time = datetime.now()
        
        def enhanced_collect_loop():
            data_count = 0
            while self.collecting:
                try:
                    data_point = self.collect_comprehensive_sensor_data()
                    if data_point:
                        self.data_buffer.append(data_point)
                        data_count += 1
                        
                        # Progress indicator
                        if data_count % 100 == 0:
                            print(f"   Collected {data_count} enhanced data points...")
                    
                    time.sleep(interval)
                except Exception as e:
                    print(f"Collection loop error: {e}")
                    time.sleep(interval)
        
        self.collection_thread = threading.Thread(target=enhanced_collect_loop)
        self.collection_thread.start()
        print(f"Started enhanced telematics collection (every {interval}s)")
        print("Collecting 50+ features per data point...")
    
    def stop_collection(self):
        """Stop data collection and show summary"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        
        print(f"Enhanced collection stopped!")
        print(f"Total data points collected: {len(self.data_buffer)}")
        
        if len(self.data_buffer) > 0:
            # Show feature summary
            sample_point = self.data_buffer[0]
            print(f"Features captured per point: {len(sample_point)}")
            print(f"Trip duration: {(datetime.now() - self.trip_start_time).total_seconds():.1f} seconds")
    
    def restore_manual_control(self):
        """Restore manual keyboard control"""
        try:
            self.client.enableApiControl(False)
            print("Manual keyboard control restored!")
            print("You can now use WASD keys in AirSim window to drive.")
            print("Controls: W=Throttle, S=Brake, A=Left, D=Right, Q=Handbrake, R=Reset")
        except Exception as e:
            print(f"Error restoring manual control: {e}")
    
    def get_enhanced_dataframe(self):
        """Convert enhanced data to pandas DataFrame"""
        return pd.DataFrame(self.data_buffer)
    
    def save_enhanced_data(self, filename=None):
        """Save enhanced telematics data with feature summary"""
        if not filename:
            filename = f"enhanced_telematics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.get_enhanced_dataframe()
        df.to_csv(filename, index=False)
        
        # Save feature metadata
        metadata_file = filename.replace('.csv', '_metadata.json')
        if len(self.data_buffer) > 0:
            sample_point = self.data_buffer[0]
            metadata = {
                'total_features': len(sample_point),
                'feature_names': list(sample_point.keys()),
                'collection_timestamp': datetime.now().isoformat(),
                'total_data_points': len(df),
                'collection_duration_seconds': (datetime.now() - self.trip_start_time).total_seconds()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Enhanced data saved to: {filename}")
        print(f"Metadata saved to: {metadata_file}")
        return filename, metadata_file

def test_enhanced_collector():
    """Test the enhanced collector with sample data collection"""
    
    print("=" * 60)
    print("ENHANCED TELEMATICS COLLECTOR TEST")
    print("=" * 60)
    
    # Initialize collector
    collector = EnhancedTelematicsCollector()
    
    # Check sensor availability
    print("\nChecking sensor availability...")
    sensor_status = collector.get_sensor_availability()
    for sensor, available in sensor_status.items():
        status = "✓ Available" if available else "✗ Unavailable"
        print(f"  {sensor}: {status}")
    
    # Collect sample data
    print(f"\nCollecting sample data point...")
    sample_data = collector.collect_comprehensive_sensor_data()
    
    if sample_data:
        print(f"✓ Sample data collected successfully!")
        print(f"Total features captured: {len(sample_data)}")
        
        # Show key features
        print(f"\nKey telematics features:")
        key_features = ['speed_kmh', 'accel_magnitude', 'jerk', 'harsh_braking', 
                       'harsh_acceleration', 'overall_driving_score', 'g_force']
        for feature in key_features:
            if feature in sample_data:
                print(f"  {feature}: {sample_data[feature]}")
        
        return True
    else:
        print("✗ Failed to collect sample data")
        return False

def run_enhanced_collection_demo():
    """Run a complete enhanced data collection demo"""
    
    print("Choose enhanced collection mode:")
    print("1. Quick test (10 seconds)")
    print("2. Short collection (60 seconds)")  
    print("3. Full collection (300 seconds)")
    print("4. Just test sensors")
    print("5. Manual driving collection (drive with WASD)")
    print("6. Just restore keyboard control")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "4":
        return test_enhanced_collector()
    
    if choice == "6":
        # Just restore manual control
        try:
            client = airsim.CarClient()
            client.confirmConnection()
            client.enableApiControl(False)
            print("Manual keyboard control restored!")
            print("You can now use WASD keys in AirSim window to drive.")
            return None
        except Exception as e:
            print(f"Error restoring manual control: {e}")
            return None
    
    collector = EnhancedTelematicsCollector()
    
    if choice == "5":
        # Manual driving collection
        print("\nManual Driving Collection Mode")
        print("=" * 40)
        print("Instructions:")
        print("1. Keyboard control will be enabled")
        print("2. Drive around using WASD keys in AirSim")
        print("3. Data will be collected automatically")
        print("4. Press Ctrl+C here to stop collection")
        
        # Enable manual control for driving
        collector.client.enableApiControl(False)
        print("\nManual control enabled! Use WASD keys in AirSim window.")
        
        try:
            collector.start_enhanced_collection(interval=0.1)
            print("Enhanced data collection started...")
            print("Drive around! Press Ctrl+C to stop collection.")
            
            # Keep collecting until user stops
            while True:
                time.sleep(1)
                if len(collector.data_buffer) % 100 == 0 and len(collector.data_buffer) > 0:
                    print(f"   Collected {len(collector.data_buffer)} enhanced data points...")
        
        except KeyboardInterrupt:
            print("\nStopping collection...")
            collector.stop_collection()
            
            if len(collector.data_buffer) > 0:
                filename, metadata_file = collector.save_enhanced_data()
                df = collector.get_enhanced_dataframe()
                
                print(f"\nMANUAL DRIVING RESULTS:")
                print(f"  Total data points: {len(df)}")
                print(f"  Total features: {df.shape[1]}")
                
                # Show risk events
                risk_events = ['harsh_braking', 'harsh_acceleration', 'harsh_cornering', 'speeding']
                print(f"\nRISK EVENTS DETECTED:")
                for event in risk_events:
                    if event in df.columns:
                        count = df[event].sum()
                        print(f"  {event}: {count} events")
                
                if 'overall_driving_score' in df.columns:
                    avg_score = df['overall_driving_score'].mean()
                    print(f"\nYOUR DRIVING QUALITY SCORE: {avg_score:.1f}/100")
                
                print(f"\nData saved to: {filename}")
                return filename
            else:
                print("No data collected")
                return None
    
    # Automated collection modes (1, 2, 3)
    duration_map = {"1": 10, "2": 60, "3": 300}
    duration = duration_map.get(choice, 60)
    
    try:
        print(f"\nStarting enhanced collection for {duration} seconds...")
        print("Car will drive automatically while collecting data...")
        
        # Start collection
        collector.start_enhanced_collection(interval=0.1)
        
        # Wait for specified duration
        time.sleep(duration)
        
        # Stop collection
        collector.stop_collection()
        
        # Restore manual control
        collector.restore_manual_control()
        
        # Save enhanced data
        if len(collector.data_buffer) > 0:
            filename, metadata_file = collector.save_enhanced_data()
            
            # Show data summary
            df = collector.get_enhanced_dataframe()
            print(f"\nENHANCED DATA SUMMARY:")
            print(f"  Total data points: {len(df)}")
            print(f"  Total features: {df.shape[1]}")
            print(f"  Collection rate: {len(df)/duration:.1f} points/second")
            
            # Show risk event summary
            risk_events = ['harsh_braking', 'harsh_acceleration', 'harsh_cornering', 'speeding']
            print(f"\nRISK EVENTS DETECTED:")
            for event in risk_events:
                if event in df.columns:
                    count = df[event].sum()
                    print(f"  {event}: {count} events")
            
            # Show quality scores
            if 'overall_driving_score' in df.columns:
                avg_score = df['overall_driving_score'].mean()
                print(f"\nAVERAGE DRIVING QUALITY SCORE: {avg_score:.1f}/100")
            
            print(f"\nEnhanced telematics data ready for risk model training!")
            return filename
        else:
            print("No data collected")
            return None
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        collector.stop_collection()
        collector.restore_manual_control()
        if len(collector.data_buffer) > 0:
            filename, _ = collector.save_enhanced_data()
            return filename
        return None

if __name__ == "__main__":
    # Run the enhanced collection demo
    result = run_enhanced_collection_demo()
    
    if result:
        print(f"\nNext steps:")
        print(f"1. Use this enhanced data file: {result}")
        print(f"2. Train the baseline risk model")
        print(f"3. Compare with your original data")
        print(f"4. Deploy real-time risk scoring")