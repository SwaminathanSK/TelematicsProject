"""
Industry-Standard Baseline Risk Scoring Model for Telematics Insurance
Based on academic research and industry best practices from LexisNexis, OCTO Telematics, and published studies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class IndustryStandardRiskScorer:
    """
    Industry-standard risk scoring model based on published research:
    - LexisNexis Drive Metrics (79% lift above standard factors)
    - OCTO Telematics DriveAbility Score
    - Academic research on gradient boosting for insurance
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.risk_thresholds = {
            'low_risk': 0.3,      # Bottom 30% (safest drivers)
            'high_risk': 0.7      # Top 30% (riskiest drivers)
        }
        
        # Industry-standard feature groups (based on research)
        self.feature_groups = {
            'harsh_events': ['harsh_braking', 'harsh_acceleration', 'harsh_cornering'],
            'speed_behavior': ['speed_kmh', 'speeding', 'speed_change_rate'],
            'driving_quality': ['smoothness_score', 'overall_driving_score', 'jerk'],
            'contextual': ['time_of_day', 'day_of_week', 'trip_duration_seconds'],
            'advanced': ['g_force', 'lateral_force', 'acceleration_variance']
        }
        
    def load_and_prepare_data(self, csv_file_path):
        """Load and prepare telematics data for risk scoring"""
        
        print("Loading and preparing telematics data...")
        df = pd.read_csv(csv_file_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # Basic data quality checks
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        
        # Remove duplicates and handle missing values
        df = df.drop_duplicates()
        df = df.fillna(0)  # Conservative approach for telematics data
        
        return df
    
    def engineer_risk_features(self, df):
        """
        Engineer risk features based on industry research
        Features aligned with LexisNexis and OCTO methodologies
        """
        
        print("Engineering industry-standard risk features...")
        
        # Create a copy for feature engineering
        df_features = df.copy()
        
        # 1. AGGREGATE RISK EVENTS (Most Important)
        harsh_event_cols = [col for col in ['harsh_braking', 'harsh_acceleration', 
                           'harsh_cornering', 'rapid_lane_change'] if col in df.columns]
        if harsh_event_cols:
            df_features['total_harsh_events'] = df_features[harsh_event_cols].sum(axis=1)
            df_features['harsh_event_rate'] = df_features['total_harsh_events'] / df_features.groupby('vehicle_id').cumcount().add(1)
        
        # 2. SPEED-RELATED FEATURES
        if 'speed_kmh' in df.columns:
            df_features['speed_percentile'] = df_features.groupby('vehicle_id')['speed_kmh'].rank(pct=True)
            df_features['excessive_speed'] = (df_features['speed_kmh'] > 80).astype(int)
            df_features['speed_variance'] = df_features.groupby('vehicle_id')['speed_kmh'].transform('std')
            
        # 3. ACCELERATION PATTERNS
        if 'accel_magnitude' in df.columns:
            df_features['high_accel_events'] = (df_features['accel_magnitude'] > 3.0).astype(int)
            df_features['accel_smoothness'] = 1 / (df_features['jerk'].abs() + 0.1) if 'jerk' in df.columns else 50
            
        # 4. DRIVING CONSISTENCY METRICS
        if 'overall_driving_score' in df.columns:
            df_features['driving_quality_category'] = pd.cut(
                df_features['overall_driving_score'], 
                bins=[0, 60, 80, 100], 
                labels=['poor', 'average', 'good']
            )
            
        # 5. TIME-BASED RISK FACTORS
        if 'time_of_day' in df.columns:
            df_features['night_driving'] = ((df_features['time_of_day'] < 6) | 
                                          (df_features['time_of_day'] > 22)).astype(int)
            df_features['rush_hour'] = ((df_features['time_of_day'].between(7, 9)) | 
                                      (df_features['time_of_day'].between(17, 19))).astype(int)
        
        # 6. TRIP-LEVEL AGGREGATIONS (Industry Standard)
        trip_level_features = df_features.groupby('vehicle_id').agg({
            # Speed metrics
            'speed_kmh': ['mean', 'std', 'max', 'quantile'],
            
            # Risk events
            'total_harsh_events': ['sum', 'mean'],
            'harsh_braking': ['sum', 'mean'] if 'harsh_braking' in df.columns else ['sum'],
            'harsh_acceleration': ['sum', 'mean'] if 'harsh_acceleration' in df.columns else ['sum'],
            'speeding': ['sum', 'mean'] if 'speeding' in df.columns else ['sum'],
            
            # Quality metrics
            'overall_driving_score': ['mean', 'std', 'min'] if 'overall_driving_score' in df.columns else ['mean'],
            'smoothness_score': ['mean', 'std'] if 'smoothness_score' in df.columns else ['mean'],
            
            # Advanced metrics
            'g_force': ['mean', 'max', 'std'] if 'g_force' in df.columns else ['mean'],
            'jerk': ['mean', 'max', 'std'] if 'jerk' in df.columns else ['mean'],
            
            # Contextual
            'night_driving': ['sum', 'mean'],
            'trip_duration_seconds': ['sum', 'mean', 'count'] if 'trip_duration_seconds' in df.columns else ['count'],
        }).round(3)
        
        # Flatten column names
        trip_level_features.columns = ['_'.join(col).strip() for col in trip_level_features.columns]
        
        # Calculate derived metrics
        trip_level_features['risk_events_per_100_trips'] = (
            trip_level_features['total_harsh_events_sum'] / 
            trip_level_features.get('trip_duration_seconds_count', 100) * 100
        )
        
        trip_level_features['avg_driving_quality'] = trip_level_features.get(
            'overall_driving_score_mean', 75
        )
        
        trip_level_features['speed_consistency'] = (
            trip_level_features.get('speed_kmh_mean', 50) / 
            (trip_level_features.get('speed_kmh_std', 1) + 0.1)
        )
        
        print(f"Engineered features shape: {trip_level_features.shape}")
        return trip_level_features
    
    def create_risk_labels(self, df_features):
        """
        Create risk labels using industry-standard methodology
        Based on LexisNexis and insurance research approaches
        """
        
        print("Creating industry-standard risk labels...")
        
        risk_scores = []
        
        for idx, row in df_features.iterrows():
            score = 0
            
            # 1. HARSH EVENTS SCORING (40% of total risk - highest weight)
            harsh_events_rate = row.get('risk_events_per_100_trips', 0)
            if harsh_events_rate > 15:
                score += 40
            elif harsh_events_rate > 8:
                score += 25
            elif harsh_events_rate > 3:
                score += 10
            
            # 2. SPEED BEHAVIOR SCORING (25% of total risk)
            max_speed = row.get('speed_kmh_max', 0)
            speed_violations = row.get('speeding_sum', 0)
            if max_speed > 100 or speed_violations > 20:
                score += 25
            elif max_speed > 80 or speed_violations > 10:
                score += 15
            elif max_speed > 60 or speed_violations > 5:
                score += 8
            
            # 3. DRIVING QUALITY SCORING (20% of total risk)
            driving_quality = row.get('avg_driving_quality', 75)
            if driving_quality < 50:
                score += 20
            elif driving_quality < 70:
                score += 12
            elif driving_quality < 85:
                score += 5
            
            # 4. CONSISTENCY SCORING (10% of total risk)
            consistency = row.get('speed_consistency', 10)
            if consistency < 5:
                score += 10
            elif consistency < 10:
                score += 5
            
            # 5. NIGHT/HIGH-RISK DRIVING (5% of total risk)
            night_driving_rate = row.get('night_driving_mean', 0)
            if night_driving_rate > 0.3:  # >30% night driving
                score += 5
            elif night_driving_rate > 0.15:  # >15% night driving
                score += 2
            
            risk_scores.append(score)
        
        # Convert to risk categories based on score distribution
        risk_scores = np.array(risk_scores)
        
        # Industry-standard 3-tier classification
        low_threshold = np.percentile(risk_scores, 33)
        high_threshold = np.percentile(risk_scores, 67)
        
        risk_labels = []
        for score in risk_scores:
            if score <= low_threshold:
                risk_labels.append('LOW_RISK')
            elif score <= high_threshold:
                risk_labels.append('MEDIUM_RISK')
            else:
                risk_labels.append('HIGH_RISK')
        
        print(f"Risk distribution: {pd.Series(risk_labels).value_counts()}")
        return risk_labels, risk_scores
    
    def train_ensemble_models(self, X, y):
        """
        Train ensemble of industry-standard models
        Based on research showing gradient boosting superiority
        """
        
        print("Training industry-standard ensemble models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        models_to_train = {
            # 1. LOGISTIC REGRESSION (Interpretability baseline)
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            
            # 2. RANDOM FOREST (Industry standard ensemble)
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            
            # 3. XGBOOST (Research-proven for telematics)
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            
            # 4. CATBOOST (Best for insurance claims per research)
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            ),
            
            # 5. GRADIENT BOOSTING (Classical approach)
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Use scaled data for logistic regression
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Multi-class ROC AUC
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = 0.5
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")
        
        # Store models and results
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        # Select best model based on comprehensive scoring
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x]['accuracy'] + results[x]['f1_score'] + results[x]['roc_auc'])
        
        self.best_model_name = best_model_name
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest performing model: {best_model_name}")
        return results
    
    def analyze_feature_importance(self, X):
        """Analyze feature importance across models"""
        
        print("Analyzing feature importance...")
        
        feature_names = X.columns
        importance_data = {}
        
        # Get feature importance from tree-based models
        tree_models = ['random_forest', 'xgboost', 'catboost', 'gradient_boosting']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = model.feature_importances_
                elif hasattr(model, 'get_feature_importance'):  # CatBoost
                    importance_data[model_name] = model.get_feature_importance()
        
        # Create feature importance DataFrame
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=feature_names)
            importance_df['average'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('average', ascending=False)
            
            self.feature_importance = importance_df
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10)['average'])
            
            return importance_df
        
        return None
    
    def create_comprehensive_report(self, df_original):
        """Create comprehensive risk scoring report"""
        
        print("\n" + "="*60)
        print("INDUSTRY-STANDARD RISK SCORING MODEL REPORT")
        print("="*60)
        
        # 1. MODEL PERFORMANCE COMPARISON
        print("\n1. MODEL PERFORMANCE COMPARISON")
        print("-" * 40)
        
        performance_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'F1_Score': [self.results[model]['f1_score'] for model in self.results.keys()],
            'ROC_AUC': [self.results[model]['roc_auc'] for model in self.results.keys()]
        })
        
        performance_df = performance_df.sort_values('Accuracy', ascending=False)
        print(performance_df.round(3))
        
        # 2. DETAILED CLASSIFICATION REPORT
        print(f"\n2. DETAILED CLASSIFICATION REPORT - {self.best_model_name.upper()}")
        print("-" * 60)
        
        best_predictions = self.results[self.best_model_name]['predictions']
        print(classification_report(self.y_test, best_predictions))
        
        # 3. RISK DISTRIBUTION ANALYSIS
        print("\n3. RISK DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        risk_distribution = pd.Series(best_predictions).value_counts(normalize=True)
        print("Predicted Risk Distribution:")
        for risk_level, percentage in risk_distribution.items():
            print(f"  {risk_level}: {percentage:.1%}")
        
        # 4. FEATURE IMPORTANCE INSIGHTS
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            print("\n4. TOP RISK FACTORS")
            print("-" * 30)
            
            top_features = self.feature_importance.head(10)
            for feature, importance in top_features['average'].items():
                print(f"  {feature}: {importance:.3f}")
        
        # 5. BUSINESS IMPACT ANALYSIS
        print("\n5. BUSINESS IMPACT ANALYSIS")
        print("-" * 35)
        
        # Calculate potential premium adjustments
        risk_multipliers = {'LOW_RISK': 0.8, 'MEDIUM_RISK': 1.0, 'HIGH_RISK': 1.3}
        base_premium = 1000  # Base annual premium
        
        total_adjusted_premium = 0
        total_base_premium = len(best_predictions) * base_premium
        
        for risk_level, count in pd.Series(best_predictions).value_counts().items():
            adjusted_premium = base_premium * risk_multipliers[risk_level] * count
            total_adjusted_premium += adjusted_premium
            
            print(f"  {risk_level}: {count} drivers, Premium multiplier: {risk_multipliers[risk_level]}x")
        
        revenue_impact = total_adjusted_premium - total_base_premium
        print(f"\nEstimated Revenue Impact: ${revenue_impact:,.2f}")
        print(f"Average Premium Change: {(total_adjusted_premium/total_base_premium - 1)*100:.1f}%")
        
        return performance_df
    
    def save_model_and_results(self, filename_prefix="risk_scoring_model"):
        """Save trained models and results"""
        
        import joblib
        
        # Save best model
        joblib.dump(self.best_model, f"{filename_prefix}_best_model.pkl")
        joblib.dump(self.scalers, f"{filename_prefix}_scalers.pkl")
        
        # Save feature importance
        if hasattr(self, 'feature_importance'):
            self.feature_importance.to_csv(f"{filename_prefix}_feature_importance.csv")
        
        # Save results summary
        results_summary = {
            'best_model': self.best_model_name,
            'model_performance': {name: {
                'accuracy': self.results[name]['accuracy'],
                'f1_score': self.results[name]['f1_score'],
                'roc_auc': self.results[name]['roc_auc']
            } for name in self.results.keys()}
        }
        
        import json
        with open(f"{filename_prefix}_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nModels and results saved with prefix: {filename_prefix}")

def run_complete_risk_scoring_pipeline(csv_file_path):
    """
    Run the complete industry-standard risk scoring pipeline
    """
    
    print("STARTING INDUSTRY-STANDARD RISK SCORING PIPELINE")
    print("="*60)
    
    # Initialize risk scorer
    risk_scorer = IndustryStandardRiskScorer()
    
    # Step 1: Load and prepare data
    df = risk_scorer.load_and_prepare_data(csv_file_path)
    
    # Step 2: Engineer risk features
    df_features = risk_scorer.engineer_risk_features(df)
    
    # Step 3: Create risk labels
    risk_labels, risk_scores = risk_scorer.create_risk_labels(df_features)
    
    # Step 4: Prepare features for modeling
    # Select numerical features for modeling
    feature_columns = [col for col in df_features.columns 
                      if df_features[col].dtype in ['int64', 'float64'] and 
                      not col.startswith('timestamp')]
    
    X = df_features[feature_columns].fillna(0)
    y = risk_labels
    
    print(f"\nFeatures for modeling: {X.shape[1]}")
    print(f"Target distribution: {pd.Series(y).value_counts()}")
    
    # Step 5: Train models
    results = risk_scorer.train_ensemble_models(X, y)
    
    # Step 6: Analyze feature importance
    importance_df = risk_scorer.analyze_feature_importance(X)
    
    # Step 7: Create comprehensive report
    performance_df = risk_scorer.create_comprehensive_report(df)
    
    # Step 8: Save results
    risk_scorer.save_model_and_results()
    
    print("\n" + "="*60)
    print("RISK SCORING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return risk_scorer, performance_df

# Example usage and testing
if __name__ == "__main__":
    print("Industry-Standard Risk Scoring Model")
    print("Based on LexisNexis, OCTO Telematics, and Academic Research")
    print("="*60)
    
    # Instructions for use
    print("\nTO USE THIS MODEL:")
    print("1. Replace 'your_enhanced_telematics_data.csv' with your actual CSV file")
    print("2. Run: python baseline_risk_scoring_model.py")
    print("3. The model will train and generate comprehensive results")
    
    # If you have a CSV file, uncomment the next line:
    csv_file_path = "enhanced_telematics_20250801_211431.csv"  # Replace with your file
    risk_scorer, results = run_complete_risk_scoring_pipeline(csv_file_path)
    
    # print("\nExpected Outputs:")
    # print("- Model performance comparison")
    # print("- Feature importance analysis") 
    # print("- Risk distribution insights")
    # print("- Business impact analysis")
    # print("- Saved models for deployment")