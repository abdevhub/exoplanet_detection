import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd

class ExoplanetClassifier:
    def __init__(self):
        # State-of-the-art XGBoost Classifier architecture
        self.model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            eval_metric='logloss', 
            random_state=42
        )
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_trained = False
        # The specific features our model expects, in order
        self.feature_cols = ['period', 'snr', 'flux_std', 'flux_skew', 't_tot', 't_in', 't_flat', 'shape_ratio', 'fit_depth']
        
    def train(self, X_train, y_train):
        """Trains the XGBoost model on real extracted data features."""
        print("🏋️ Training XGBoost model on real dataset...")
        
        # Train the model
        self.model.fit(X_train[self.feature_cols], y_train)
        
        # Calibrate probabilities using Isotonic Regression
        raw_probs = self.model.predict_proba(X_train[self.feature_cols])[:, 1]
        self.calibrator.fit(raw_probs, y_train)
        
        self.is_trained = True
        print("✅ Model training complete and calibrated!")
        
    def classify(self, feature_dict):
        """Predicts the target label and returns a statistically calibrated confidence."""
        if not self.is_trained:
            raise Exception("Model is not trained yet! Please train on real data first.")
            
        feature_vector = pd.DataFrame([feature_dict])[self.feature_cols]
        
        raw_prob = self.model.predict_proba(feature_vector)[0, 1]
        confidence = self.calibrator.predict(np.array([raw_prob]))[0]
        
        label = "Exoplanet Transit" if confidence > 0.5 else "False Positive"
        
        return label, confidence