import sys
import os
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from src.data_ingestion import download_star_data
from src.detrending import clean_and_flatten
from src.detection import find_planet_signal, fold_lightcurve
from src.features import extract_hybrid_features
from src.classifier import ExoplanetClassifier

FEATURE_FILE = "data/processed/extracted_features.csv"

def process_isro_dataset(csv_filepath):
    """
    Loops through the ISRO-provided CSV of TIC IDs, processes each raw light curve, 
    extracts the physical shapes, and saves the results to a feature dataset.
    """
    print(f"📂 Loading target list from {csv_filepath}...")
    targets_df = pd.read_csv(csv_filepath)
    
    all_features = []
    
    for index, row in targets_df.iterrows():
        tic_id = row['tic_id']
        label = row['label']
        print(f"\n[{index+1}/{len(targets_df)}] Processing {tic_id}...")
        
        try:
            # 1. Pipeline Execution
            raw_lc = download_star_data(tic_id)
            if raw_lc is None: 
                print(f"⏭️ Skipping {tic_id} (No data found)")
                continue
                
            clean_lc, _ = clean_and_flatten(raw_lc)
            planet_params, _ = find_planet_signal(clean_lc)
            folded_lc = fold_lightcurve(clean_lc, planet_params['period'], planet_params['t0'])
            
            # 2. Extract Shape Parameters
            features = extract_hybrid_features(clean_lc, planet_params, folded_lc)
            
            # Remove the array curve so we can save to CSV cleanly
            if 'best_fit_curve' in features:
                del features['best_fit_curve']
            
            features['tic_id'] = tic_id
            features['label'] = label # 1 for Planet, 0 for False Positive
            all_features.append(features)
            
        except Exception as e:
            print(f"⚠️ Error processing {tic_id}: {e}")
            continue

    # Save to CSV so we don't have to re-process raw data every time!
    os.makedirs("data/processed", exist_ok=True)
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(FEATURE_FILE, index=False)
    print(f"\n💾 Feature extraction complete! Saved to {FEATURE_FILE}")
    return features_df

def train_and_evaluate():
    """Trains the XGBoost model on the extracted features and plots the Confusion Matrix."""
    if not os.path.exists(FEATURE_FILE):
        print("❌ Extracted features not found! Please run the extraction process first.")
        return
        
    print("\n🚀 Starting Phase 5: Model Training and Evaluation...")
    df = pd.read_csv(FEATURE_FILE)
    
    # Split features (X) and labels (y)
    ai_engine = ExoplanetClassifier()
    X = df[ai_engine.feature_cols]
    y = df['label']
    
    # Split into 80% Training Data and 20% Testing Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train the Model
    ai_engine.train(X_train, y_train)
    
    # Generate Predictions
    print("🧠 Generating predictions on the unseen test dataset...")
    y_pred_probs = ai_engine.model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("📊 CLASSIFICATION PERFORMANCE METRICS (REAL DATA)")
    print("="*50)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=["False Positive (0)", "Exoplanet Transit (1)"]))
    
    # Plot Confusion Matrix
    print("🎨 Generating Confusion Matrix Plot...")
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["False Positive", "Exoplanet"], 
                yticklabels=["False Positive", "Exoplanet"], annot_kws={"size": 16})
    plt.title("AI Pipeline Confusion Matrix\n(Real TESS Validation Data)", fontsize=14, pad=15)
    plt.ylabel('True ASTROPHYSICAL Phenomenon', fontsize=12)
    plt.xlabel('AI Pipeline PREDICTION', fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/Phase5_RealData_Confusion_Matrix.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    target_csv = "isro_training_targets.csv"
    
    # Only process raw data if we haven't done it yet
    if not os.path.exists(FEATURE_FILE):
        if os.path.exists(target_csv):
            process_isro_dataset(target_csv)
        else:
            print(f"⚠️ Please create '{target_csv}' with 'tic_id' and 'label' columns to begin.")
            exit()
            
    # Train and evaluate on the extracted features
    train_and_evaluate()