import os
from src.data_ingestion import download_star_data
from src.detrending import clean_and_flatten
from src.detection import find_planet_signal, fold_lightcurve
from src.features import extract_hybrid_features
from src.classifier import ExoplanetClassifier

def run_phase3():
    target_star = "TIC 25155310" # Benchmark Test Case
    print(f"🚀 Starting Phase 3 Integration Pipeline for {target_star}\n")
    
    # --- Execute Preceding Phases ---
    raw_lc = download_star_data(target_star)
    if raw_lc is None: 
        return
        
    clean_lc, trend_flux = clean_and_flatten(raw_lc)
    planet_params, tls_results = find_planet_signal(clean_lc)
    folded_lc = fold_lightcurve(clean_lc, planet_params['period'], planet_params['t0'])
    
    print("\n" + "="*50 + "\n")
    
    # --- Execute Phase 3 Modules ---
    features = extract_hybrid_features(clean_lc, planet_params, folded_lc)
    
    ai_engine = ExoplanetClassifier()
    prediction_label, confidence = ai_engine.classify(features)
    
    # --- Output Final Summary Report ---
    print("\n" + "="*50)
    print("📋 FINAL AI PIPELINE REPORT")
    print("="*50)
    print(f"Target Object          : {target_star}")
    print(f"Calculated Orbit Period: {features['period']:.4f} Days")
    print(f"Calculated Signal SNR  : {features['snr']:.2f}")
    print(f"Stellar Flux Skewness  : {features['flux_skew']:.4f}")
    print(f"AI Classification      : {prediction_label}")
    print(f"Calibrated Confidence  : {confidence * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_phase3()