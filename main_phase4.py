import os
import matplotlib.pyplot as plt
import numpy as np
from src.data_ingestion import download_star_data
from src.detrending import clean_and_flatten
from src.detection import find_planet_signal, fold_lightcurve
from src.features import extract_hybrid_features
from src.classifier import ExoplanetClassifier

def run_phase4():
    target_star = "TIC 25155310" # Test Case
    print(f"🚀 Starting Phase 4 Integration Pipeline for {target_star}")
    
    # 1. Pipeline Execution
    raw_lc = download_star_data(target_star)
    if raw_lc is None: return
    clean_lc, _ = clean_and_flatten(raw_lc)
    planet_params, _ = find_planet_signal(clean_lc)
    folded_lc = fold_lightcurve(clean_lc, planet_params['period'], planet_params['t0'])
    
    # 2. Extract Shape Parameters (Now with Trapezoid Fitting)
    features = extract_hybrid_features(clean_lc, planet_params, folded_lc)
    
    # --- 3. RECREATE ISRO SLIDE 03 PLOT ---
    print("🎨 Generating ISRO Verification Plot...")
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Convert phase from days to hours for the plot
    time_hours = folded_lc.time.value * 24
    
    # Plot observed data
    plt.scatter(time_hours, folded_lc.flux.value, s=4, color='#3b528b', label='Observed Data', alpha=0.6)
    
    # Plot the Red Best-Fit Line
    plt.plot(time_hours, features['best_fit_curve'], color='#e41a1c', linewidth=2.5, label='Best-Fit Transit Model')
    
    # Vertical lines for Ingress/Egress boundaries
    half_tot_hrs = features['t_tot'] / 2.0
    half_flat_hrs = features['t_flat'] / 2.0
    
    plt.axvline(-half_tot_hrs, color='gray', linestyle=':', alpha=0.5, label='Ingress Start / Egress End')
    plt.axvline(half_tot_hrs, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(-half_flat_hrs, color='green', linestyle=':', alpha=0.5, label='Ingress End / Egress Start')
    plt.axvline(half_flat_hrs, color='green', linestyle=':', alpha=0.5)
    
    plt.xlim(-features['t_tot']*2, features['t_tot']*2)
    plt.title(f"Transit Shape Fitting: {target_star}")
    plt.xlabel("Time From Mid-Transit (Hours)")
    plt.ylabel("Normalized Flux")
    plt.legend(loc='lower right')
    
    # Add the text box matching ISRO slides
    param_text = (f"Estimated Parameters\n\n"
                  f"Baseline Flux: {features['baseline_flux']:.6f}\n"
                  f"Transit Depth: {features['fit_depth']*100:.4f}%\n"
                  f"Total Duration (T_tot): {features['t_tot']:.3f} hours\n"
                  f"Ingress/Egress (T_in): {features['t_in']:.3f} hours\n"
                  f"Flat Bottom: {features['t_flat']:.3f} hours\n"
                  f"Shape Ratio (U vs V): {features['shape_ratio']:.3f}")
                  
    plt.text(1.05, 0.5, param_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.subplots_adjust(right=0.75) # Make room for text box
    plt.savefig(f"plots/{target_star.replace(' ', '_')}_shape_fit.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_phase4()