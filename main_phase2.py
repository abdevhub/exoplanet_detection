import sys
import os
import matplotlib.pyplot as plt

# Fix windows terminal emoji encoding issues
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from src.data_ingestion import download_star_data
from src.detrending import clean_and_flatten
from src.detection import find_planet_signal, fold_lightcurve

def run_phase2():
    target_star = "TIC 25155310" # WASP-126
    
    # --- PHASE 1 (Quick Recap) ---
    raw_lc = download_star_data(target_star)
    if raw_lc is None: 
        print("Failed to download data.")
        return
    clean_lc, trend_flux = clean_and_flatten(raw_lc)
    
    # --- PHASE 2 ---
    # 1. Find the hidden signal using TLS
    planet_params, tls_results = find_planet_signal(clean_lc)
    
    # 2. Fold the light curve to stack the transits
    folded_lc = fold_lightcurve(clean_lc, planet_params['period'], planet_params['t0'])
    
    # --- VISUALIZATION ---
    print("🎨 Generating Phase 2 plots...")
    os.makedirs("plots", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: The full cleaned timeline (we draw red lines where the planet should be)
    ax1.scatter(clean_lc.time.value, clean_lc.flux.value, s=1, color='blue', alpha=0.5)
    
    # Add vertical red lines to show the rhythm we found
    transit_times = tls_results.transit_times
    for t in transit_times:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.4)
        
    ax1.set_title(f"Clean Data + Planet Rhythm (P={planet_params['period']:.2f}d)")
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("Normalized Flux")
    
    # Plot 2: The Phase-Folded (Stacked) Light Curve
    ax2.scatter(folded_lc.time.value, folded_lc.flux.value, s=3, color='black', alpha=0.5)
    
    # Zoom in tightly on the transit shape
    ax2.set_xlim(-0.2, 0.2) 
    ax2.set_title("Zoomed & Phase Folded (The Transit Shape)")
    ax2.set_xlabel("Phase (Days from Mid-Transit)")
    ax2.axvline(0, color='gray', linestyle=':') # Center line
    
    # Add a text box with our extracted features for the AI
    text_info = (f"Extracted Features:\n"
                 f"Period: {planet_params['period']:.3f} d\n"
                 f"Duration: {planet_params['duration']:.2f} hrs\n"
                 f"Depth: {planet_params['depth']*100:.2f} %\n"
                 f"SNR: {planet_params['snr']:.1f}")
    ax2.text(0.05, 0.05, text_info, transform=ax2.transAxes, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_path = f"plots/{target_star.replace(' ', '_')}_phase2.png"
    plt.savefig(plot_path)
    print(f"✨ Success! Check out your stacked transit shape here: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_phase2()