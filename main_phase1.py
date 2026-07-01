import os
import matplotlib.pyplot as plt
from src.data_ingestion import download_star_data
from src.detrending import clean_and_flatten

def run_pipeline():
    # Let's test with a famous planet-hosting star (WASP-126)
    target_star = "TIC 25155310"
    
    # 1. Download
    raw_lc = download_star_data(target_star)
    if raw_lc is None:
        return
        
    # 2. Clean and Flatten
    clean_lc, trend_flux = clean_and_flatten(raw_lc)
    
    # 3. Save a Diagnostic Plot
    print("🎨 Generating verification plots...")
    os.makedirs("plots", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot Raw + Trend line
    ax1.scatter(raw_lc.time.value, raw_lc.flux.value, s=1, color='gray', alpha=0.5)
    ax1.plot(raw_lc.time.value, trend_flux, color='red', label='Stellar Trend Line')
    ax1.set_title(f"Raw Starlight with Trend Line ({target_star})")
    ax1.legend()
    
    # Plot Cleaned data
    ax2.scatter(clean_lc.time.value, clean_lc.flux.value, s=1, color='blue')
    ax2.set_title("Flattened, Science-Ready Light Curve")
    
    plt.tight_layout()
    plot_path = f"plots/{target_star.replace(' ', '_')}_phase1.png"
    plt.savefig(plot_path)
    print(f"✨ Success! Your clean dataset visualization is saved at: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()