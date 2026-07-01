import lightkurve as lk
import os

def download_star_data(tic_id, output_dir="data/raw"):
    """Downloads a raw light curve from TESS using a Star ID."""
    print(f"🔭 Searching for {tic_id}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for high-cadence data from the SPOC pipeline (or any available)
    search_result = lk.search_lightcurve(tic_id)
    if len(search_result) == 0:
        print(f"❌ Could not find data for {tic_id}")
        return None
        
    # Download the first available sector data
    print(f"⬇️ Downloading data from MAST archive...")
    lc = search_result[0].download(download_dir=output_dir)
    return lc.remove_nans()