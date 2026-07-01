import numpy as np
from src.shape_fitting import fit_transit_shape

def extract_hybrid_features(clean_lc, planet_params, folded_lc):
    """
    Now upgraded with Geometric Shape Fitting (Phase 4 integration)
    """
    print("🧠 Extracting hybrid feature matrix...")
    
    features = {
        'period': planet_params['period'],
        'snr': planet_params['snr']
    }
    
    flux = clean_lc.flux.value
    features['flux_std'] = np.std(flux)
    features['flux_skew'] = float(np.mean((flux - np.mean(flux))**3) / np.std(flux)**3) if np.std(flux) > 0 else 0
    
    # --- Phase 4 Shape Fitting Integration ---
    time_folded = folded_lc.time.value
    flux_folded = folded_lc.flux.value
    
    shape_params = fit_transit_shape(
        time_folded, 
        flux_folded, 
        initial_duration=planet_params['duration']/24, 
        initial_depth=planet_params['depth']
    )
    
    if shape_params:
        features['t_tot'] = shape_params['t_tot']
        features['t_in'] = shape_params['t_in']
        features['t_flat'] = shape_params['t_flat']
        features['shape_ratio'] = shape_params['shape_ratio']
        features['fit_depth'] = shape_params['fit_depth']
        features['baseline_flux'] = shape_params['baseline_flux']
        features['best_fit_curve'] = shape_params['best_fit_curve'] 
    else:
        features['t_tot'] = planet_params['duration']
        features['t_in'] = planet_params['duration'] / 2.0
        features['t_flat'] = 0
        features['shape_ratio'] = 0.5
        features['fit_depth'] = planet_params['depth']
        features['baseline_flux'] = 1.0
        features['best_fit_curve'] = np.ones_like(time_folded)
        
    return features