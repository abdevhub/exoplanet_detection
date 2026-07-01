from wotan import flatten
import os

def clean_and_flatten(lightcurve):
    """Removes long-term stellar variations and cosmic ray spikes."""
    print("🧽 Polishing starlight data...")
    
    time = lightcurve.time.value
    flux = lightcurve.flux.value
    
    # Flatten the curve using a biweight filter (window of 0.5 days)
    flattened_flux, trend_flux = flatten(
        time, flux, method='biweight', window_length=0.5, return_trend=True
    )
    
    # Store the flattened flux back into a new lightkurve object
    clean_lc = lightcurve.copy()
    
    # Re-attach units if they existed, and scale flux_err
    if hasattr(lightcurve.flux, 'unit'):
        clean_lc.flux = flattened_flux * lightcurve.flux.unit
        if hasattr(lightcurve, 'flux_err') and lightcurve.flux_err is not None:
            clean_lc.flux_err = (lightcurve.flux_err.value / trend_flux) * lightcurve.flux.unit
    else:
        clean_lc.flux = flattened_flux
        if hasattr(lightcurve, 'flux_err') and lightcurve.flux_err is not None:
            clean_lc.flux_err = lightcurve.flux_err / trend_flux
    
    # Clip out extreme outliers (3 sigma above, 20 sigma below)
    clean_lc = clean_lc.remove_outliers(sigma_upper=3, sigma_lower=20)
    
    return clean_lc, trend_flux