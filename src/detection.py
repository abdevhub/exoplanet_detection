try:
    import transitleastsquares as tls
    HAS_TLS = True
except ImportError:
    HAS_TLS = False
    from astropy.timeseries import BoxLeastSquares

import numpy as np

class BLSResultsFallback:
    def __init__(self, results, time, idx):
        self.results = results
        self.period = float(results.period[idx])
        self.T0 = float(results.transit_time[idx])
        # TLS results.depth is the normalized flux level at the bottom (e.g. 0.98),
        # whereas BLS results.depth is the blocked depth (e.g. 0.02).
        self.depth = 1.0 - float(results.depth[idx])
        self.duration = float(results.duration[idx])
        self.snr = float(results.depth_snr[idx])
        
        # Calculate transit times in the time range of observations
        t_min = np.min(time)
        t_max = np.max(time)
        n_start = int(np.ceil((t_min - self.T0) / self.period))
        n_end = int(np.floor((t_max - self.T0) / self.period))
        self.transit_times = [self.T0 + n * self.period for n in range(n_start, n_end + 1)]

def find_planet_signal(clean_lc):
    """
    Searches the cleaned light curve for a repeating planet shadow.
    Returns a dictionary of the planet's shape parameters.
    """
    if HAS_TLS:
        print("🔍 Scanning for the planet's heartbeat (TLS Search)...")
        print("⏳ (This might take a minute, we are checking thousands of possible orbits!)")
        
        # 1. Extract the raw numbers safely
        time = clean_lc.time.value
        flux = clean_lc.flux.value
        
        # 2. Give the data to the TLS tool
        # We must provide the uncertainties (flux_err) for the best results!
        model = tls.transitleastsquares(time, flux, clean_lc.flux_err.value)
        
        # 3. Run the search! 
        # We turn off the progress bar to keep our terminal clean
        # use_threads=4 is a middle ground to speed things up without causing MemoryError
        results = model.power(show_progress_bar=False, use_threads=4)
        
        # 4. Extract the physical shape numbers the AI will need later
        planet_params = {
            'period': results.period,                # How many days in a year?
            't0': results.T0,                        # When did the first transit happen?
            'depth': 1.0 - results.depth,            # How much light was blocked?
            'duration': results.duration * 24,       # Total time in hours (T_tot)
            'snr': results.snr                       # Signal-to-Noise Ratio (Is it a strong signal?)
        }
        
        print(f"🎯 Signal Found! Period: {planet_params['period']:.3f} days | SNR: {planet_params['snr']:.1f}")
        return planet_params, results
    else:
        print("🔍 Scanning for the planet's heartbeat (BoxLeastSquares Fallback)...")
        print("⏳ (Running fast search via astropy.timeseries.BoxLeastSquares...)")
        
        # 1. Extract the raw numbers safely
        time = clean_lc.time.value
        flux = clean_lc.flux.value
        flux_err = clean_lc.flux_err.value if hasattr(clean_lc, 'flux_err') and clean_lc.flux_err is not None else None
        
        # 2. Setup BoxLeastSquares model
        model = BoxLeastSquares(time, flux, flux_err)
        
        # 3. Run the search
        results = model.autopower(duration=0.1) # Default duration guess: 0.1 days
        
        # Find index of peak power
        idx = np.argmax(results.power)
        
        # 4. Create fallback results wrapper
        fallback_results = BLSResultsFallback(results, time, idx)
        
        # 5. Extract the physical shape numbers
        planet_params = {
            'period': fallback_results.period,
            't0': fallback_results.T0,
            'depth': 1.0 - fallback_results.depth,  # Will be 1.0 - (1.0 - depth) = depth
            'duration': fallback_results.duration * 24,
            'snr': fallback_results.snr
        }
        
        print(f"🎯 Signal Found! Period: {planet_params['period']:.3f} days | SNR: {planet_params['snr']:.1f}")
        return planet_params, fallback_results

def fold_lightcurve(clean_lc, period, t0):
    """
    Folds the timeline over itself so all the transits stack up.
    """
    print("📂 Folding the light curve like origami...")
    # Lightkurve has a built-in folding tool!
    folded_lc = clean_lc.fold(period=period, epoch_time=t0)
    return folded_lc