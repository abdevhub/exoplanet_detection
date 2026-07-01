import numpy as np
from scipy.optimize import curve_fit

def trapezoid_model(t, depth, t_tot, t_in, baseline):
    """
    Geometrical Trapezoid Transit Model matching ISRO slide 03.
    t: phase/time array (centered at 0)
    depth: transit depth
    t_tot: total duration
    t_in: ingress/egress duration
    baseline: out-of-transit flux level
    """
    flux = np.ones_like(t) * baseline
    
    # Ensure physical constraints (ingress cannot be larger than half the total duration)
    t_in = max(1e-5, min(t_in, t_tot / 2.0))
    t_flat = t_tot - 2 * t_in
    
    half_tot = t_tot / 2.0
    half_flat = t_flat / 2.0
    
    # 1. Ingress
    idx_ingress = (t >= -half_tot) & (t < -half_flat)
    if np.any(idx_ingress):
        flux[idx_ingress] = baseline - depth * ((t[idx_ingress] + half_tot) / t_in)
    
    # 2. Flat bottom (Planet fully covering the star)
    idx_flat = (t >= -half_flat) & (t <= half_flat)
    if np.any(idx_flat):
        flux[idx_flat] = baseline - depth
    
    # 3. Egress
    idx_egress = (t > half_flat) & (t <= half_tot)
    if np.any(idx_egress):
        flux[idx_egress] = baseline - depth * (1 - (t[idx_egress] - half_flat) / t_in)
        
    return flux

def fit_transit_shape(folded_time, folded_flux, initial_duration, initial_depth):
    """
    Uses scipy to find the best-fit trapezoid shape parameters.
    Returns a dictionary of the optimized shape features.
    """
    print("📐 Fitting geometric transit shape (Trapezoid Model)...")
    
    # Initial guesses for the optimizer: [depth, t_tot, t_in, baseline]
    # We guess t_in is about 20% of the total duration initially
    p0 = [initial_depth, initial_duration, initial_duration * 0.2, 1.0]
    
    # Bounds: depth(>0), t_tot(>0), t_in(>0), baseline(~1.0)
    bounds = (
        [0, 0, 0, 0.99], 
        [1.0, initial_duration * 3, initial_duration, 1.01]
    )
    
    try:
        popt, _ = curve_fit(trapezoid_model, folded_time, folded_flux, p0=p0, bounds=bounds)
        
        depth, t_tot, t_in, baseline = popt
        t_flat = t_tot - (2 * t_in)
        
        # Calculate U-shape vs V-shape ratio for the AI
        shape_ratio = t_in / t_tot
        
        return {
            'baseline_flux': baseline,
            'fit_depth': depth,
            't_tot': t_tot * 24, # Convert to hours for readability
            't_in': t_in * 24,   # Convert to hours
            't_flat': max(0, t_flat * 24), # Convert to hours
            'shape_ratio': shape_ratio,
            'best_fit_curve': trapezoid_model(folded_time, *popt)
        }
    except Exception as e:
        print(f"⚠️ Fitting failed: {e}")
        return None