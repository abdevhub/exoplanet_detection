# 🌌 Exoplanet Detection Pipeline

A modular, state-of-the-art Python pipeline to ingest starlight curves from space telescopes (like TESS), clean and filter stellar trends, detect periodic transits, fit geometric models, and classify exoplanet candidates using calibrated Machine Learning.

---

## 🚀 Overview

This pipeline is designed to search for and identify exoplanet candidates from light curves retrieved from the **MAST (Mikulski Archive for Space Telescopes)** database using the TESS mission data. It implements a multi-stage approach combining astrophysics-based signal processing (Wotan, Transit Least Squares) with machine learning classification (XGBoost).

### Key Features
- **Data Ingestion**: Programmatic search and download of TESS light curves using `lightkurve`.
- **Detrending**: Stellar trend line flattening using biweight filters via `wotan`.
- **Transit Detection**: Detection of periodic transit signals using **Transit Least Squares (TLS)** (with a fast **Box Least Squares (BLS)** fallback via Astropy).
- **Geometric Transit Fitting**: Fit a physical trapezoidal transit model to determine transit parameters (Ingress/Egress duration, flat bottom width, transit depth).
- **ML Classification**: Features are fed into an **XGBoost Classifier** combined with **Isotonic Regression** for statistically calibrated classification confidence.

---

## 📁 Repository Structure

```text
├── data/
│   ├── processed/            # Saved feature datasets (e.g. extracted_features.csv)
│   └── raw/                  # Downloaded raw TESS lightcurves (ignored by Git)
├── plots/                    # Generated pipeline plots and metrics (ignored by Git)
├── src/
│   ├── __init__.py           # Package initializer
│   ├── classifier.py         # Calibrated XGBoost machine learning model
│   ├── data_ingestion.py     # MAST/TESS data downloader via Lightkurve
│   ├── detection.py          # TLS / BLS transit search engine and folder
│   ├── detrending.py         # Signal cleaning & biweight detrending via Wotan
│   ├── features.py           # Hybrid features assembly and engineering
│   └── shape_fitting.py      # SciPy-based trapezoidal transit model fitting
├── isro_training_targets.csv # Input targets CSV containing TIC IDs and labels
├── requirements.txt          # Python dependencies
├── main_phase1.py            # Phase 1: Ingestion & Detrending Demo
├── main_phase2.py            # Phase 2: Transit Detection & Folding
├── main_phase3.py            # Phase 3: Basic ML Inference Integration
├── main_phase4.py            # Phase 4: Trapezoid Shape Fitting Visualizer
└── main_phase5.py            # Phase 5: Batch Processing, ML Training & Evaluation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository** (or download the source files).
2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows PowerShell/CMD
   source .venv/bin/activate   # On macOS/Linux
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*Note: The pipeline will check if `transitleastsquares` is installed. If it is not found, the code automatically falls back to standard Astropy `BoxLeastSquares` (BLS).*

---

## 🛰️ Pipeline Phases

The pipeline is split into 5 chronological developmental phases. Each phase builds upon the previous, offering full modularity.

### 📈 Phase 1: Ingestion & Detrending
Downloads a target star's light curve, detrends the stellar background variation using a biweight filter, and removes outliers. Saves a visual validation plot to `plots/`.
```bash
python main_phase1.py
```

### 🎯 Phase 2: Transit Signal Search
Scans the cleaned light curve for periodic transit dips. Once the transit period and mid-transit epoch ($T_0$) are identified, the light curve is phase-folded to reveal the transit shape.
```bash
python main_phase2.py
```

### 🧠 Phase 3: Hybrid Feature Extraction & Inference
Calculates astrophysical features (e.g., period, SNR, flux skewness) and feeds them into the calibrated ML Classifier to output a classification label and probability confidence.
```bash
python main_phase3.py
```

### 📐 Phase 4: Trapezoid Shape Fitting
Upgrades Phase 3 with a custom geometrical model. Fits a trapezoidal shape to the folded light curve to calculate:
- **Baseline Flux**
- **Transit Depth**
- **Total Duration ($T_{tot}$)**
- **Ingress/Egress Duration ($T_{in}$)**
- **Flat Bottom Duration ($T_{flat}$)**
- **Shape Ratio ($T_{in} / T_{tot}$)** (distinguishes U-shaped exoplanets from V-shaped binary stars)
```bash
python main_phase4.py
```

### 📊 Phase 5: Batch Processing & Training
Loops through a target csv list (`isro_training_targets.csv`), downloads and runs the pipeline for each target, saves a compiled dataset (`data/processed/extracted_features.csv`), and trains the XGBoost model using an 80/20 train-test split. Outputs a classification report and Confusion Matrix.
```bash
python main_phase5.py
```

---

## 📐 Mathematical Model: Geometrical Trapezoid Transit

The trapezoidal model in `src/shape_fitting.py` models the normalized flux $F(t)$ of the star as:

$$
F(t) = \text{baseline} - \text{depth} \times K(t)
$$

Where the scale factor $K(t)$ is:
- **$1$** during flat bottom (when the planet is fully inside the stellar disc)
- **Linear interpolation** between $0$ and $1$ during ingress and egress
- **$0$** out of transit

This model provides physical parameters such as the **Shape Ratio ($T_{in}/T_{tot}$)**, which is critical for identifying grazing eclipsing binary star systems (V-shape) from genuine planetary transits (U-shape).
