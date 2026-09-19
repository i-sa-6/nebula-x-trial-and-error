# NebulaX 2026: Problem Statement 3 — Train Condition Monitoring (Rail Corrugation Subsystem)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Macro F1](https://img.shields.io/badge/5--Fold%20Macro%20F1-0.8507-brightgreen.svg)]()
[![Overall Accuracy](https://img.shields.io/badge/CV%20Accuracy-95.59%25-success.svg)]()
[![Deliverable](https://img.shields.io/badge/predictions.zip-Verified%20Compliant-blue.svg)]()

Production-grade automated condition monitoring system developed for the **Singapore Land Transport Authority (LTA)** track maintenance operations under **NebulaX 2026 Hackathon (Problem Statement 3)**.

---

## 🏆 Key Performance Metrics (5-Fold Stratified Cross-Validation)

| Model Architecture | Macro F1 (Competition Metric) | Normal F1 | Side I F1 | Side II F1 | Overall Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| Balanced Random Forest | 0.6387 | 0.9605 | 0.1111 | 0.8444 | 93.38% |
| Balanced Logistic Regression | 0.7310 | 0.9522 | 0.4324 | 0.8085 | 92.28% |
| HistGradientBoosting (Balanced) | 0.8253 | 0.9787 | 0.6400 | 0.8571 | 94.85% |
| 🏆 **Champion Ensemble (50% HGB + 50% LR)** | **0.8507** | **0.9807** | **0.7143** | **0.8571** | **95.59%** |

### 5-Fold Confusion Matrix (Total N = 272 Training Files)
```
                  Predicted Normal   Predicted Side I   Predicted Side II
--------------------------------------------------------------------------
True Normal   :         229                  2                  3
True Side I   :           3                 10                  1
True Side II  :           1                  2                 21
```

---

## 🚆 Architecture & Physics Engine

1. **Tooth Wheel Pulse Speed Demodulation**:
   - Computes train velocity $v$ from 90-tooth wheel pulse transitions:
     $$v = \frac{\pi \cdot D}{N} \cdot \frac{\Delta \text{pulses}}{\Delta t}$$
   - Maps temporal excitation frequency $f$ to invariant physical spatial corrugation wavelength $\lambda = v / f$, separating short-pitch corrugation ($\lambda \in [20, 40]\text{ mm}$) from operational train speed changes.

2. **64-Wheelset Bilateral Asymmetry Extraction**:
   - Decomposes 128 axle-box vibration and shock accelerometers across 8 cars into independent bilateral rail states:
     - **Side I (Left Rail)**: Axle-box positions 1, 3, 5, 7
     - **Side II (Right Rail)**: Axle-box positions 2, 4, 6, 8
   - Computes Bilateral Differential Asymmetry Index ($\text{SDI}$), per-car leading-to-trailing differentials, and Welch PSD band powers.

3. **Champion Classifier**:
   - Soft probability ensemble combining 50% Histogram Gradient Boosting and 50% Balanced Logistic Regression, optimized for class-imbalance with out-of-fold threshold calibration.

---

## 📁 Repository Structure

```
├── app/
│   ├── backend/
│   │   ├── server.py              # Zero-dependency HTTP API + SSE token streaming server
│   │   └── gemini_streamer.py     # generateContentStream integration for AI advisory
│   ├── frontend/
│   │   ├── index.html             # Fleet Predictions & Triage Dashboard (Landing Page)
│   │   ├── landing.js             # Predictions table controller, filters & CSV upload handler
│   │   ├── twin.html              # Live 8-Car Train Digital Twin & Streaming Recommendations
│   │   ├── accuracy.html          # Dedicated Model Accuracy & Benchmark Explorer (linked to GitHub)
│   │   ├── app.js                 # Train schematic & SSE stream consumer
│   │   ├── accuracy.js            # Interactive 68 test files filter & metrics controller
│   │   └── style.css              # GPU-composited high-tech glassmorphic design system
│   └── run_app.py                 # Web application launcher (supports local $PORT & Cloud Run)
├── Dockerfile                     # Optimized container definition for Google Cloud Run
├── cloudbuild.yaml                # Automated Google Cloud Build CI/CD deployment pipeline
├── deploy_gcp.sh                  # Automated Google Cloud deployment engine
├── data/
│   ├── Train_Labels.csv           # Ground-truth training labels (272 recordings)
│   └── train_features.csv         # Pre-extracted 134 physical features for fast training
├── models/
│   └── rail_corrugation_champion.pkl # Champion Ensemble Model Bundle (653 KB)
├── Optional_Items/
│   └── technical_report.md        # Comprehensive engineering report for LTA P-Way judging
├── submission/
│   ├── rail_predictions.csv       # Official 68 test predictions (60 Normal, 4 Side I, 4 Side II)
│   ├── rail_predictions_detailed.csv # Detailed confidence, speed, and probability breakdown
│   └── predictions.zip            # Validated competition submission package
├── src/
│   ├── config.py                  # Geometric layout & physical constants
│   ├── speed_estimator.py         # Pulse transition speed estimator
│   ├── feature_extraction.py      # Time-domain, frequency, and SDI feature extractor
│   ├── models.py                  # Champion Ensemble model definition
│   ├── train_model.py             # 5-Fold Stratified Cross-Validation pipeline
│   ├── evaluate.py                # Submission ZIP validator simulating judge_leaderboard.py
│   ├── download_train.py          # Automated parallel downloader for 272 training CSVs
│   └── download_test.py           # Automated parallel downloader for 68 test CSVs
└── predict.py                     # Official CLI inference entrypoint (--input and --output)
```

---

## ☁️ Google Cloud Deployment (NebulaX Rule Compliance)

The entire application is containerized and deployed on **Google Cloud Run** in accordance with NebulaX judging requirements:

- 🌐 **Live Cloud Service**: [https://nebula-rail-twin-863881211691.asia-southeast1.run.app](https://nebula-rail-twin-863881211691.asia-southeast1.run.app)
- **GCP Region**: `asia-southeast1` (Singapore)
- **Container Registry**: `gcr.io/qwiklabs-gcp-01-c57489cc9b86/nebula-rail-twin:latest`
- **One-Click Deploy**: `./deploy_gcp.sh`

---

## 🚀 Quickstart & Usage

### 1. Launch the Live Interactive Web Application
```bash
python3 app/run_app.py 8080
```
Open your browser at:
- **Fleet Predictions & Triage (Landing Page)**: [http://127.0.0.1:8080/](http://127.0.0.1:8080/)
- **Live 8-Car Diagnostic Twin**: [http://127.0.0.1:8080/twin.html](http://127.0.0.1:8080/twin.html)
- **Model Accuracy Explorer**: [http://127.0.0.1:8080/accuracy.html](http://127.0.0.1:8080/accuracy.html)

*(Binding to `127.0.0.1` bypasses macOS IPv6 DNS resolution latency for instant 0ms responses).*

### 2. Run Inference CLI (Mandatory Hackathon Specification)
Run inference on any test file or directory:
```bash
# Evaluate all 68 test files into submission/rail_predictions.csv
python3 predict.py --input data/Test --output submission/rail_predictions.csv
```

### 3. Verify Official Submission Package
```bash
python3 src/evaluate.py
```
Outputs validation check confirming row counts (68 rows), format, and valid classes.

### 4. (Optional) Re-download Raw Datasets
To re-fetch the raw 5.5 GB uncompressed datasets from the competition repo:
```bash
python3 src/download_train.py
python3 src/download_test.py
```

---

## 📊 Held-Out Test Set Predictions (68 Files)

- **Normal**: 60 files (88.2%)
- **Side I Corrugation (Left Rail)**: 4 files (5.9%)
  - `Test22.csv` (Confidence 54.7%, Speed 41.7 km/h, $\text{SDI} = +0.114$)
  - `Test27.csv` (Confidence 98.2%, Speed 40.6 km/h, $\lambda \approx 23.6\text{ mm}$)
  - `Test32.csv` (Confidence 59.9%, Speed 45.2 km/h, $\text{SDI} = +0.063$)
  - `Test33.csv` (Confidence 99.3%, Speed 46.1 km/h, $\text{SDI} = +0.046$)
- **Side II Corrugation (Right Rail)**: 4 files (5.9%)
  - `Test26.csv` (Confidence 100.0%, Speed 57.5 km/h, $\text{SDI} = -0.171$)
  - `Test36.csv` (Confidence 45.7%, Speed 54.9 km/h, $\lambda \approx 173.5\text{ mm}$)
  - `Test43.csv` (Confidence 99.9%, Speed 56.4 km/h, $\text{SDI} = -0.101$)
  - `Test66.csv` (Confidence 100.0%, Speed 47.2 km/h, $\text{SDI} = -0.065$)

---

## 👥 Author
**Vikidaman** (<kannanvigneshraj@gmail.com>)
Developed for NebulaX 2026 Hackathon.
