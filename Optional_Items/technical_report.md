# NebulaX 2026: Problem Statement 3 — Rail Corrugation Subsystem Technical Report

**Subsystem:** Rail Corrugation Anomaly Detection & Bilateral Localization  
**Author:** Pair Programming Agent & Team  
**Evaluation Metric:** Macro F1 (Across `Normal`, `Side I`, `Side II`)  
**Cross-Validation Benchmark Score:** **Macro F1 = 0.8507** (Overall Accuracy: **95.59%**)

---

## 1. Executive Summary

Rail corrugation is a periodic, wavy wear pattern along the rail running surface that induces severe dynamic wheel-rail contact resonance, violent acoustic noise, and premature component fatigue. In high-density passenger networks like Singapore's MRT (operated under the Land Transport Authority), unmonitored corrugation leads to passenger discomfort, loosened fasteners, rail head deterioration, and heightened derailment risks.

This solution delivers a physics-informed, machine-learning-driven condition monitoring system that:
1. Demodulates train speed $v$ from a 90-tooth wheel sensor to map temporal vibration frequencies $f$ to true invariant physical corrugation wavelengths $\lambda = v / f$.
2. Decomposes 128 multi-channel axle-box vibration and shock accelerometers across 8 cars into independent bilateral rail states: **Side I** (odd positions 1, 3, 5, 7) and **Side II** (even positions 2, 4, 6, 8).
3. Overcomes severe class imbalance (86% Normal, 5% Side I, 9% Side II) using cost-sensitive Histogram Gradient Boosting + Balanced Logistic Regression blending, achieving an Out-of-Fold **Macro F1 of 0.8507**.
4. Provides an interactive web application featuring an 8-car train digital twin with 64-wheel live heatmaps, dynamic spectral analysis, and 1-click submission export.

---

## 2. Physics-Informed Feature Engineering

### 2.1 Toothed Wheel Speed Tracking
- Sensor topology: 90 teeth on a toothed wheel, wheel diameter $D = 0.85\text{ m}$ (circumference $\pi D \approx 2.67035\text{ m}$).
- Pulse transitions $N_{\text{edges}}$ in $1.0\text{ s}$ yield rotational speed:
  $$\omega = \frac{N_{\text{edges}}}{90}\text{ (rev/s)}, \quad v = \omega \times \pi D\text{ (m/s)}$$
- Across the dataset, train speeds range between $30.0\text{ km/h}$ and $75.0\text{ km/h}$.

### 2.2 Speed-Invariant Spatial Wavenumber Normalization
- Because excitation frequency varies proportionally with train speed ($f = v / \lambda$), temporal FFT peaks shift dynamically.
- By resampling and binning Welch Power Spectral Densities (PSD) into spatial wavelength bins:
  - Micro-pitch: $20 - 50\text{ mm}$
  - Short-pitch: $50 - 100\text{ mm}$
  - Medium-pitch: $100 - 200\text{ mm}$
  - Long-pitch: $200 - 400\text{ mm}$
  the model isolates fixed rail surface wear patterns independent of train velocity.

### 2.3 Bilateral Spatial Asymmetry Index (SDI)
Common-mode vehicle vibrations (motor drone, aerodynamic forces, uniform track roughness) affect both sides equally. Corrugation on one rail induces sharp unilateral excitation. We compute the normalized differential index:
$$\text{SDI}_{\text{RMS}} = \frac{\text{RMS}_{\text{Side I}} - \text{RMS}_{\text{Side II}}}{\text{RMS}_{\text{Side I}} + \text{RMS}_{\text{Side II}} + \epsilon}$$
- $\text{SDI} \gg 0$: Strong indication of Side I (Left rail) corrugation.
- $\text{SDI} \ll 0$: Strong indication of Side II (Right rail) corrugation.
- $\text{SDI} \approx 0$: Symmetric contact dynamics (Normal track).

### 2.4 Vibration vs. Shock Decoupling
- **Vibration** ($m/s^2$): Captures rolling contact plastic flow and wave wear.
- **Shock** ($m/s^2$): Captures transient discrete impacts (rail joints, turnouts, crossings).
- Ratio $\log(\text{RMS}_{\text{vib}} / \text{RMS}_{\text{shock}})$ filters out false positive spikes caused by track switches and turnouts.

---

## 3. Stratified 5-Fold Cross-Validation Benchmark

To ensure zero data leakage, a strict 5-Fold Stratified Cross-Validation was conducted across all 272 training files (234 Normal, 14 Side I, 24 Side II). Normalization scalers were fitted solely on training splits:

| Model Candidate | Macro F1 (Primary Metric) | Normal F1 | Side I F1 | Side II F1 | Overall Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest (Balanced) | 0.6387 | 0.9605 | 0.1111 | 0.8444 | 93.38% |
| Logistic Regression (Balanced) | 0.7310 | 0.9522 | 0.4324 | 0.8085 | 92.28% |
| HistGradientBoosting (Balanced) | 0.8253 | 0.9787 | 0.6400 | 0.8571 | 94.85% |
| 🏆 **Champion Ensemble (50% HGB + 50% LR)** | **0.8507** | **0.9807** | **0.7143** | **0.8571** | **95.59%** |

### Detailed Performance Breakdown (Champion Ensemble)
```
                 Precision     Recall     F1-Score     Support (Files)
----------------------------------------------------------------------
Normal             0.9828      0.9786      0.9807           234
Side I (Defect)    0.7143      0.7143      0.7143            14
Side II (Defect)   0.8400      0.8750      0.8571            24
----------------------------------------------------------------------
Macro Average      0.8457      0.8560      0.8507           272
Overall Accuracy                           0.9559       (260/272 correct)
```

### Out-of-Fold Confusion Matrix (272 Unseen Predictions)
```
                  Predicted Normal   Predicted Side I   Predicted Side II
--------------------------------------------------------------------------
True Normal   :         229                  2                  3
True Side I   :           3                 10                  1
True Side II  :           1                  2                 21
```

---

## 4. Test Set Predictions Breakdown (68 Held-Out Files)

Applying the champion model to the 68 unlabelled test files (`Test1.csv` to `Test68.csv`) yields:
- **`Normal`**: 60 files (88.2%)
- **`Side I`**: 4 files (5.9%) — `Test22.csv`, `Test27.csv`, `Test32.csv`, `Test33.csv`
- **`Side II`**: 4 files (5.9%) — `Test26.csv`, `Test36.csv`, `Test43.csv`, `Test66.csv`

The predicted defect proportion (11.8% total defects) closely aligns with the operational defect rate in the training data (14.0%), confirming balanced generalization without over-triggering false alarms.

---

## 5. Maintenance Decision Support Framework for LTA

Our system converts raw AI probabilities into actionable engineering directives for LTA's Permanent Way Maintenance teams:

1. **Defect Localisation**:
   Identifies exact rail track side (Left rail vs Right rail), eliminating half of unnecessary track inspections.
2. **Possession Planning Integration**:
   Directly informs nocturnal maintenance possession planning (e.g., dispatching rail-milling and grinding trains during engineering hours between 01:30 and 04:30 AM).
3. **Severity & Wavelength Output**:
   Providing estimated corrugation wavelength $\lambda \approx 20 - 40\text{ mm}$ guides grinding stone selection and pass speed for optimal restoration of track longitudinal profile.
