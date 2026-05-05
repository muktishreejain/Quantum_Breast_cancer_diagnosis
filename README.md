# Adaptive Ensemble-Guided Hybrid Quantum-Classical Framework for Breast Cancer Diagnosis

This project implements a research-oriented hybrid pipeline for breast cancer diagnosis using an adaptive classical-to-quantum routing strategy on the Wisconsin Breast Cancer dataset.

## Overview

The framework combines:

- A **three-model classical ensemble**:
  - `RandomForestClassifier`
  - `LogisticRegression`
  - `GradientBoostingClassifier`
- A **Variational Quantum Classifier (VQC)** for uncertain cases only
- An **adaptive ambiguity score + threshold sweep** to optimize routing behavior

All experiments are designed to remain laptop-friendly while preserving methodological rigor.

## Methodology

### 1) Data and Feature Processing

- Input: Wisconsin Breast Cancer dataset (`data.csv`)
- Label mapping:
  - `M -> 1` (malignant)
  - `B -> 0` (benign)
- Preprocessing:
  - Standardization with `StandardScaler`
  - PCA reduction from **30 features to 4 features**
- Train/test split uses fixed reproducibility:
  - `test_size=0.2`
  - `stratify=y`
  - `random_state=42`

### 2) Classical Ensemble Predictions

For each test sample, the framework stores:

- `rf_pred`, `rf_conf`
- `lr_pred`, `lr_conf`
- `gb_pred`, `gb_conf`

where each confidence is the max class probability from the corresponding classifier.

Classical consensus prediction for non-ambiguous samples is computed by **majority vote** over RF/LR/GB.

### 3) Ambiguity Score Engine

For each sample:

- `average_confidence = mean([rf_conf, lr_conf, gb_conf])`
- `confidence_variance = variance([rf_conf, lr_conf, gb_conf])`
- `ambiguity_score = alpha * (1 - average_confidence) + beta * confidence_variance`

with:

- `alpha = 0.6`
- `beta = 0.4`

Higher ambiguity score indicates higher uncertainty.

### 4) Adaptive Threshold Optimization

Instead of a fixed confidence cutoff, the framework sweeps:

```python
threshold_values = np.arange(0.10, 0.71, 0.10)
```

For each threshold:

1. Route samples with `ambiguity_score > threshold` to the quantum stage.
2. Train a fresh VQC on ambiguous **training** samples only.
3. Predict ambiguous **test** samples with VQC.
4. Keep majority-vote ensemble predictions for non-ambiguous test samples.
5. Evaluate hybrid performance.

### 5) Quantum Model (Preserved Core)

The VQC configuration is intentionally kept aligned with the original lightweight setup:

- `ZZFeatureMap(feature_dimension=4, reps=1)`
- `RealAmplitudes(num_qubits=4, reps=1)`
- `COBYLA(maxiter=100)`
- Aer simulator backend via `qiskit_aer`

## Evaluation Outputs

For each threshold, the notebook records:

- `threshold`
- `hybrid_accuracy`
- `precision`
- `recall`
- `f1`
- `malignant_recall`
- `quantum_samples_used`

The best threshold is selected by **maximum hybrid accuracy**.

## Visualizations

The notebook generates:

1. **Ambiguity Score Distribution Histogram**
2. **Threshold vs Hybrid Accuracy** line plot
3. **Threshold vs Number of Quantum Routed Samples** line plot
4. **Bar chart comparison**:
   - Random Forest accuracy
   - Classical Ensemble accuracy
   - Best Adaptive Hybrid accuracy
5. **Confusion matrix** for the best hybrid model

## Expected Behavior

Compared to a fixed-threshold single-model gate, this adaptive ensemble-guided approach is intended to:

- Improve uncertainty estimation before quantum routing
- Reduce unnecessary quantum calls at stricter thresholds
- Preserve malignant-case sensitivity through threshold tuning
- Provide transparent trade-offs between hybrid accuracy and quantum workload

## Environment

Compatible with:

- Python 3.10
- `qiskit==0.45.0`
- `qiskit-machine-learning==0.6.0`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

No heavy external dependencies are required.

