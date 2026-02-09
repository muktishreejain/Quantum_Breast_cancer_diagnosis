# 🧬 Hybrid Quantum-Classical Breast Cancer Classification

### 🏆 1st Place — Advanced Track

A hybrid machine learning pipeline that combines a **Random Forest** classifier with a **Variational Quantum Classifier (VQC)** to improve breast cancer diagnosis — particularly on the cases classical models find most uncertain. This project achieved **93% overall accuracy** by offloading low-confidence predictions to a quantum classifier.

Built with [Qiskit](https://qiskit.org/) and [scikit-learn](https://scikit-learn.org/).

---

## 💡 Inspiration

We were inspired by the potential of quantum computing to enhance real-world applications, especially in critical fields like healthcare. This project explores whether quantum machine learning can add value where classical models struggle — specifically on low-confidence, ambiguous predictions in breast cancer classification.

Key themes we presented to judges include data efficiency and computational tradeoffs in early-stage quantum ML development — from simulation bottlenecks and feature dimensionality constraints to iterative circuit design.

---

## 🎥 Demo

[![Watch the Demo](https://img.youtube.com/vi/7Q_QIN7CAjE/maxresdefault.jpg)](https://www.youtube.com/watch?v=7Q_QIN7CAjE)

▶️ *Click the image above to watch the full video walkthrough on YouTube.*

---

## 🔬 How It Works

The pipeline follows a **confidence-based hybrid** approach:

1. **Data Preprocessing** — The Wisconsin Breast Cancer dataset is loaded, cleaned, standardized, and reduced to 4 principal features for quantum compatibility.
2. **Classical Baseline** — A Random Forest classifier (100 estimators) is trained on the full training set, achieving **~88.6% accuracy**.
3. **Ambiguity Detection** — Test samples where the Random Forest's max predicted probability falls below a **0.7 confidence threshold** are flagged as ambiguous (9 out of 114 test samples).
4. **Quantum Refinement** — A Variational Quantum Classifier is trained on these ambiguous cases using:
   - **ZZFeatureMap** (4 qubits, 1 rep) for data encoding
   - **RealAmplitudes** ansatz (4 qubits, 1 rep) for the trainable circuit
   - **COBYLA** optimizer (100 iterations) with cross-entropy loss
5. **Hybrid Prediction** — The final model uses Random Forest predictions for high-confidence cases and VQC predictions for ambiguous ones.

---

## 📊 Results

| Model                  | Scope             | Accuracy |
|------------------------|-------------------|----------|
| Random Forest          | Full test set     | 88.6%    |
| Random Forest          | Ambiguous cases   | 55.6%    |
| VQC                    | Ambiguous cases   | 77.8%    |
| **Hybrid (RF + VQC)**  | **Full test set** | **93%**  |

The VQC significantly outperformed Random Forest on the ambiguous subset, and the hybrid model improved overall accuracy.

---

## 📈 Visualizations

The notebook includes several visualizations:

- **Random Forest Confidence Distribution** — histogram showing prediction confidence across the test set with the ambiguity threshold marked
- **VQC Confusion Matrix** — performance breakdown on ambiguous cases (Benign vs. Malignant)
- **Hybrid Accuracy Comparison** — bar chart comparing RF, VQC, and Hybrid accuracy
- **VQC Loss Curve** — COBYLA optimization progress over 100 iterations
- **Quantum Circuit Diagrams** — decomposed Feature Map and Ansatz circuits

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Qiskit 0.45.0** / Qiskit Aer / Qiskit Machine Learning 0.6.0
- **scikit-learn** — Random Forest, preprocessing, evaluation metrics
- **matplotlib** / **seaborn** — visualization
- **pandas** / **numpy** — data handling

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install qiskit==0.45.0 qiskit-aer qiskit-machine-learning==0.6.0 scikit-learn matplotlib seaborn pandas numpy
```

### Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Place the Wisconsin Breast Cancer dataset as `data.csv` in the project root.
3. Open and run the notebook:
   ```bash
   jupyter notebook Machine_Learning_Breastcancer.ipynb
   ```

---

## 📂 Project Structure

```
├── Machine_Learning_Breastcancer.ipynb   # Main notebook
├── data.csv                              # Wisconsin Breast Cancer dataset
└── README.md
```

---

## ⚠️ Challenges & Limitations

- **Qubit constraints** — Feature space was reduced to 4 dimensions to fit on a 4-qubit circuit, discarding potentially useful information.
- **Simulation overhead** — Quantum circuit simulation on classical hardware is slow, limiting the training set size for VQC.
- **Small ambiguous subset** — Only 9 samples were flagged as ambiguous, so VQC results should be interpreted with caution.
- **No real quantum hardware** — All experiments were run on the Aer simulator; results on actual quantum devices may differ due to noise.

---

## 🔮 What We Learned

- How to preprocess classical data for quantum input (scaling, dimensionality reduction).
- How to construct meaningful quantum circuits using ZZFeatureMap and RealAmplitudes.
- How to interpret results from quantum simulations and combine them with classical predictions.
- The promise — and current practical limits — of quantum machine learning for real-world tasks.

