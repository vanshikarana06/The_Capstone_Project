# 🛡️ Dual-Stream Image Tampering Detection
### *Advanced Image Forensics using RGB and Error Level Analysis (ELA)*

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-89.2%25-green.svg)

## 📌 Project Overview
In the era of digital manipulation, identifying authentic vs. tampered images is critical. This project implements a **Dual-Input Convolutional Neural Network (CNN)** that analyzes images through two distinct lenses:
1. **Spatial Stream (Raw RGB):** Detects visual inconsistencies, edge anomalies, and lighting mismatches.
2. **Forensic Stream (ELA):** Analyzes JPEG compression artifacts to find regions with different error levels, signaling a localized edit.

---

## 📊 Performance & Results
After iterating through multiple architectures, the **Dual-Input Model** emerged as the most stable and reliable.

| Model Version | Accuracy | Characteristics |
| :--- | :--- | :--- |
| **ELA-Only CNN** | 79.1% | Highly jittery, struggled with generalization. |
| **Raw-RGB CNN** | 88.5% | High accuracy but showed signs of overfitting. |
| **Dual-Stream (Final)** | **89.2%** | **Stable convergence and best real-world reliability.** |

### **Training Statistics**
* **Validation Accuracy:** ~89%
* **Loss Stability:** Significantly improved via BatchNormalization.
* **Generalization:** Minimal gap between training and validation curves.

---

## 🚀 Key Features
* **Dual-Stream Architecture:** Merges features from two separate CNN branches.
* **Stability Enhancements:** Utilized `BatchNormalization` and `GlobalAveragePooling2D` to ensure smooth convergence.
* **Optimized Data Pipeline:** Built using `tf.data.Dataset` with `.cache()` and `.prefetch()` for high-speed training.
* **Forensic Preprocessing:** Custom ELA (Error Level Analysis) pipeline using PIL.

---

## 🧠 Engineering Decisions (The "Why")
* **Why Dual-Input?** RGB data excels at finding "what" changed, while ELA data excels at finding "where" the compression was broken.
* **Why GlobalAveragePooling?** Instead of `Flatten()`, GAP reduces the total parameters, which is crucial for preventing overfitting.
* **Why 128x128?** This resolution provides the best balance between preserving ELA noise details and maintaining fast training speeds.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib, Seaborn, Scikit-Learn, PIL (Pillow)
* **Environment:** Google Colab (GPU Accelerated)

---

## 📂 Project Structure
```text
├── models/
│   └── best_dual_input_model.h5    # Final trained weights
├── notebooks/
│   └── Image_Forensics_Main.ipynb  # Full training & evaluation pipeline
├── README.md                       # Project documentation
└── sample_tests/                   # Images for local inferenceImage Tampering Detection 
