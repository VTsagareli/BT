# Fuel-Pump Sound Anomaly Detection (Audio + ML / 1D-CNN) - Bachelor Thesis Project

> Detecting **broken vs. normal** vehicle behavior from short interior audio clips using classical ML on MFCC features and a lightweight **1D-CNN**. Built for early, non-invasive fault awareness and as a foundation for predictive maintenance.

---

## 📌 TL;DR

- **Goal:** Binary classify 3-second interior car audio as **Normal** or **Broken (fuel-pump buzz)**  
- **Data:**  
  - **Normal:** ~1,500 clips sampled from a 5k+ open-access interior-sound dataset (Kaggle)  
  - **Broken:** ~400 clips curated via multilingual web search (EN/DE/RU/KA), **augmented** to ~1,500  
- **Stack:** Python, Librosa, scikit-learn, PyTorch, NumPy, SoundFile, Pandas, Matplotlib, Seaborn
- **Note:** Both ML and CNN showed signs of **overfitting** (see Limitations).

---

## 🧭 Table of Contents

- [Why this matters](#why-this-matters)  
- [Project scope](#project-scope)  
- [Data sources](#data-sources)  
- [Methodology](#methodology)  
- [Models](#models)  
- [Design decisions & trade-offs](#design-decisions--trade-offs)  
- [Limitations](#limitations)  
- [Future work](#future-work)  
- [FAQ](#faq)  
- [Cite & license](#cite--license)  

---

## Why this matters

Interior audio provides a **non-invasive** diagnostic signal. Detecting a characteristic **fuel-pump buzzing** pattern from short clips can help:

- Give **drivers** early warnings before costly failures  
- Help **shops/fleets** triage faster  
- Inform **product** ideas for embedded/mobile diagnostics

---

## Project scope

- **Task:** Binary classification — `normal` vs `broken`  
- **Unit of analysis:** 3-second interior audio segments  
- **Focus:** Fuel-pump related buzzing in the **broken** class  
- **Out of scope:** Root-cause identification, real-time streaming, multi-fault taxonomy

---

## Data sources

- **Normal audio:**  
  Kaggle — *Open-Access Vehicle Interior Sound Dataset*  
  https://www.kaggle.com/datasets/omkarmb/dataset-open-access-vehicle-interior-sound-dataset

- **Broken audio (fuel-pump buzz):**  
  Curated from web audio/video via multilingual search (**EN/DE/RU/KA**), standardized to **3-second** segments.

> 🔧 **Augmentations (mainly for Broken):** noise injection, pitch shift, time-stretch, mild reverb/room impulse, EQ, and light frequency masking — to increase diversity and improve generalization.

- **After processing:**  
  - Normal: **~1,500** clips (down-sampled from 5k+)  
  - Broken: **~1,500** clips (≈400 originals + augmentation)

---

## Methodology

1) **Standardize & segment**  
Convert all audio to a common format (e.g., WAV, 22,050 Hz, mono). Slice to **3 s** windows.

2) **Feature prep**  
Compute **MFCCs** (e.g., 40 coefficients) with Librosa; normalize features for ML/CNN pipelines.

3) **Split strategy**  
Stratified train/test split and a separate **unseen holdout** from different sources to check robustness.

4) **Train**  
- ML baselines with scikit-learn  
- A simple **1D-CNN** with PyTorch on MFCC sequences

5) **Evaluate**  
Accuracy, Precision, Recall, F1, Confusion Matrix. Compare **Test** vs **Unseen** performance.

---

## Models

### Classical ML (MFCC features)
- Logistic Regression, SVM, Decision Tree, Random Forest  
- Grid/heuristic tuning on key hyperparameters

### 1D-CNN (PyTorch) on MFCC sequences — high level
- Operates directly on **MFCC time series**  
- Small stack of **convolutions + pooling + dropout**, then a **fully-connected** output layer  
- Trained with **Adam**, cross-entropy loss, and basic regularization (dropout, weight decay, early stopping)

> Intention: keep the network lightweight and readable while capturing temporal patterns in the buzzing.

---

## Design decisions & trade-offs

- **3-second windows** to align with the Kaggle dataset and keep batches consistent  
- **MFCCs** for compact, robust features on small datasets (used by both ML and CNN)  
- **Lightweight 1D-CNN** to capture temporal patterns without heavy compute  
- **Targeted augmentation** for the scarce **broken** class to reduce imbalance

---

## Limitations

- **Validation setup:** Ensure a true **train/validation/test** split for the CNN (avoid using train data as validation)  
- **Overfitting risk:** Broken samples rely heavily on augmentation; more real-world diversity is needed  
- **Binary focus:** Flags “broken vs normal,” not specific component types  
- **Recording variance:** Performance can vary with devices, mic placement, and cabin noise

---

## Future work

- **Data:** Collect more genuine broken samples across car models, devices, environments  
- **Modeling:** Proper DataLoaders; early stopping on a real validation set; try CRNN/Transformers or transfer learning; threshold tuning  
- **Evaluation:** Robustness at different SNRs and devices; clearer cost-of-errors framing (FP vs FN)  
- **Productization:** On-device/mobile inference; real-time streaming with sliding windows; simple explainability (saliency over time)

---

## FAQ

**What is the “broken” class?**  
Fuel-pump buzzing patterns indicative of failure.

**Does the model localize the fault in time?**  
Currently **clip-level** classification. Segment scoring is a possible extension.

**Can I run this on my own recordings?**  
Yes — place 3-second WAV clips in `data/unseen/` and run the unseen evaluation script.

---

## Cite & license

- **Normal audio dataset:**  
  Kaggle — *Open-Access Vehicle Interior Sound Dataset*  
  https://www.kaggle.com/datasets/omkarmb/dataset-open-access-vehicle-interior-sound-dataset

---
