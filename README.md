# 🧠 BioSignal Decoder

> EEG mental state classifier — raw brainwave data → signal processing → ML ensemble → interactive dashboard. Zero LLMs. 100% classical ML + signal processing.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![MNE](https://img.shields.io/badge/MNE-1.8.0-purple?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What it does

BioSignal Decoder takes a raw EEG `.edf` recording and classifies each 3-second epoch into one of three mental states:

| State | Label | Description |
|-------|-------|-------------|
| 😴 | **Rest** | Baseline / no motor imagery |
| 🤜 | **Left Fist** | Imagining left-hand movement |
| 🤛 | **Right Fist** | Imagining right-hand movement |

No large language models involved. The entire pipeline is signal processing + classical ML:

```
Raw .edf → Bandpass Filter (8–30 Hz) → Wavelet (db4) + FFT → 704 features → SVM + RF Ensemble → Prediction
```

---

## 🖥️ Demo

| Tab | What you see |
|-----|-------------|
| 🔬 Epoch Analysis | Per-epoch predictions with class probability bars |
| 📈 Signal Plots | Raw EEG waveform per channel + prediction timeline |
| 🌊 Band Power | PSD spectrum with frequency bands + band power bar chart |
| 📊 Statistics | Class distribution pie + confidence histogram + confidence over time |

---

## 🗂️ Project Structure

```
BioSignalDecoder/
├── data/
│   ├── raw/                  ← PhysioNet .edf files (auto-downloaded)
│   └── processed/            ← epochs.npy, labels.npy, X.npy, y.npy
├── src/
│   ├── __init__.py
│   ├── download_data.py      ← fetch PhysioNet EEGBCI dataset via MNE
│   ├── preprocess.py         ← bandpass filter, epoch extraction
│   ├── features.py           ← wavelet + FFT band power feature extraction
│   ├── train.py              ← SVM + RF ensemble training + evaluation
│   └── predict.py            ← inference on new .edf files
├── models/
│   ├── ensemble.pkl          ← trained SVM + RF VotingClassifier
│   └── scaler.pkl            ← fitted StandardScaler
├── app.py                    ← Streamlit dashboard
└── requirements.txt
```

---

## ⚙️ Pipeline

### 1. Data — PhysioNet EEG Motor Imagery
- Dataset: [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, public domain)
- Runs used: R04, R08, R12 (motor imagery: left fist / right fist / rest)
- 64 EEG channels · 160 Hz sampling rate · ~125 seconds per run

### 2. Preprocessing
- Concatenate 3 runs per subject
- Standardize channel names to 10-05 montage
- Bandpass filter: **8–30 Hz** (motor imagery range: alpha + beta)
- Artifact removal via MNE pipeline
- Epoch extraction: **0 to 3 seconds** per event

### 3. Feature Extraction — 704 features per epoch
| Feature type | Method | Output shape |
|---|---|---|
| Band power | Welch PSD (8–30 Hz) | 64 values |
| Wavelet mean abs | db4 level 4, mean\|coeff\| per channel | 5 × 64 × 2 = 640 values |

Total: **704 features** per 3-second epoch.

### 4. Model — SVM + Random Forest Ensemble
- `SVC(kernel='rbf', C=10, gamma='scale', probability=True)`
- `RandomForestClassifier(n_estimators=200)`
- Combined via `VotingClassifier(voting='soft')` — averages class probabilities
- Features standardised with `StandardScaler` before SVM input
- 5-fold cross-validation on full dataset

### 5. Results (1 subject, 90 epochs)

```
CV Accuracy : 0.567 ± 0.042

              precision    recall  f1-score   support
        Rest       0.60      1.00      0.75         9
   Left Fist       0.33      0.20      0.25         5
  Right Fist       0.00      0.00      0.00         4
```

> ⚠️ Accuracy is expected to be low with 1 subject / 90 epochs. Training on 5–10 subjects raises accuracy significantly. This is a known challenge in cross-epoch EEG classification.

---

## 🚀 Setup & Run

### 1. Clone & create venv

```bash
git clone https://github.com/YOUR_USERNAME/BioSignalDecoder.git
cd BioSignalDecoder
python -m venv venv
venv\Scripts\activate        # Windows cmd
# source venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python src/download_data.py
```

Downloads ~7.4 MB of PhysioNet EEG data for subject S001 into `data/raw/`.

### 4. Run full pipeline

```bash
python src/preprocess.py     # filter + epoch
python src/features.py       # extract 704 features
python src/train.py          # train + save models
```

### 5. Launch dashboard

```bash
streamlit run app.py
```

Open `localhost:8501`, upload any `.edf` file from `data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/`.

---

## 📦 Requirements

```
mne==1.8.0
numpy==2.1.3
scipy==1.14.1
PyWavelets==1.7.0
scikit-learn==1.5.2
pandas==2.2.3
matplotlib==3.9.4
seaborn==0.13.2
streamlit==1.41.1
joblib==1.4.2
```

Tested on Python 3.13 · Windows 11 · RTX 3050 6GB

---

## 🔬 What makes this stand out

Brain-computer interfaces are a **frontier product category**. This project demonstrates:

- **Zero LLMs** — pure signal processing and classical ML
- **Real neuroscience dataset** — PhysioNet, used in actual BCI research
- **Full end-to-end pipeline** — from raw `.edf` to live interactive predictions
- **Interpretable outputs** — per-epoch confidence scores, band power charts, PSD visualisation
- **Production-style code** — modular `src/` layout, saved `.pkl` models, Streamlit frontend

---

## 🗺️ Roadmap

- [ ] Multi-subject training (5–10 subjects) for better generalisation
- [ ] Cross-subject generalisation test
- [ ] SEED dataset support (emotion: positive / neutral / negative)
- [ ] Real-time inference via OpenBCI EEG headset
- [ ] Docker containerisation for one-command deploy

---

## 🙋 Author

**Swapnil Hazra**
- X / Twitter: [@SwapnilHazra4](https://twitter.com/SwapnilHazra4)
- Instagram: [@swapnil_hazra_](https://instagram.com/swapnil_hazra_)

*Part of the [100 Days of Vibe Coding](https://github.com/Swapnil-bo) challenge.*