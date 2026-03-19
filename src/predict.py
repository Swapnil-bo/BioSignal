import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import joblib
import mne
from src.preprocess import load_and_preprocess
from src.features import extract_features


# Label map matching PhysioNet annotation codes
LABEL_MAP = {1: 'Rest', 2: 'Left Fist', 3: 'Right Fist'}

def predict_from_edf(edf_path):
    ensemble = joblib.load('models/ensemble.pkl')
    scaler   = joblib.load('models/scaler.pkl')

    # Load and preprocess the edf
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, verbose=False)
    raw.filter(8., 30., fir_design='firwin', verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id,
                        tmin=0.0, tmax=3.0,
                        baseline=None, preload=True, verbose=False)

    data = epochs.get_data()
    X = extract_features(data)
    X_scaled = scaler.transform(X)

    predictions  = ensemble.predict(X_scaled)
    probabilities = ensemble.predict_proba(X_scaled)

    print(f"\nFile: {edf_path}")
    print(f"Epochs processed: {len(predictions)}")
    print(f"\nPredictions per epoch:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        confidence = prob.max() * 100
        print(f"  Epoch {i+1:02d}: {LABEL_MAP[pred]:12s} ({confidence:.1f}% confidence)")

if __name__ == '__main__':
    # Test on a known file
    fnames = mne.datasets.eegbci.load_data(1, [4], path='data/raw/', verbose=False)
    predict_from_edf(fnames[0])
