import mne
import numpy as np
import os

def load_and_preprocess(subject=1, runs=[4, 8, 12]):
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs, path='data/raw/')

    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)

    # Standardize channel names
    mne.datasets.eegbci.standardize(raw)

    # Set standard 10-05 montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # Bandpass filter 8–30 Hz (motor imagery range)
    raw.filter(8., 30., fir_design='firwin', verbose=False)

    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Epoch: 0 to 3 seconds after each event
    epochs = mne.Epochs(raw, events, event_id,
                        tmin=0.0, tmax=3.0,
                        baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]
    data   = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    print(f"Epochs shape : {data.shape}")
    print(f"Labels       : {np.unique(labels)} → counts {np.bincount(labels)[np.unique(labels)]}")

    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/epochs.npy', data)
    np.save('data/processed/labels.npy', labels)
    print("Saved to data/processed/")

if __name__ == '__main__':
    load_and_preprocess()
