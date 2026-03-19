import mne
import os

# Download subject S001, runs 4,8,12 (motor imagery: left/right fist + feet)
subject = 1
runs = [4, 8, 12]

raw_fnames = mne.datasets.eegbci.load_data(subject, runs, path='data/raw/')

print("Downloaded files:")
for f in raw_fnames:
    print(f"  {os.path.basename(f)}")

# Quick sanity check — load the first file
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
print(f"\nChannels : {len(raw.ch_names)}")
print(f"Duration : {raw.times[-1]:.1f} seconds")
print(f"Sampling : {raw.info['sfreq']} Hz")
