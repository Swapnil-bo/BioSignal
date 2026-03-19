import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import joblib
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tempfile
from scipy.signal import welch
from src.features import extract_features

LABEL_MAP    = {1: 'Rest', 2: 'Left Fist', 3: 'Right Fist'}
LABEL_ICONS  = {'Rest': '😴', 'Left Fist': '🤜', 'Right Fist': '🤛'}
LABEL_COLORS = {'Rest': '#00bcd4', 'Left Fist': '#4caf50', 'Right Fist': '#f44336'}
BANDS = {
    'Delta (0–4 Hz)'  : (0,  4,  '#ef5350'),
    'Theta (4–8 Hz)'  : (4,  8,  '#ff9800'),
    'Alpha (8–13 Hz)' : (8,  13, '#00bcd4'),
    'Beta (13–30 Hz)' : (13, 30, '#4caf50'),
    'Gamma (30–45 Hz)': (30, 45, '#ab47bc'),
}

st.set_page_config(page_title='BioSignal Decoder', page_icon='🧠', layout='wide')

st.title('🧠 BioSignal Decoder')
st.markdown('Upload an EEG `.edf` file to classify mental states from brainwave signals.')
st.divider()

@st.cache_resource
def load_models():
    ensemble = joblib.load('models/ensemble.pkl')
    scaler   = joblib.load('models/scaler.pkl')
    return ensemble, scaler

ensemble, scaler = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Settings')
    n_channels_plot = st.slider('Channels to plot', 1, 64, 10)
    epoch_to_plot   = st.slider('Epoch for deep analysis', 1, 30, 1)
    show_raw_signal = st.checkbox('Show raw EEG signal', value=True)
    show_band_power = st.checkbox('Show band power spectrum', value=True)
    show_timeline   = st.checkbox('Show prediction timeline', value=True)

    st.divider()
    st.header('ℹ️ Model Info')
    st.markdown("""
    - **Model**: SVM + RF Ensemble
    - **Filter**: 8–30 Hz (motor)
    - **Wavelet**: Daubechies-4, L4
    - **Window**: 3.0 sec epochs
    - **Sampling**: 160 Hz
    - **Features**: 704 per epoch
    - **Classes**: Rest / L.Fist / R.Fist
    - **Dataset**: PhysioNet EEGBCI
    - **Subjects**: 109 available
""")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader('Upload EEG recording (.edf)', type=['edf'])

if not uploaded_file:
    col1, col2, col3 = st.columns(3)
    col1.info('📁 **Format**: PhysioNet .edf files')
    col2.info('🤖 **Model**: SVM + Random Forest ensemble')
    col3.info('🎯 **Output**: Rest · Left Fist · Right Fist')

    st.markdown("""
    ### How it works
    1. Upload any PhysioNet EEGBCI `.edf` file
    2. Signal is bandpass filtered at **8–30 Hz**
    3. Wavelet (db4) + FFT band power features extracted — **704 features/epoch**
    4. SVM + RF soft-voting ensemble predicts mental state
    5. Per-epoch predictions shown with confidence scores and visualisations
    """)

else:
    # ── Process ───────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner('🧠 Processing EEG signal...'):
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, verbose=False)
        sfreq      = raw.info['sfreq']
        n_channels = len(raw.ch_names)

        raw.filter(8., 30., fir_design='firwin', verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id,
                            tmin=0.0, tmax=3.0,
                            baseline=None, preload=True, verbose=False)
        data     = epochs.get_data()
        X        = extract_features(data)
        X_scaled = scaler.transform(X)
        preds    = ensemble.predict(X_scaled)
        probs    = ensemble.predict_proba(X_scaled)

    os.unlink(tmp_path)

    n_epochs    = len(preds)
    ep_idx      = min(epoch_to_plot - 1, n_epochs - 1)
    label_names = [LABEL_MAP[p] for p in preds]
    confidences = probs.max(axis=1) * 100
    avg_conf    = confidences.mean()
    counts      = {l: label_names.count(l) for l in ['Rest', 'Left Fist', 'Right Fist']}
    dominant    = max(counts, key=counts.get)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.success(f'✅ Processed **{n_epochs} epochs** from `{uploaded_file.name}`')

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Epochs', n_epochs, f'{sfreq:.0f} Hz · {n_channels} ch')
    k2.metric('Dominant State', f'{LABEL_ICONS[dominant]} {dominant}', f'{counts[dominant]}/{n_epochs} epochs')
    k3.metric('Avg Confidence', f'{avg_conf:.1f}%', 'Ensemble certainty')
    k4.metric('File size', f'{uploaded_file.size // 1024} KB', uploaded_file.name[:20])

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(['🔬 Epoch Analysis', '📈 Signal Plots', '🌊 Band Power', '📊 Statistics'])

    # ─────────────────────────── TAB 1 ───────────────────────────────────────
    with tab1:
        st.subheader(f'Per-Epoch Predictions — {n_epochs} epochs')
        c1, c2, c3 = st.columns(3)
        c1.metric('😴 Rest',       counts['Rest'],       f"{counts['Rest']/n_epochs*100:.0f}%")
        c2.metric('🤜 Left Fist',  counts['Left Fist'],  f"{counts['Left Fist']/n_epochs*100:.0f}%")
        c3.metric('🤛 Right Fist', counts['Right Fist'], f"{counts['Right Fist']/n_epochs*100:.0f}%")
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            label      = LABEL_MAP[pred]
            confidence = prob.max() * 100
            color      = LABEL_COLORS[label]
            col_a, col_b, col_c, col_d = st.columns([1, 2, 5, 1])
            col_a.markdown(f'**{i+1:02d}**')
            col_b.markdown(f'<span style="color:{color};font-weight:600">{LABEL_ICONS[label]} {label}</span>',
                           unsafe_allow_html=True)
            col_c.progress(int(confidence))
            col_d.markdown(f'`{confidence:.0f}%`')

    # ─────────────────────────── TAB 2 ───────────────────────────────────────
    with tab2:
        n_plot     = min(n_channels_plot, data.shape[1])
        epoch_data = data[ep_idx][:n_plot]

        if show_raw_signal:
            st.subheader(f'Raw EEG — Epoch {ep_idx+1} of {n_epochs} · {n_plot} channels · Predicted: {LABEL_ICONS[LABEL_MAP[preds[ep_idx]]]} {LABEL_MAP[preds[ep_idx]]} ({confidences[ep_idx]:.0f}% confidence)')
            fig, axes = plt.subplots(n_plot, 1,
                                     figsize=(12, n_plot * 0.65 + 0.5),
                                     facecolor='#0e1117', sharex=True)
            fig.subplots_adjust(hspace=0.05, left=0.07, right=0.98, top=0.97, bottom=0.07)
            if n_plot == 1:
                axes = [axes]
            times = np.linspace(0, 3, epoch_data.shape[1])
            cmap  = plt.cm.cool
            for i, (ax, ch) in enumerate(zip(axes, epoch_data)):
                ax.plot(times, ch * 1e6, color=cmap(i / max(n_plot-1, 1)),
                        linewidth=0.7, alpha=0.9)
                ax.set_facecolor('#262730')
                ax.set_yticks([])
                ax.tick_params(colors='#aaa', labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')
                ax.text(0.005, 0.5, f'Ch{i+1}', transform=ax.transAxes,
                        fontsize=6.5, color='#aaa', va='center')
            axes[-1].set_xlabel('Time (s)', color='#aaa', fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if show_timeline:
            st.subheader('Prediction Timeline')
            fig, ax = plt.subplots(figsize=(12, 1.8), facecolor='#0e1117')
            ax.set_facecolor('#262730')
            for i, (label, conf) in enumerate(zip(label_names, confidences)):
                color = LABEL_COLORS[label]
                ax.barh(0, 1, left=i, height=0.6,
                        color=color, alpha=0.8, edgecolor='#0e1117', linewidth=0.5)
                if n_epochs <= 40:
                    ax.text(i + 0.5, 0, f'{conf:.0f}%', ha='center', va='center',
                            fontsize=6, color='white', fontweight='bold')
            ax.set_xlim(0, n_epochs)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Epoch index', color='#aaa', fontsize=8)
            ax.set_yticks([])
            ax.tick_params(colors='#aaa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            patches = [mpatches.Patch(color=LABEL_COLORS[l], label=l)
                       for l in ['Rest', 'Left Fist', 'Right Fist']]
            ax.legend(handles=patches, loc='upper right', fontsize=7,
                      framealpha=0.3, labelcolor='white')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ─────────────────────────── TAB 3 ───────────────────────────────────────
    with tab3:
        if show_band_power:
            st.subheader(f'Power Spectral Density — Epoch {ep_idx+1}, first {n_plot} channels')
            fig = plt.figure(figsize=(12, 5), facecolor='#0e1117')
            gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            ax1.set_facecolor('#262730')
            cmap = plt.cm.cool
            for i, ch in enumerate(data[ep_idx][:n_plot]):
                freqs, psd = welch(ch, sfreq, nperseg=int(sfreq * 2))
                ax1.semilogy(freqs, psd, color=cmap(i / max(n_plot-1, 1)),
                             alpha=0.6, linewidth=0.9)

            for bname, (flo, fhi, bc) in BANDS.items():
                ax1.axvspan(flo, fhi, alpha=0.08, color=bc)
                ax1.text((flo+fhi)/2, 1e-16, bname.split()[0],
                         ha='center', va='bottom', fontsize=6.5,
                         color=bc, alpha=0.9)

            ax1.set_xlim(0, 50)
            ax1.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=8)
            ax1.set_ylabel('Power (µV²/Hz)', color='#aaa', fontsize=8)
            ax1.tick_params(colors='#aaa', labelsize=8)
            for spine in ax1.spines.values(): spine.set_edgecolor('#444')
            ax1.grid(True, color='#444', linewidth=0.4, alpha=0.5)

            # Band power bar chart
            ax2.set_facecolor('#262730')
            band_powers, band_labels, band_colors = [], [], []
            for bname, (flo, fhi, bc) in BANDS.items():
                bp_vals = []
                for ch in data[ep_idx]:
                    freqs, psd = welch(ch, sfreq, nperseg=int(sfreq * 2))
                    idx = np.logical_and(freqs >= flo, freqs <= fhi)
                    bp_vals.append(np.mean(psd[idx]))
                band_powers.append(np.mean(bp_vals))
                band_labels.append(bname.split()[0])
                band_colors.append(bc)

            ax2.bar(band_labels, band_powers, color=band_colors,
                    alpha=0.8, edgecolor='#0e1117', linewidth=0.5)
            ax2.set_ylabel('Mean Power', color='#aaa', fontsize=8)
            ax2.set_yscale('log')
            ax2.tick_params(colors='#aaa', labelsize=7)
            for spine in ax2.spines.values(): spine.set_edgecolor('#444')
            ax2.grid(True, color='#444', linewidth=0.4, alpha=0.5, axis='y')

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ─────────────────────────── TAB 4 ───────────────────────────────────────
    with tab4:
        s1, s2 = st.columns(2)

        with s1:
            st.subheader('Class Distribution')
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
            ax.set_facecolor('#262730')
            values = [counts[l] for l in ['Rest', 'Left Fist', 'Right Fist']]
            colors = [LABEL_COLORS[l] for l in ['Rest', 'Left Fist', 'Right Fist']]
            ax.pie(values, labels=['Rest', 'Left Fist', 'Right Fist'],
                   colors=colors, autopct='%1.0f%%', startangle=90,
                   wedgeprops=dict(edgecolor='#0e1117', linewidth=2),
                   textprops=dict(color='white', fontsize=9))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with s2:
            st.subheader('Confidence Histogram')
            col_min, col_max = st.columns(2)
            col_min.metric('Min Confidence', f'{confidences.min():.1f}%')
            col_max.metric('Max Confidence', f'{confidences.max():.1f}%')
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
            ax.set_facecolor('#262730')
            ax.hist(confidences, bins=15, color='#00bcd4',
                    edgecolor='#0e1117', linewidth=0.5, alpha=0.85)
            ax.axvline(avg_conf, color='#4caf50', linewidth=1.5,
                       linestyle='--', label=f'Mean: {avg_conf:.1f}%')
            ax.set_xlabel('Confidence (%)', color='#aaa', fontsize=8)
            ax.set_ylabel('Count', color='#aaa', fontsize=8)
            ax.tick_params(colors='#aaa', labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor('#444')
            ax.grid(True, color='#444', linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=8, framealpha=0.3, labelcolor='white')
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.subheader('Confidence Over Time')
        fig, ax = plt.subplots(figsize=(12, 2.5), facecolor='#0e1117')
        ax.set_facecolor('#262730')
        x = np.arange(1, n_epochs + 1)
        ax.fill_between(x, confidences, alpha=0.2, color='#00bcd4')
        ax.plot(x, confidences, color='#00bcd4', linewidth=1.5,
                marker='o', markersize=4,
                markerfacecolor='#4caf50', markeredgecolor='#0e1117',
                markeredgewidth=0.5)
        ax.axhline(avg_conf, color='#ab47bc', linewidth=0.8,
                   linestyle='--', alpha=0.8, label=f'Mean {avg_conf:.1f}%')
        ax.set_xlim(1, n_epochs)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Epoch', color='#aaa', fontsize=8)
        ax.set_ylabel('Confidence (%)', color='#aaa', fontsize=8)
        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#444')
        ax.grid(True, color='#444', linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.3, labelcolor='white')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("BioSignal Decoder · PhysioNet EEGBCI · SVM+RF Ensemble · MNE Python · Built by Swapnil")
        plt.close(fig)

