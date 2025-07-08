import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


def denoise_ecg(signal: np.ndarray,
                fs: float = 256,
                lp_cut: float = 45.0,
                mains_freq: float = 50.0,
                q: float = 30.0) -> np.ndarray:

    sos_lp = butter(
        N=4,
        Wn=lp_cut,
        btype='low',
        fs=fs,
        output='sos'
    )
    lowpassed = sosfiltfilt(sos_lp, signal)

    b_notch, a_notch = iirnotch(w0=mains_freq, Q=q, fs=fs)
    clean = filtfilt(b_notch, a_notch, lowpassed)

    clean = (clean-clean.mean())/clean.std()

    return clean



def find_label_edges(lbl_bool: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Return the sample indices where the label flips
      0 → 1  (seizure onset)
      1 → 0  (seizure offset)
    """
    padded = np.concatenate([[False], lbl_bool, [False]])
    edges   = np.flatnonzero(np.diff(padded))          # every change
    onset_i = edges[0::2]                              # even positions → 0→1
    offset_i = edges[1::2]                             # odd  positions → 1→0
    return onset_i.tolist(), offset_i.tolist()

def plot_context(ax, t, sig, lbl, idx, pre_s=5, post_s=5):
    """
    Plot signal ± (pre_s, post_s) seconds around `idx`.
    """
    fs       = 1.0 / np.mean(np.diff(t))               # sampling‐rate ≈ Hz
    win      = int((pre_s + post_s) * fs)
    start    = max(idx - int(pre_s*fs), 0)
    stop     = min(idx + int(post_s*fs), len(sig)-1)
    sel      = slice(start, stop)

    ax.plot(t[sel], sig[sel])
    ax.fill_between(t[sel], sig[sel].min(), sig[sel].max(),
                    where=lbl[sel], alpha=0.15)        # shaded seizure
    ax.axvline(t[idx], ls='--')                        # transition marker
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [mV]')

def plot_all_transitions(raw_df, pre_s=5, post_s=5, max_plots=4):
    """
    Convenience wrapper — shows the first few onsets *and* offsets.
    """
    t   = raw_df["Time [s]"].values
    sig = raw_df["Signal [mV]"].values
    lbl = raw_df["Seizure [bool]"].values.astype(bool)

    onsets, offsets = find_label_edges(lbl)

    n = min(len(onsets)+len(offsets), max_plots)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.2*n), sharex=False)

    pairs = [('onset', i)  for i in onsets] + \
            [('offset', i) for i in offsets]
    # show earliest first, truncate if > max_plots
    for ax, (kind, idx) in zip(axes, sorted(pairs, key=lambda p: p[1])[:n]):
        plot_context(ax, t, sig, lbl, idx, pre_s, post_s)
        ax.set_title(f'Seizure {kind} at t = {t[idx]:.2f}s (idx={idx})')

    plt.tight_layout()
    plt.show()
PATH=""
raw_df = pd.read_csv(PATH).iloc[1:].reset_index(drop=True)
plot_all_transitions(raw_df, pre_s=5, post_s=5, max_plots=6)



