import os
import soundfile as sf
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from pysptk import sptk

target_f0_path = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/test_raptf0/p232/p232_001.npy"
source_f0_path = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/test_raptf0/p225/p225_001.npy"
generate_wav_path = "/ceph/home/yangsc21/Python/autovc/Final/My_model_/"

fig = plt.figure()
axes = fig.subplots(nrows=2, ncols=1)
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


b, a = butter_highpass(30, 16000, order=5)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0

def wav2f0_(item):
    if 'p232' in item:  # man
        lo, hi = 50, 250
    else:
        lo, hi = 100, 600

    x, fs = sf.read(item)
    assert fs == 16000
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    f0_rapt = sptk.rapt(y.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    f0_norm[f0_norm == -1e10] = 0
    f0_rapt[f0_rapt == -1e10] = 0
    return f0_rapt

# source_f0 = np.load(source_f0_path)
# source_f0[source_f0 == -1e10] = 0

source_f0 = wav2f0_("/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/test_wav/p225/p225_001.wav")

# target_f0 = np.load(target_f0_path)
# target_f0[target_f0 == -1e10] = 0

target_f0 = wav2f0_("/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/test_wav/p232/p232_001.wav")

# plt.subplot(211)
axes[0].plot(source_f0, color='#1f77b4', label='Source', linewidth=2)
axes[0].set_xlim(0, len(target_f0))
# plt.subplot(212)
axes[1].plot(target_f0, color='#ff7f0e', label='Target', linewidth=2)
axes[1].set_xlim(0, len(target_f0))


def wav2f0(item):
    if 'U' in item:  # man
        lo, hi = 50, 250
    else:
        lo, hi = 100, 600

    x, fs = sf.read(os.path.join(generate_wav_path, item))
    assert fs == 16000
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    f0_rapt = sptk.rapt(y.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    f0_norm[f0_norm == -1e10] = 0
    f0_rapt[f0_rapt == -1e10] = 0
    return f0_rapt


for item in os.listdir(generate_wav_path)[::-1]:
    print(item)
    f0_norm = wav2f0(item)
    # if "R" in item:
    #     plt.subplot(212)
    # else:
    #     plt.subplot(211)
    if "_R.wav" in item:
        # plt.subplot(212)
        axes[1].plot(f0_norm, color='#9467bd', label='Rhythm Conversion', linewidth=2)
    elif "_U.wav" in item:
        # plt.subplot(211)
        axes[0].plot(f0_norm[:105], color='#d62728',label='Timbre Conversion', linewidth=2)
    elif "_F.wav" in item:
        # plt.subplot(211)
        axes[0].plot(f0_norm, color='#2ca02c',label='Pitch Conversion', linewidth=2)

    # if "_FU.wav" in item:
    #     plt.subplot(211)
    #     plt.plot(f0_norm, label=item.split('_')[-1])
    # elif "_U.wav" in item:
    #     plt.subplot(211)
    #     plt.plot(f0_norm, label=item.split('_')[-1])
    # elif "_RU.wav" in item:
    #     plt.subplot(212)
    #     plt.plot(f0_norm, label=item.split('_')[-1])
    # elif "_RFU.wav" in item:
    #     plt.subplot(212)
    #     plt.plot(f0_norm, label=item.split('_')[-1])

lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

fig.legend(lines, labels,
           loc = 3,bbox_to_anchor=(1.1, 1.1))

# plt.subplot(211)
# axes[0].legend(loc=3, fontsize=12)

axes[0].set_ylabel('Pitch Contours (Logf0)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.subplot(212)
axes[1].legend(lines, labels, loc=3, fontsize=12)
plt.xlabel('Times (Frames)',fontsize=12)
axes[1].set_ylabel('Pitch Contours (Logf0)',fontsize=12)

# plt.ylabel('Pitch Contours (Logf0)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('6.pdf')
