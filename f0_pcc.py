import os
import soundfile as sf
from scipy import signal
import numpy as np
from pysptk import sptk
from sklearn import metrics


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


def wav2f0(item, lo, hi):
    x, fs = sf.read(item)
    assert fs == 16000
    # if x.shape[0] % 256 == 0:
    #     x = np.concatenate((x, np.array([1e-06])), axis=0)
    #     x = x[:-1]
    y = signal.filtfilt(b, a, x)
    f0_rapt = sptk.rapt(y.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    f0_norm[f0_norm == -1e10] = 0
    return f0_norm


root_wav_path = '/ceph/datasets/VCTK-Corpus/wav16/'
target_wav_path = "/ceph/home/yangsc21/Python/autovc/Final/My_model___/dim2_pitch_wo_adv_mi/"

f0_source = []
f0_source_ = []
f0_target = []

with open("/ceph/home/yangsc21/Python/autovc/Final/New_WER/clsvc_filter/filter_My_model.txt", 'r',
          encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')[2:5]
        if tmp[0] in ['p247', 'p278', 'p272']:
            lo, hi = 50, 250
        elif tmp[0] in ['p335', 'p264', 'p262']:
            lo, hi = 100, 600
        try:
            f0_source.append(wav2f0(os.path.join(root_wav_path, tmp[0], tmp[0] + '_' + tmp[2][:3] + '.wav'), lo, hi))
        except:
            print(os.path.join(root_wav_path, tmp[1], tmp[1] + '_' + tmp[2][:3] + '.wav'))
        f0_target.append(
            wav2f0(os.path.join(target_wav_path, tmp[0] + '_' + tmp[1] + '_' + tmp[2] + '_U.wav'), lo, hi))


pcc = 0.0
RMSE = 0.0
cnt = 0
for i in range(len(f0_source)):
    print(len(f0_source[i]), len(f0_target[i]))
    if len(f0_source[i]) == len(f0_target[i]):
        if np.isnan(np.corrcoef(f0_source[i], f0_target[i])[0][1]):
            print(i)
            continue
        pcc += (np.corrcoef(f0_source[i], f0_target[i]))[0][1]
        RMSE += metrics.mean_squared_error(f0_source[i], f0_target[i]) ** 0.5
        cnt += 1
    else:
        continue

print(cnt)
print(pcc / cnt)
print(RMSE / cnt)
