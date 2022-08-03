import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import matplotlib.pyplot as plot

def load_wav(wav_file, sr):
    """
    Load a wav file with librosa.
    :param wav_file: path to wav file
    :param sr: sampling rate
    :return: audio time series numpy array
    """
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)

    return wav


def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y

    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))


SAMPLING_RATE = 16000      # 22050
FRAME_PERIOD = 5.0


# Load the wavs
def wav2mcep_numpy(wavfile, target_directory, alpha=0.65, fft_size=512, mcep_size=34):
    # make relevant directories
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    wavfile_tmp = wavfile.split('RFU')[0]

    if len(wavfile.split('.')) > 3:
        source = wavfile_tmp.split('.')[1][-8:]
        target = wavfile_tmp.split('.')[-2][-8:]
        fname = source + '_' + target
    else:
        fname = os.path.basename(wavfile).split('.')[0]

    loaded_wav = load_wav(wavfile, sr=SAMPLING_RATE)

    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=SAMPLING_RATE,
                                 frame_period=FRAME_PERIOD, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    # fname = os.path.basename(wavfile).split('.')[0]

    print(mgc.shape)        # (411, 35)

    np.save(os.path.join(target_directory, fname + '.npy'),
            mgc,
            allow_pickle=False)


alpha = 0.65  # commonly used at 22050 Hz
fft_size = 512
mcep_size = 34

# p262_001.npy	p362_001.npy	189

# vc_trg_wavs = glob.glob('/ceph/home/yangsc21/Python/autovc/autovc/mcd/mcd/target/*')
root = '/ceph/datasets/VCTK-Corpus/wav16/'
path = "/ceph/home/yangsc21/Python/VCTK/wav16/select_from_train.txt"
ls = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if int(line.strip().split('\t')[-1]) <= 128 * 3:
            ls.append(line.strip().split('\t')[0])
            ls.append(line.strip().split('\t')[1])
print(len(set(ls)))     # 523, 978
vc_trg_wavs = [os.path.join(root, item[:4], item[:-4] + '.wav') for item in set(ls)]
# vc_trg_wavs = ['/ceph/datasets/VCTK-Corpus/wav16/p255/p255_001.wav']        # (667, 35) for wav48, (667, 35) for wav16
vc_trg_mcep_dir = "/ceph/home/yangsc21/Python/autovc/autovc/mcd/ClsVC/target_mel/"
vc_conv_wavs = glob.glob('/ceph/home/yangsc21/Python/autovc/autovc/mcd/ClsVC/results_200000/*')
vc_conv_mcep_dir = "/ceph/home/yangsc21/Python/autovc/autovc/mcd/ClsVC/syn_mel/"



vc_conv_wavs = os.listdir('/ceph/home/yangsc21/Python/autovc/Final/VQMIVC__/')
vc_trg_mcep_dir = "/ceph/home/yangsc21/Python/autovc/Final/mcd/vc_trg_mcep/"
vc_conv_mcep_dir = "/ceph/home/yangsc21/Python/autovc/Final/mcd/vc_conv_mcep/"

for wav in vc_conv_wavs:
    try:
        tmp = wav.split('_')
        target = tmp[1]
        target_ = tmp[2][-3:]
        wav = os.path.join('/ceph/datasets/VCTK-Corpus/wav16/', target, target + '_' + target_ + '.wav')
        wav2mcep_numpy(wav, vc_trg_mcep_dir, fft_size=fft_size, mcep_size=mcep_size)
    except:
        continue

for wav in vc_conv_wavs:
    try:
        wav2mcep_numpy(os.path.join('/ceph/home/yangsc21/Python/autovc/Final/VQMIVC__/', wav), vc_conv_mcep_dir, fft_size=fft_size, mcep_size=mcep_size)
    except:
        continue

def average_mcd(ref_mcep_files, synth_mcep_files, cost_function):
    """
    Calculate the average MCD.
    :param ref_mcep_files: list of strings, paths to MCEP target reference files
    :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
    :param cost_function: distance metric used
    :returns: average MCD, total frames processed
    """
    min_cost_tot = 0.0
    frames_tot = 0
    cnt = 0
    tmp = []
    for ref in ref_mcep_files:
        for synth in synth_mcep_files:
            # get the trg_ref and conv_synth speaker name and sample id
            ref_fsplit, synth_fsplit = os.path.basename(ref).split('_'), os.path.basename(synth).split('_')

            ref_spk, ref_id = ref_fsplit[0], ref_fsplit[-1][:3]
            synth_spk, synth_id = synth_fsplit[2], synth_fsplit[3][:3]

            # if the speaker name is the same and sample id is the same, do MCD
            if ref_spk == synth_spk and ref_id == synth_id:
                cnt += 1
                # load MCEP vectors
                ref_vec = np.load(ref)
                ref_frame_no = len(ref_vec)
                synth_vec = np.load(synth)


                # dynamic time warping using librosa
                # min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T,
                #                                    metric=cost_function)

                min_cost, _ = librosa.sequence.dtw(ref_vec[:, :].T, synth_vec[:, :].T,
                                                   metric=cost_function)

                min_cost_tot += np.mean(min_cost)
                frames_tot += ref_frame_no
                tmp.append(np.mean(min_cost)/ref_frame_no)
                print(cnt)

    mean_mcd = min_cost_tot / frames_tot
    print(max(tmp), min(tmp))
    return mean_mcd, frames_tot, cnt


vc_trg_refs = glob.glob("/ceph/home/yangsc21/Python/autovc/Final/mcd/vc_trg_mcep/*")
vc_conv_synths = glob.glob("/ceph/home/yangsc21/Python/autovc/Final/mcd/vc_conv_mcep/*")

cost_function = log_spec_dB_dist

vc_mcd, vc_tot_frames_used, cnt = average_mcd(vc_trg_refs, vc_conv_synths, cost_function)

# 4.887870852471173 dB, 40.208741547421795 dB for 80 bin.
print(f'MCD = {vc_mcd} dB, calculated over a total of {vc_tot_frames_used} frames, total {cnt} pairs')
# MCD = 6.729270791070735 dB, calculated over a total of 658567 frames, total 896 pairs