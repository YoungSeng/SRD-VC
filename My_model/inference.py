import numpy as np
import random
import os
import torch
from hparams import hparams as hparams_
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_6 as F0_Converter
from model import Generator_MI, Generator_Decoder
import pickle

random.seed(137)
np.random.seed(137)
n_test = 100
MAX_LEN = 128 * 3  # 192

device = 'cuda:0'

use_VQCPC = False
use_VQCPC_2 = False
use_pitch = True

unseen_speaker = ['p335', 'p264', 'p247', 'p278', 'p272', 'p262']
unseen_speaker_content = {'p335': [], 'p264': [], 'p247': [], 'p278': [], 'p272': [], 'p262': []}

log_path = "/ceph/home/yangsc21/Python/VCTK/wav16/mel_dim_6_crop.txt"

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line == 'test:\n': continue
        [_, mel_name, _] = line.strip().split('\t')
        unseen_speaker_content[mel_name[:4]].append(mel_name)

root = "/ceph/home/yangsc21/Python/VCTK/wav16/"

f0_path = root + 'raptf0'
mel_path = root + 'spmel_6'

split_speaker = "/ceph/home/yangsc21/Python/VCTK/wav16/split_speaker.txt"
with open(split_speaker, 'r', encoding='utf-8') as f:
    c = f.read().strip().split('\n')[1].strip().split(' ')


def speaker2index(string):
    id = c.index(string)
    return id


G1 = Generator_MI(hparams_, use_VQCPC, use_VQCPC_2).eval().to(device)
G2 = Generator_Decoder(hparams_, use_pitch).eval().to(device)

G1 = torch.nn.DataParallel(G1, device_ids=[0, 1], output_device=0)  # 主要就是这句
G2 = torch.nn.DataParallel(G2, device_ids=[0, 1], output_device=0)  # 主要就是这句

G_path = "/ceph/home/yangsc21/Python/autovc/My_model/run_pitch_2/models/800000-G.ckpt"
g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
G1.load_state_dict(g_checkpoint['G1'])
G2.load_state_dict(g_checkpoint['G2'])

P = F0_Converter(hparams_).eval().to(device)
P_path = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/640000-P.ckpt"
p_checkpoint = torch.load(P_path, map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])


def Generator_F(f0, mel_1, mel_2):
    if use_VQCPC or use_VQCPC_2:
        content, pitch, rhythm, _, _, _, _ = G1(f0, mel_1)  # emb_trg
    else:
        content, pitch, rhythm = G1(f0, mel_1)
    if use_pitch:
        _, mel_, _, _, _ = G2(content, pitch, rhythm, mel_2, MAX_LEN)
    else:
        _, mel_, _, _ = G2(content, pitch, rhythm, mel_2, MAX_LEN)
    return mel_

# your own path
source_speaker = '...'
source_mel = '...'
target_speaker = '...'
target_mel = '...'

if __name__ == '__main__':
    spect_vc = []
    i = 0
    for i in range(n_test):
        print(i)
        uid = ''

        # spkid = np.zeros((100,), dtype=np.float32)
        # spkid[speaker2index(speaker)] = 1.0
        # utterances.append(np.array([spkid]))

        x_org = np.load(os.path.join(mel_path, source_speaker, source_mel))
        f0_org = np.load(os.path.join(f0_path, source_speaker, source_mel))
        len_org = len(x_org)
        uid += source_mel[5:8]

        x_trg = np.load(os.path.join(mel_path, target_speaker, target_mel))
        f0_trg = np.load(os.path.join(f0_path, target_speaker, target_mel))
        len_trg = len(x_trg)
        uid += target_mel[5:8]

        print(uid)

        uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis, :, :], MAX_LEN)
        uttr_org_pad = torch.from_numpy(uttr_org_pad).to(device)
        f0_org_pad = np.pad(f0_org, (0, MAX_LEN - len_org), 'constant', constant_values=(0, 0))
        f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
        f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
        f0_org_onehot = torch.from_numpy(f0_org_onehot).to(device)
        uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

        uttr_trg_pad, len_trg_pad = pad_seq_to_2(x_trg[np.newaxis, :, :], MAX_LEN)
        uttr_trg_pad = torch.from_numpy(uttr_trg_pad).to(device)
        f0_trg_pad = np.pad(f0_trg, (0, MAX_LEN - len_trg), 'constant', constant_values=(0, 0))
        f0_trg_quantized = quantize_f0_numpy(f0_trg_pad)[0]
        f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
        f0_trg_onehot = torch.from_numpy(f0_trg_onehot).to(device)

        with torch.no_grad():
            f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
            f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
            f0_con_onehot = torch.zeros((1, MAX_LEN, 257), device=device)
            f0_con_onehot[0, torch.arange(MAX_LEN), f0_pred_quantized] = 1
        uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1)

        # conditions = ['U']
        conditions = ['R', 'F', 'RF', 'RU', 'FU', 'RFU']

        with torch.no_grad():
            for condition in conditions:
                if condition == 'R':  # R - uttr_trg_pad, F - uttr_f0_trg, U - emb_trg
                    x_identic_val = Generator_F(uttr_f0_org, uttr_trg_pad, torch.from_numpy(x_org).unsqueeze(0))
                if condition == 'F':
                    x_identic_val = Generator_F(uttr_f0_trg, uttr_org_pad, torch.from_numpy(x_org).unsqueeze(0))
                if condition == 'U':
                    x_identic_val = Generator_F(uttr_f0_org, uttr_org_pad, torch.from_numpy(x_trg).unsqueeze(0))
                if condition == 'RF':
                    x_identic_val = Generator_F(uttr_f0_trg, uttr_trg_pad, torch.from_numpy(x_org).unsqueeze(0))
                if condition == 'RU':
                    x_identic_val = Generator_F(uttr_f0_org, uttr_trg_pad, torch.from_numpy(x_trg).unsqueeze(0))
                if condition == 'FU':
                    x_identic_val = Generator_F(uttr_f0_trg, uttr_org_pad, torch.from_numpy(x_trg).unsqueeze(0))
                if condition == 'RFU':
                    x_identic_val = Generator_F(uttr_f0_trg, uttr_trg_pad, torch.from_numpy(x_trg).unsqueeze(0))

                if 'R' in condition:
                    uttr_trg = x_identic_val[0, :len_trg, :].cpu().numpy()
                else:
                    uttr_trg = x_identic_val[0, :len_org, :].cpu().numpy()

                spect_vc.append(('{}_{}_{}_{}'.format(source_speaker, target_speaker, uid, condition), uttr_trg))
        i += 1

    result_path = '/ceph/home/yangsc21/Python/autovc/My_model/assets/result_run_pitch_2_580000/800000_/'
    with open(result_path + 'result__.pkl', 'wb') as f:
        pickle.dump(spect_vc, f)


# spectrogram to waveform
import torch
import pickle
import sys
sys.path.append('../')
from autovc.synthesis import build_model
from autovc.synthesis import wavegen
import soundfile as sf

# if not os.path.exists('results'):
#     os.makedirs('results')

device = 'cuda:1'
model = build_model().to(device)
checkpoint = torch.load("/ceph/home/yangsc21/Python/autovc/autovc/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])


result_path = '/ceph/home/yangsc21/Python/autovc/My_model/assets/result_run_pitch_2_580000/800000_/'
spect_vc = pickle.load(open(result_path + 'result__dim2_pitch_wo_pitch.pkl', 'rb'))

output = "/ceph/home/yangsc21/Python/autovc/Final/My_model___/dim2_pitch_wo_pitch/"

print(len(spect_vc))

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)
    sf.write(output + name + '.wav', waveform, 16000)

