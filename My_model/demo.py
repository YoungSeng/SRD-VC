# demo conversion
import torch
import pickle
import numpy as np
from hparams import hparams as hparams_
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_6 as F0_Converter
from model import Generator_MI, Generator_Decoder
import os

device = 'cuda:5'

use_VQCPC = False
use_VQCPC_2 = False
use_pitch = True

G1 = Generator_MI(hparams_, use_VQCPC, use_VQCPC_2).eval().to(device)
G2 = Generator_Decoder(hparams_, use_pitch).eval().to(device)

G1 = torch.nn.DataParallel(G1, device_ids=[5, 0], output_device=5)  # 主要就是这句
G2 = torch.nn.DataParallel(G2, device_ids=[5, 0], output_device=5)  # 主要就是这句

G_path = "/ceph/home/yangsc21/Python/autovc/My_model/run_pitch_2/models/800000-G.ckpt"
g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
G1.load_state_dict(g_checkpoint['G1'])
G2.load_state_dict(g_checkpoint['G2'])


P = F0_Converter(hparams_).eval().to(device)
P_path = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/640000-P.ckpt"
p_checkpoint = torch.load(P_path, map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])

# test_pkl_path = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/test_mel/test.pkl"
test_pkl_path = "/ceph/home/yangsc21/Python/autovc/My_model/assets/test_spmel_/test.pkl"
metadata = pickle.load(open(test_pkl_path, "rb"))

MAX_LEN = 128 * 3       # 192

sbmt_i = metadata[0]
# emb_org = torch.from_numpy(sbmt_i[1]).to(device)
x_org, f0_org, len_org, uid_org = sbmt_i[2]

sbmt_j = metadata[1]
# emb_trg = torch.from_numpy(sbmt_j[1]).to(device)
x_trg, f0_trg, len_trg, uid_trg = sbmt_j[2]

# MAX_LEN = max(len_org, len_trg)
# while MAX_LEN % 8 != 0:
#     MAX_LEN += 1
# print(MAX_LEN)

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

with torch.no_grad():
    f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
    f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    f0_con_onehot = torch.zeros((1, MAX_LEN, 257), device=device)
    f0_con_onehot[0, torch.arange(MAX_LEN), f0_pred_quantized] = 1
uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1)

# conditions = ['FU']
conditions = ['U', 'R', 'F', 'RF', 'RU', 'FU', 'RFU']
spect_vc = []
with torch.no_grad():
    for condition in conditions:
        if condition == 'R':      # R - uttr_trg_pad, F - uttr_f0_trg, U - emb_trg
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

        spect_vc.append(('{}_{}_{}_{}'.format(sbmt_i[0], sbmt_j[0], uid_org, condition), uttr_trg))



# spectrogram to waveform
import torch

import sys
sys.path.append('../')
from autovc.synthesis import build_model
from autovc.synthesis import wavegen
import soundfile as sf

# if not os.path.exists('results'):
#     os.makedirs('results')

device = 'cuda:0'
model = build_model().to(device)
checkpoint = torch.load("/ceph/home/yangsc21/Python/autovc/autovc/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

# result_path = "/ceph/home/yangsc21/Python/autovc/Final/test_wav_16000/"
result_path = "/ceph/home/yangsc21/Python/autovc/My_model/assets/test_result_/"


for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)
    # librosa.output.write_wav('assets/results/'+name+'.wav', waveform, sr=16000)
    sf.write(result_path + name + '.wav', waveform, 16000)
