import os
import numpy as np
import pickle

root = "/ceph/home/yangsc21/Python/autovc/My_model/assets/"

f0_path = root + 'test_raptf0_/'
mel_path = root + 'test_spmel_'

split_speaker = "/ceph/home/yangsc21/Python/VCTK/wav16/split_speaker.txt"
with open(split_speaker, 'r', encoding='utf-8') as f:
    c = f.read().strip().split('\n')[1].strip().split(' ')
    # print(len(c))       # 100


def speaker2index(string):
    
    id = c.index(string)
    return id

speakers = []
for speaker in sorted(os.listdir(mel_path)):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)

    spkid = np.zeros((100,), dtype=np.float32)

    # spkid[speaker2index(speaker)] = 1.0
    # utterances.append(np.array([spkid]))
    utterances.append(None)

    feature = []
    uid = ''
    _, _, fileList = next(os.walk(os.path.join(mel_path, speaker)))
    for filename in fileList:
        feature.append(np.load(os.path.join(mel_path, speaker, filename)))
        uid += filename[5:8]
    _, _, fileList_ = next(os.walk(os.path.join(f0_path, speaker)))
    for filename in fileList_:
        feature.append(np.load(os.path.join(f0_path, speaker, filename)))
        uid += filename[5:8]
    assert len(feature[0]) == len(feature[1])
    feature.append(len(feature[0]))
    feature.append(uid)
    utterances.append(tuple(feature))
    speakers.append(utterances)

with open(os.path.join(mel_path, 'test.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

