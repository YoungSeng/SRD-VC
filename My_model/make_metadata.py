import os
import pickle
import numpy as np

# rootDir = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_100_crop_cat/'
rootDir = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/spmel/"

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


split_speaker = "/ceph/home/yangsc21/Python/VCTK/wav16/split_speaker.txt"
with open(split_speaker, 'r', encoding='utf-8') as f:
    c = f.read().strip().split('\n')[1].strip().split(' ')
    # print(len(c))       # 100


def speaker2index(string):
    id = c.index(string)
    return id

# print(speaker2index('p339'))        # 99


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))

    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    spkid = np.zeros((100,), dtype=np.float32)
    # if speaker == 'p226':
    #     spkid[1] = 1.0
    # else:
    #     spkid[7] = 1.0
    spkid[speaker2index(speaker)] = 1.0
    utterances.append(spkid)

    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)

with open(os.path.join(rootDir, 'train_speechsplit.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
