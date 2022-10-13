import wave
import os

root = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_100_crop/'
wav_root = '/ceph/datasets/VCTK-Corpus/wav16/'
output_root = '/ceph/home/yangsc21/Python/VCTK/wav16/cat/'
speakers = os.listdir(root)

for speaker in speakers:
    mels = os.listdir(os.path.join(root, speaker))
    wav_input = []
    for mel in mels:
        wav_input.append(os.path.join(wav_root, speaker, mel[:-4] + '.wav'))

    if not os.path.exists(os.path.join(output_root, speaker)):      # 如果不存在则创建目录
        os.makedirs(os.path.join(output_root, speaker))       # 创建目录操作函数
    output = os.path.join(output_root, speaker, speaker + '_cat' + '.wav')
    data = []
    cnt = 0
    for infile in wav_input:
        try:
            w = wave.open(infile, 'rb')
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()
        except:
            print(infile)
            cnt += 1
    output = wave.open(output, 'wb')
    output.setparams(data[0][0])
    for i in range(len(wav_input) - cnt):
        output.writeframes(data[i][1])
    output.close()

# path = '/ceph/home/yangsc21/Python/autovc/autovc/wavs/p225/'
#
#
# inputs = os.listdir(path)
# pathnames = [os.path.join(path, c) for c in inputs]
# output = '/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/Others/test.wav'
#
# print(pathnames)
#
# data= []
# for infile in pathnames:
#     w = wave.open(infile, 'rb')
#     data.append([w.getparams(), w.readframes(w.getnframes())] )
#     w.close()
#
# output = wave.open(output, 'wb')
# output.setparams(data[0][0])
#
# for i in range(len(pathnames)):
#     output.writeframes(data[i][1])
# output.close()
