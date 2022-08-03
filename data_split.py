import os
import random
import shutil

root = '/ceph/datasets/VCTK-Corpus/wav48/'

# root = '/ceph/home/yangsc21/Python/autovc/autovc/wavs/'
my_root = '/ceph/home/yangsc21/Python/VCTK/'
training_log = my_root + 'split_training_log.txt'
testing_log = my_root + 'split_testing_log.txt'


def get_filelist(dir):
    with open(training_log, 'w', encoding='utf-8') as f_train:
        with open(testing_log, 'w', encoding='utf-8') as f_test:
            for name in os.listdir(dir):
                f_train.write(name + '\n')
                f_test.write(name + '\n')
                wav_list = os.listdir(os.path.join(dir, name))
                random.shuffle(wav_list)
                split_point = int(len(wav_list) * 0.9)
                for wav in wav_list[:split_point]:
                    f_train.write(wav + '\n')
                for wav in wav_list[split_point:]:
                    f_test.write(wav + '\n')


def copy_train_file():
    with open(training_log, 'r', encoding='utf-8') as f_train:
        for line in f_train.readlines():
            if line.strip() != '' and line.strip()[-4:] != '.wav':
                if not os.path.exists(os.path.join(my_root, 'training_data', line.strip())):
                    os.makedirs(os.path.join(my_root, 'training_data', line.strip()))
            if line.strip()[-4:] == '.wav':
                shutil.copy(os.path.join(root, line[:4], line.strip()), os.path.join(my_root, 'training_data', line[:4]))


def copy_test_file():
    with open(testing_log, 'r', encoding='utf-8') as f_test:
        for line in f_test.readlines():
            if line.strip() != '' and line.strip()[-4:] != '.wav':
                if not os.path.exists(os.path.join(my_root, 'testing_data', line.strip())):
                    os.makedirs(os.path.join(my_root, 'testing_data', line.strip()))
            if line.strip()[-4:] == '.wav':
                shutil.copy(os.path.join(root, line[:4], line.strip()), os.path.join(my_root, 'testing_data', line[:4]))


if __name__ == '__main__':
    print('get filelist...')
    get_filelist(root)
    print('copy train file...')
    copy_train_file()
    print('copy test file...')
    copy_test_file()


import os
import random
import shutil
import numpy as np

VCTK_txt = '/ceph/datasets/VCTK-Corpus/txt/'

log_speaker_path = '/ceph/home/yangsc21/Python/VCTK/wav16/split_speaker.txt'
# log_dim_path = '/ceph/home/yangsc21/Python/VCTK/mel_dim.txt'
log_dim_path = '/ceph/home/yangsc21/Python/VCTK/wav16/mel_dim.txt'
log_dim_9_path = '/ceph/home/yangsc21/Python/VCTK/wav16/mel_dim_6.txt'
log_dim_100_path = '/ceph/home/yangsc21/Python/VCTK/wav16/mel_dim_100.txt'
log_test_path = '/ceph/home/yangsc21/Python/VCTK/wav16/select_from_test.txt'
log_train_path = '/ceph/home/yangsc21/Python/VCTK/wav16/select_from_train.txt'


def split_speaker(root, num_train):
    speaker_list = os.listdir(root)
    random.shuffle(speaker_list)
    print(len(speaker_list))
    with open(log_speaker_path, 'w', encoding='utf-8') as f:
        f.write('train:\n')
        for i in speaker_list[:num_train]:
            f.write(i + ' ')
        f.write('\n')
        f.write('test:\n')
        for i in speaker_list[num_train:]:
            f.write(i + ' ')


def mel_dim(root):
    with open(log_speaker_path, 'r', encoding='utf-8') as f:
        speaker_list = f.read().strip().split('\n')
        train_speaker_list = speaker_list[1].strip().split(' ')
        test_speaker_list = speaker_list[3].strip().split(' ')
        # print(train_speaker_list, test_speaker_list)
        with open(log_dim_path, 'w', encoding='utf-8') as ff:
            ff.write('train:\n')
            i = 0
            for speak in train_speaker_list:
                print(str(i) + '\t' + speak)
                for mel in os.listdir(os.path.join(root, speak)):
                    ff.write(str(i) + '\t')
                    ff.write(mel + '\t')
                    ff.write(str(np.load(os.path.join(root, speak, mel)).shape[0]))
                    ff.write('\n')
                i += 1

            i = 0
            ff.write('test:\n')
            for speak in test_speaker_list:
                print(str(i) + '\t' + speak)
                for mel in os.listdir(os.path.join(root, speak)):
                    ff.write(str(i) + '\t')
                    ff.write(mel + '\t')
                    ff.write(str(np.load(os.path.join(root, speak, mel)).shape[0]))
                    ff.write('\n')
                i += 1


def select_from_test():     # select_same_size
    speakers = [[] for _ in range(6)]
    with open(log_test_path, 'w', encoding='utf-8') as out:
        with open(log_dim_9_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip() == 'test:':
                    continue
                (id, rideo, dim) = line.strip().split('\t')
                speakers[int(id)].append([rideo, int(dim)])
            for speaker_i in range(len(speakers) - 1):
                print(speaker_i)
                for speaker_j in range(speaker_i + 1, len(speakers)):
                    for item_i in speakers[speaker_i]:
                        for item_j in speakers[speaker_j]:
                            if item_i[0][5:8] == item_j[0][5:8] and item_i[1] == item_j[1]:
                                with open(os.path.join(VCTK_txt, item_i[0][:4], item_i[0][:8] + '.txt'), 'r', encoding='utf-8') as f_i:
                                    with open(os.path.join(VCTK_txt, item_j[0][:4], item_j[0][:8] + '.txt'), 'r', encoding='utf-8') as f_j:
                                        if f_i.read().strip() == f_j.read().strip():
                                            # print(item_i, item_j)
                                            out.write(item_i[0] + '\t' + item_j[0] + '\t' + str(item_i[1]) + '\n')


def select_from_train():     # select_same_size
    speakers = [[] for _ in range(100)]
    with open(log_train_path, 'w', encoding='utf-8') as out:
        with open(log_dim_100_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip() == 'train:':
                    continue
                (id, rideo, dim) = line.strip().split('\t')
                speakers[int(id)].append([rideo, int(dim)])
            for speaker_i in range(len(speakers) - 1):
                print(speaker_i)
                for speaker_j in range(speaker_i + 1, len(speakers)):
                    for item_i in speakers[speaker_i]:
                        for item_j in speakers[speaker_j]:
                            if item_i[0][:4] == 'p315' or item_j[0][:4] == 'p315':
                                continue
                            if item_i[0][5:8] == item_j[0][5:8] and item_i[1] == item_j[1]:
                                with open(os.path.join(VCTK_txt, item_i[0][:4], item_i[0][:8] + '.txt'), 'r', encoding='utf-8') as f_i:
                                    with open(os.path.join(VCTK_txt, item_j[0][:4], item_j[0][:8] + '.txt'), 'r', encoding='utf-8') as f_j:
                                        if f_i.read().strip() == f_j.read().strip():
                                            # print(i, item_i, item_j)
                                            out.write(item_i[0] + '\t' + item_j[0] + '\t' + str(item_i[1]) + '\n')


def movefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print(srcfile + " not exist!")
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件


def move_rideo():
    source_path = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_6/'
    target_path = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_6_/'

    with open(log_test_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            (source1, source2, dim) = line.strip().split('\t')
            try:
                movefile(os.path.join(source_path, source1[:4], source1), os.path.join(target_path, source1[:4], source1))
                movefile(os.path.join(source_path, source2[:4], source2), os.path.join(target_path, source2[:4], source2))
            except:
                continue

    source_path = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_100/'
    target_path = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_100_/'
    with open(log_train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            (source1, source2, dim) = line.strip().split('\t')
            try:
                movefile(os.path.join(source_path, source1[:4], source1),
                         os.path.join(target_path, source1[:4], source1))
                movefile(os.path.join(source_path, source2[:4], source2),
                         os.path.join(target_path, source2[:4], source2))
            except:
                continue


def crop_len():
    root = "/ceph/home/yangsc21/Python/spmel_100_crop/"
    log = "/ceph/home/yangsc21/Python/VCTK/wav16/mel_dim_100.txt"
    out_log = "/ceph/home/yangsc21/Python/mel_dim_100_crop.txt"
    with open(log, 'r', encoding='utf-8') as f1:
        with open(out_log, 'w', encoding='utf-8') as f2:
            for line in f1.readlines():
                if line.strip() == 'train:':
                    f2.write(line)
                    continue
                (id, rideo, dim) = line.strip().split('\t')
                if int(dim) <= 128 * 3:
                    f2.write(line)
                else:
                    print(line.strip())
                    try:
                        os.remove(os.path.join(root, rideo[:4], rideo))
                    except:
                        continue

def crop_len_test():
    root = "/ceph/home/yangsc21/Python/spmel_6_crop/"
    log = "/ceph/home/yangsc21/Python/VCTK/wav16/mel_dim_6.txt"
    out_log = "/ceph/home/yangsc21/Python/mel_dim_6_crop.txt"
    with open(log, 'r', encoding='utf-8') as f1:
        with open(out_log, 'w', encoding='utf-8') as f2:
            for line in f1.readlines():
                if line.strip() == 'test:':
                    f2.write(line)
                    continue
                (id, rideo, dim) = line.strip().split('\t')
                if int(dim) <= 128 * 3:
                    f2.write(line)
                else:
                    print(line.strip())
                    try:
                        os.remove(os.path.join(root, rideo[:4], rideo))
                    except:
                        continue


if __name__ == '__main__':
    # root = '/ceph/home/yangsc21/Python/autovc/autovc/test_spmel/'
    # root = '/ceph/home/yangsc21/Python/VCTK/wav48_mel/'
    # root = "/ceph/home/yangsc21/Python/VCTK/wav16/spmel/"
    # split_speaker(root, 100)        #  100 for training, total 109 speakers
    # mel_dim(root)

    # select_from_test()
    # select_from_train()

    # move_rideo()

    # crop_len()
    crop_len_test()
