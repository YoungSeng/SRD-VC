
from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

hparams__ = HParams(
    freq=8,
    dim_neck=8,  # 8
    freq_2=8,
    dim_neck_2=1,
    freq_3=8,
    dim_neck_3=32,

    dim_enc=512,        # 512
    dim_enc_2=128,
    dim_enc_3=256,

    dim_freq=80,
    dim_spk_emb=256,     # 100
    dim_f0=257,
    dim_dec=512,
    len_raw=128,
    chs_grp=16,

    # interp
    min_len_seg=19,
    max_len_seg=32,
    min_len_seq=64,     # 64
    max_len_seq=128,        # 128
    max_len_pad=192,        # 192

    # data loader
    root_dir='/ceph/home/yangsc21/Python/VCTK/wav16/spmel_100_crop_cat/',
    feat_dir='/ceph/home/yangsc21/Python/VCTK/wav16/raptf0_100_crop_cat/',
    # root_dir="/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/spmel/",
    # feat_dir='/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/raptf0/',
    batch_size=16,
    mode='train',
    shuffle=True,
    num_workers=0,
    samplier=8,
)

# Default hyperparameters:
hparams = HParams(
    # model
    freq=8,
    dim_neck=8 * 2,     # 8
    freq_2=8,
    dim_neck_2=1 * 2,
    freq_3=8,
    dim_neck_3=32 * 2,

    # freq=8,
    # dim_neck=8,  # 8
    # freq_2=8,
    # dim_neck_2=1,
    # freq_3=8,
    # dim_neck_3=32,

    dim_enc=512,        # 512
    dim_enc_2=128,
    dim_enc_3=256,

    dim_freq=80,
    dim_spk_emb=256,     # 100
    dim_f0=257,
    dim_dec=512,
    len_raw=128,
    chs_grp=16,

    # interp
    min_len_seg=19,
    max_len_seg=32,
    min_len_seq=64,     # 64
    max_len_seq=128,        # 128
    max_len_pad=192,        # 192

    # data loader
    root_dir='/ceph/home/yangsc21/Python/VCTK/wav16/spmel_100_crop_cat/',
    feat_dir='/ceph/home/yangsc21/Python/VCTK/wav16/raptf0_100_crop_cat/',
    # root_dir="/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/spmel/",
    # feat_dir='/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/raptf0/',
    batch_size=16,
    mode='train',
    shuffle=True,
    num_workers=0,
    samplier=8,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
