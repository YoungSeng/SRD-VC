
import os
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# from sklearn import datasets
#
# digits = datasets.load_digits(n_class=6)
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape
#
# print(X.shape)      # (1083, 64)
# print(y.shape)      # (1083,)

# '''显示原始数据'''
# n = 20  # 每行20个数字，每列20个数字
# img = np.zeros((10 * n, 10 * n))
# for i in range(n):
#     ix = 10 * i + 1
#     for j in range(n):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
# plt.figure(figsize=(8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('1.png')


# print(f">>> t-SNE fitting")
# # 初始化一个TSNE模型，这里的参数设置可以查看SKlearn的官网
# tsne = TSNE(n_components=2, init='pca', perplexity=30)
# Y = tsne.fit_transform(features)
# print(f"<<< fitting over")
# print(Y)

# fig, ax = plt.subplots()
# fig.set_size_inches(21.6, 14.4)
# plt.axis('off')
# print(f">>> plotting images")
# imscatter(Y[:, 0], Y[:, 1], imgs, zoom=0.1, ax=ax)
# print(f"<<< plot over")
# plt.savefig(fname='figure.eps', format='eps')
# plt.show()

# tsne = TSNE(n_components=2, init='pca', random_state=501)
# X_tsne = tsne.fit_transform(X)
#
# print("Org data dimension is {}.Embedded data dimension is {} ".format(X.shape[-1], X_tsne.shape[-1]))
#
# '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.savefig('tmp.pdf')


from model import Generator_Decoder
from hparams import hparams as hparams_
import torch


use_pitch = True

device = 'cuda:0'
G2 = Generator_Decoder(hparams_, use_pitch).eval().to(device)
G2 = torch.nn.DataParallel(G2, device_ids=[0, 1], output_device=0)

G_path = "/ceph/home/yangsc21/Python/autovc/My_model/run_pitch_2/models/620000-G.ckpt"
g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
G2.load_state_dict(g_checkpoint['G2'])

test_root = '/ceph/home/yangsc21/Python/VCTK/wav16/spmel_6/'
num_uttr = 50

X = []
y = []

speaker2color = {
    'p335':'blue',
    'p264':"cyan",
    'p247':"green",
    'p278':"red",
    'p272':"yellow",
    'p262':"magenta",
}

for speaker in os.listdir(test_root):
    print(speaker)
    for mel in random.sample(os.listdir(os.path.join(test_root, speaker)), num_uttr):
        mel_featrue = np.load(os.path.join(test_root, speaker, mel))        # (len, 80)
        speaker_embedding = G2(None, None, None, torch.from_numpy(mel_featrue).unsqueeze(0), None)      # to (1, len, 80)
        # print(mel_featrue.shape)
        X.append(speaker_embedding.squeeze(0).to('cpu').detach().numpy())
        y.append(speaker)

X = np.array(X)

tsne = TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {} ".format(X.shape[-1], X_tsne.shape[-1]))

z = [[] for _ in range(6)]
speakers = ["p335", "p264", "p247", "p278", "p272", "p262"]

for item in range(len(y)):
    z[speakers.index(y[item])].append(X_tsne[item])



for i in range(len(z)):

    plt.scatter(np.array(z[i])[:, 0], np.array(z[i])[:, 1], c=speaker2color[speakers[i]], label=speakers[i])

plt.legend()
plt.savefig('1.pdf')

'''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

# print(X_norm)



