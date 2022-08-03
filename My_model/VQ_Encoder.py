import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''

    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
        self.encoder = nn.Sequential(
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def encode(self, mel):
        z = self.conv(mel)
        z_beforeVQ = self.encoder(z.transpose(1, 2))
        z, r, indices = self.codebook.encode(z_beforeVQ)
        c, _ = self.rnn(z)
        return z, c, z_beforeVQ, indices

    def forward(self, mels):  # (batch, 80, len_crop)
        z = self.conv(mels.float())  # (bz, 80, 128) -> (bz, 512, 128/2)
        z_beforeVQ = self.encoder(z.transpose(1, 2))  # (bz, 512, 128/2) -> (bz, 128/2, 512) -> (bz, 128/2, 64)
        z, r, loss, perplexity = self.codebook(z_beforeVQ)  # z: (bz, 128/2, 64)
        c, _ = self.rnn(z)  # to (batch, 128/2, 256)
        """
        loss: tensor(0.0041, device='cuda:6', grad_fn=<MulBackward0>) 
        perplexity: tensor(25.4631, device='cuda:6')
        """
        return z, c, z_beforeVQ, loss, perplexity


class VQEmbeddingEMA(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''

    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)  # only change during forward
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        residual = x - quantized
        return quantized, residual, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # calculate the distance between each ele in embedding and x

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized_ = F.embedding(indices, self.embedding)
        quantized_ = quantized_.view_as(x)

        if self.training:  # EMA based codebook learning
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized_.detach())
        loss = self.commitment_cost * e_latent_loss

        residual = x - quantized_

        quantized = x + (quantized_ - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, residual, loss, perplexity, quantized_


class CPCLoss_sameSeq(nn.Module):
    '''
    CPC-loss calculation: negative samples are drawn within-sequence/utterance
    '''

    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss_sameSeq, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c):  # z:(256, 64, 64), c:(256, 64, 256), ((bz, 128/2, 64), (batch, 128/2, 256))
        length = z.size(
            1) - self.n_prediction_steps    # 64-6=58, length is the total time-steps of each utterance used for calculated cpc loss
        n_speakers_per_batch = z.shape[0]  # each utterance is treated as a speaker, 256
        c = c[:, :-self.n_prediction_steps, :]  # (256, 58, 256)

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps + 1):
            z_shift = z[:, k:length + k, :]  # (256, 58, 64), positive samples
            # 1:59, 2:60, 3:61, 4:62, 5:63, 6:64
            Wc = self.predictors[k - 1](c)  # (256, 58, 256) -> (256, 58, 64)

            # seq_index: [1, 58) (256, 10, 58)
            seq_index = torch.randint(
                1, length,
                size=(
                    n_speakers_per_batch,
                    self.n_negatives,
                    length
                ),
                device=z.device
            )
            """
            seq_index[-1]:
            [[ 9, 19, 27,  ..., 53, 45,  1],
             [12, 22,  3,  ..., 54, 45,  3],
             [19, 51, 42,  ..., 52, 47, 21],
             ...,
             [31, 29, 22,  ..., 36, 43, 30],
             [29, 30, 45,  ..., 31, 46, 21],
             [55, 17,  3,  ..., 49, 27, 33]]])

            torch.arange(length, device=z.device):
                tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                        54, 55, 56, 57])
            """
            seq_index += torch.arange(length, device=z.device)  # (1)
            """
            seq_index
            [[  9,  20,  29,  ..., 108, 101,  58],
             [ 12,  23,   5,  ..., 109, 101,  60],
             [ 19,  52,  44,  ..., 107, 103,  78],
             ...,
             [ 31,  30,  24,  ...,  91,  99,  87],
             [ 29,  31,  47,  ...,  86, 102,  78],
             [ 55,  18,   5,  ..., 104,  83,  90]]])
            """
            seq_index = torch.remainder(seq_index,
                                        length)  # (2) (1)+(2) ensures that the current positive frame will not be selected as negative sample...
            # to (256, 10, 58)
            """
            seq_index
             [[ 9, 20, 29,  ..., 50, 43,  0],
              [12, 23,  5,  ..., 51, 43,  2],
              [19, 52, 44,  ..., 49, 45, 20],
              ...,
              [31, 30, 24,  ..., 33, 41, 29],
              [29, 31, 47,  ..., 28, 44, 20],
              [55, 18,  5,  ..., 46, 25, 32]]])
            """

            speaker_index = torch.arange(n_speakers_per_batch, device=z.device)  # within-utterance sampling
            speaker_index = speaker_index.view(-1, 1, 1)  # to (256(n_speakers_per_batch), 1, 1)

            z_negatives = z_shift[speaker_index, seq_index,
                          :]  # (256,10,58,64), z_negatives[i,:,j,:] is the negative samples set for ith utterance and jth time-step
            # (256, 58, 64)[(256, 1, 1), (256, 10, 58), :]

            zs = torch.cat((z_shift.unsqueeze(1), z_negatives), dim=1)  # (256,11,58,64)

            f = torch.sum(zs * Wc.unsqueeze(1) / math.sqrt(self.z_dim),       # self.z_dim
                          dim=-1)   # vector product in fact...
            # (256,11,58,64) * (256, 1, 58, 64) -> (256, 11, 58, 64) -> (256,11,58)

            labels = torch.zeros(
                n_speakers_per_batch, length,
                dtype=torch.long, device=z.device
            )       # (256, 58)

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels  # (256, 58)
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies
