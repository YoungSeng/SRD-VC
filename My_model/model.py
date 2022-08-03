import torch
import torch.nn as nn
import torch.nn.functional as F
from AdversarialClassifier import AdversarialClassifier
from VQ_Encoder import VQEmbeddingEMA


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder_t(nn.Module):
    """Rhythm Encoder
    """

    def __init__(self, hparams):
        super().__init__()

        self.dim_neck_2 = hparams.dim_neck_2
        self.freq_2 = hparams.freq_2
        self.dim_freq = hparams.dim_freq
        self.dim_enc_2 = hparams.dim_enc_2
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp

        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i == 0 else self.dim_enc_2,
                         self.dim_enc_2,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_2 // self.chs_grp, self.dim_enc_2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, mask):  # (batch, 80, max_len_pad), None

        for conv in self.convolutions:
            x = F.relu(conv(x))  # to (batch, 128, max_len_pad)
        x = x.transpose(1, 2)  # to (batch, max_len_pad, 128)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # to (batch, max_len_pad, 2)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_2]  # to (batch, max_len_pad, 1)
        out_backward = outputs[:, :, self.dim_neck_2:]  # to (batch, max_len_pad, 1)

        codes = torch.cat((out_forward[:, self.freq_2 - 1::self.freq_2, :], out_backward[:, ::self.freq_2, :]), dim=-1)

        return codes  # to (batch, 24, 2)


class Encoder_6(nn.Module):
    """F0 encoder
    """

    def __init__(self, hparams):
        super().__init__()

        self.dim_neck_3 = hparams.dim_neck_3
        self.freq_3 = hparams.freq_3
        self.dim_f0 = hparams.dim_f0
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i == 0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True)

        self.interp = InterpLnr(hparams)

    def forward(self, x):

        for conv in self.convolutions:
            x = F.relu(conv(x))
            x = x.transpose(1, 2)
            x = self.interp(x, self.len_org.expand(x.size(0)))
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck_3]
        out_backward = outputs[:, :, self.dim_neck_3:]

        codes = torch.cat((out_forward[:, self.freq_3 - 1::self.freq_3, :],
                           out_backward[:, ::self.freq_3, :]), dim=-1)

        return codes


class Encoder_7(nn.Module):
    """Sync Encoder module
    """

    def __init__(self, hparams, use_VQCPC):
        super().__init__()

        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.dim_enc = hparams.dim_enc  # 512
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_freq = hparams.dim_freq  # 80
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_f0 = hparams.dim_f0

        self.use_VQCPC = use_VQCPC

        # convolutions for code 1
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i == 0 else self.dim_enc,
                         self.dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc // self.chs_grp, self.dim_enc))
            convolutions.append(conv_layer)
        self.convolutions_1 = nn.ModuleList(convolutions)

        self.lstm_1 = nn.LSTM(self.dim_enc, self.dim_neck, 2, batch_first=True, bidirectional=True)

        # convolutions for f0
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i == 0 else self.dim_enc_3,
                         self.dim_enc_3,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_3 // self.chs_grp, self.dim_enc_3))
            convolutions.append(conv_layer)
        self.convolutions_2 = nn.ModuleList(convolutions)

        self.lstm_2 = nn.LSTM(self.dim_enc_3, self.dim_neck_3, 1, batch_first=True, bidirectional=True)

        self.interp = InterpLnr(hparams)

        # self.codebook = VQEmbeddingEMA(n_embeddings=512, embedding_dim=512+256)
        # self.rnn = nn.LSTM(512+256, 256, batch_first=True)

    def forward(self, x_f0):  # (batch, 337, max_len_pad)

        x = x_f0[:, :self.dim_freq, :]  # to (batch, 80, max_len_pad)
        f0 = x_f0[:, self.dim_freq:, :]  # to (batch, 257, max_len_pad)

        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            x = F.relu(conv_1(x))  # to (batch, 512, max_len_pad)
            f0 = F.relu(conv_2(f0))  # to (batch, 256, max_len_pad)
            x_f0 = torch.cat((x, f0), dim=1).transpose(1, 2)  # to (batch, max_len_pad, 512+256(768))
            x_f0 = self.interp(x_f0, self.len_org.expand(x.size(0)))  # to (batch, max_len_pad, 512+256(768))
            x_f0 = x_f0.transpose(1, 2)  # to (batch, 768, max_len_pad)
            x = x_f0[:, :self.dim_enc, :]
            f0 = x_f0[:, self.dim_enc:, :]

        if self.use_VQCPC:  # modify for VQ
            x_f0_beforeVQ = x_f0.transpose(1, 2)  # to (batch, max_len_pad, 768)

            x_f0, _, _, _, quantized_ = self.codebook(x_f0_beforeVQ)
            self.rnn.flatten_parameters()
            c, _ = self.rnn(x_f0)  # to (batch, 128/2, 256)
        else:
            x_f0 = x_f0.transpose(1, 2)

        x = x_f0[:, :, :self.dim_enc]  # to (batch, max_len_pad, 512)
        f0 = x_f0[:, :, self.dim_enc:]  # to (batch, max_len_pad, 256)

        # code 1
        self.lstm_1.flatten_parameters()
        self.lstm_2.flatten_parameters()
        x = self.lstm_1(x)[0]  # to (batch, max_len_pad, num_directions(2) * hidden_size(8))
        f0 = self.lstm_2(f0)[0]  # to (batch, max_len_pad, 2 * 32)

        x_forward = x[:, :, :self.dim_neck]  # to (batch, max_len_pad, 8)
        x_backward = x[:, :, self.dim_neck:]  # to (batch, max_len_pad, 8)

        f0_forward = f0[:, :, :self.dim_neck_3]  # to (batch, max_len_pad, 32)
        f0_backward = f0[:, :, self.dim_neck_3:]  # # to (batch, max_len_pad, 32)

        codes_x = torch.cat((x_forward[:, self.freq - 1::self.freq, :],
                             x_backward[:, ::self.freq, :]), dim=-1)  # to (batch, 24(192/8), 16)

        codes_f0 = torch.cat((f0_forward[:, self.freq_3 - 1::self.freq_3, :],
                              f0_backward[:, ::self.freq_3, :]), dim=-1)  # to (batch, 24, 64)

        if self.use_VQCPC:
            return codes_x, codes_f0, quantized_, x_f0_beforeVQ, c, x_f0
        else:
            return codes_x, codes_f0


class Decoder_pitch(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # self.lstm = nn.LSTM(2 + 64, 256, 2, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM((hparams.dim_neck_2 + hparams.dim_neck_3) * 2, 256, 2, batch_first=True, bidirectional=True)
        self.linear_projection = LinearNorm(512, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.linear_projection(x)
        # x = torch.tanh(x)       # limited to (-1, 1)
        x = F.sigmoid(x)  # limited to (0,1)
        return x


class Decoder_3(nn.Module):
    """Decoder module
    """

    def __init__(self, hparams):
        super().__init__()
        self.dim_neck = hparams.dim_neck
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_emb = hparams.dim_spk_emb
        self.dim_freq = hparams.dim_freq
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = nn.LSTM(self.dim_neck * 2 + self.dim_neck_2 * 2 + self.dim_neck_3 * 2 + self.dim_emb,
                            512, 3, batch_first=True, bidirectional=True)

        self.linear_projection = LinearNorm(1024, self.dim_freq)
        self.postnet = Postnet()

    def forward(self, x):  # (batch, max_len_pad, 16+2+64+100)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # to (batch, max_len_pad, 2 * 512)

        decoder_output = self.linear_projection(outputs)  # to (batch, max_len_pad, 80)

        mel_outputs_postnet = self.postnet(decoder_output.transpose(2, 1))
        mel_outputs_postnet = decoder_output + mel_outputs_postnet.transpose(2, 1)

        return decoder_output, mel_outputs_postnet


class Decoder_4(nn.Module):
    """For F0 converter
    """

    def __init__(self, hparams):
        super().__init__()
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_f0 = hparams.dim_f0
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = nn.LSTM(self.dim_neck_2 * 2 + self.dim_neck_3 * 2,
                            256, 2, batch_first=True, bidirectional=True)

        self.linear_projection = LinearNorm(512, self.dim_f0)

    def forward(self, x):
        outputs, _ = self.lstm(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


class Generator_MI(nn.Module):
    """SpeechSplit model"""

    def __init__(self, hparams, use_VQCPC, use_VQCPC_2):
        super().__init__()

        self.encoder_1 = Encoder_7(hparams, use_VQCPC)
        self.encoder_2 = Encoder_t(hparams)
        self.encoder_speaker = SpeakerEncoder()

        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

        # self.codebook_2 = VQEmbeddingEMA(n_embeddings=512, embedding_dim=2)
        # self.rnn = nn.LSTM(2, 256, batch_first=True)

        self.use_VQCPC = use_VQCPC
        self.use_VQCPC_2 = use_VQCPC_2

    def forward(self, x_f0, x_org):  # (batch, max_len_pad, 337), (batch, max_len_pad, 80), (batch, 100)

        x_1 = x_f0.transpose(2, 1)  # to (batch, 337, max_len_pad)

        if self.use_VQCPC:
            codes_x, codes_f0, quantized_, x_f0_beforeVQ, c, x_f0 = self.encoder_1(
                x_1)  # to (batch, 24, 16), (batch, 24, 64)
        else:
            codes_x, codes_f0 = self.encoder_1(x_1)  # to (batch, 24, 16), (batch, 24, 64)

        code_exp_1 = codes_x.repeat_interleave(self.freq, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(self.freq_3, dim=1)
        # print(code_exp_1.shape, code_exp_3.shape)       # to (batch, max_len_pad, 16), (batch, max_len_pad, 64)

        x_2 = x_org.transpose(2, 1)  # to (batch, 80, max_len_pad)
        codes_2_ = self.encoder_2(x_2, None)  # to (batch, 24, 2)
        if self.use_VQCPC_2:

            codes_2, _, _, _, quantized_ = self.codebook_2(codes_2_)
            self.rnn.flatten_parameters()
            c, _ = self.rnn(codes_2)  # to (batch, 24, 256)
        else:
            codes_2 = codes_2_

        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)  # to (batch, max_len_pad, 2)

        if self.use_VQCPC:
            return code_exp_1, code_exp_3, code_exp_2, quantized_, x_f0_beforeVQ, c, x_f0
        elif self.use_VQCPC_2:
            return code_exp_1, code_exp_3, code_exp_2, quantized_, codes_2_, c, codes_2
        else:
            return code_exp_1, code_exp_3, code_exp_2


class Generator_Decoder(nn.Module):
    """SpeechSplit model"""

    def __init__(self, hparams, use_pitch):
        super().__init__()

        self.encoder_speaker = SpeakerEncoder()
        self.decoder = Decoder_3(hparams)

        # self.adversarial1 = AdversarialClassifier(input_emb=82, num_classes=100)
        # self.adversarial2 = AdversarialClassifier(hparams.dim_spk_emb, num_classes=100)

        self.adversarial1 = AdversarialClassifier(
            input_emb=(hparams.dim_neck + hparams.dim_neck_2 + hparams.dim_neck_3) * 2,
            dim_content=(hparams.dim_neck + hparams.dim_neck_2 + hparams.dim_neck_3) * 2, num_classes=100)
        self.adversarial2 = AdversarialClassifier(hparams.dim_spk_emb,
            dim_content=(hparams.dim_neck + hparams.dim_neck_2 + hparams.dim_neck_3) * 2, num_classes=100)

        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

        self.use_pitch = use_pitch

        self.decoder_pitch = Decoder_pitch(hparams)

    def forward(self, code_exp_1, code_exp_3, code_exp_2,
                x_org, MAX_LEN):  # (batch, max_len_pad, 337), (batch, max_len_pad, 80), (batch, 100)
        x_2 = x_org.transpose(2, 1)  # to (batch, 80, max_len_pad)

        code_exp_4 = self.encoder_speaker(x_2)  # to (batch, 256)

        # return code_exp_4

        spk_pred = self.adversarial2(code_exp_4.unsqueeze(1))  # to (batch, num_classes)
        content_dim_predict = self.adversarial1(torch.cat((code_exp_1, code_exp_2, code_exp_3), dim=-1))
        # -> (batch, max_len_pad, 16 + 2 + 64) to (batch, max_len_pad, 100)

        # code_exp_1.size(-1) in train, MAX_LEN in test
        encoder_outputs = torch.cat((code_exp_1, code_exp_2, code_exp_3,
                                     code_exp_4.unsqueeze(1).expand(-1, MAX_LEN, -1)),
                                    dim=-1)  # to (batch, max_len_pad, 16+2+64+256)     # not 100

        # print(encoder_outputs.shape)        # torch.Size([16, 384, 182])

        mel_outputs, mel_outputs_postnet = self.decoder(encoder_outputs)  # to (batch, max_len_pad, 80)

        if self.use_pitch:
            pitch_ = torch.cat((code_exp_2, code_exp_3), dim=-1)  # to (batch, max_len_pad, 2+64)
            pitch_predict = self.decoder_pitch(pitch_)  # to (batch, max_len_pad, 1)
            return mel_outputs, mel_outputs_postnet, spk_pred, content_dim_predict, pitch_predict
        else:
            return mel_outputs, mel_outputs_postnet, spk_pred, content_dim_predict


class Generator_6(nn.Module):
    """F0 converter
    """

    def __init__(self, hparams):
        super().__init__()

        self.encoder_2 = Encoder_t(hparams)
        self.encoder_3 = Encoder_6(hparams)
        self.decoder = Decoder_4(hparams)
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def forward(self, x_org, f0_trg):
        x_2 = x_org.transpose(2, 1)
        codes_2 = self.encoder_2(x_2, None)
        code_exp_2 = codes_2.repeat_interleave(self.freq_2, dim=1)

        x_3 = f0_trg.transpose(2, 1)
        codes_3 = self.encoder_3(x_3)
        code_exp_3 = codes_3.repeat_interleave(self.freq_3, dim=1)

        encoder_outputs = torch.cat((code_exp_2, code_exp_3), dim=-1)

        mel_outputs = self.decoder(encoder_outputs)

        return mel_outputs


class InterpLnr(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1  # 128//19 + 1 = 7

    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]  # torch.Size([109, 81])
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]

        return out_tensor

    def forward(self, x, len_seq):  # (batch, max_len_pad, 81) (batch)

        if not self.training:
            return x

        device = x.device
        batch_size = x.size(0)  # 16

        # indices of each sub segment, max_len_seg = 32
        indices = torch.arange(self.max_len_seg * 2, device=device).unsqueeze(0).expand(batch_size * self.max_num_seg,
                                                                                        -1)
        # to (batch*max_len_seg(16*7=112), 64)
        '''
        tensor([[ 0,  1,  2,  ..., 61, 62, 63],
                ...
                [ 0,  1,  2,  ..., 61, 62, 63]], device='cuda:0')
        '''

        # scales of each sub segment
        scales = torch.rand(batch_size * self.max_num_seg,
                            device=device) + 0.5  # to (112)      # [0, 1) + 0.5 → [0.5, 1.5)

        idx_scaled = indices / scales.unsqueeze(-1)  # to (112, 64)
        idx_scaled_fl = torch.floor(idx_scaled)  # 返回一个新的张量，该张量是输入元素的下限，小于或等于每个元素的最大整数, to (112, 64)

        lambda_ = idx_scaled - idx_scaled_fl  # to (112, 64)

        len_seg = torch.randint(low=self.min_len_seg,
                                high=self.max_len_seg,
                                size=(batch_size * self.max_num_seg, 1),
                                device=device)  # to (112, 1)       # 19 - 32之间的随机数

        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)  # to (112, 64)

        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)  # to (16, 7), 累加求和

        # offset starts from the 2nd segment
        offset = F.pad(offset[:, :-1], (1, 0), value=0).view(-1, 1)  # to (112, 1)

        idx_scaled_org = idx_scaled_fl + offset  # to (112, 64)

        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)  # to (112)

        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)  # to (112, 64)

        idx_mask_final = idx_mask & idx_mask_org  # to (112, 64)

        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)  # to (16)

        index_1 = torch.repeat_interleave(torch.arange(batch_size,
                                                       device=device), counts)  # to (1591)
        index_2_fl = idx_scaled_org[idx_mask_final].long()  # to (1591)

        index_2_cl = index_2_fl + 1

        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)  # to (1591, 1)
        y = (1 - lambda_f) * y_fl + lambda_f * y_cl

        sequences = torch.split(y, counts.tolist(), dim=0)
        seq_padded = self.pad_sequences(sequences)
        return seq_padded


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    # padding
    inp = F.pad(inp,
                pad=pad,
                mode=pad_type)
    out = layer(inp)
    return out


def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out


def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()


class SpeakerEncoder(nn.Module):
    '''
    reference from speaker-encoder of AdaIN-VC: https://github.com/jjery2243542/adaptive_voice_conversion/blob/master/model.py
    '''

    def __init__(self, c_in=80, c_h=128, c_out=256, kernel_size=5,
                 bank_size=8, bank_scale=1, c_bank=128,
                 n_conv_blocks=6, n_dense_blocks=6,
                 subsample=[1, 2, 1, 2, 1, 2], act='relu', dropout_rate=0):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                                                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                                                 for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):  # (batch, 80, len_crop)
        out = conv_bank(x, self.conv_bank, act=self.act)  # to (batch, 1104, len_crop)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)  # to (batch, 128, len_crop)
        out = self.act(out)  # (batch, 128, len_crop)
        # conv blocks
        out = self.conv_blocks(out)  # (batch, 128, 16)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)  # (batch, 128)
        # dense blocks
        out = self.dense_blocks(out)
        # print(out.shape)
        out = self.output_layer(out)  # (batch, 256)
        return out
