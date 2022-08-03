from model import Generator_MI, Generator_Decoder
from model import InterpLnr
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
import torch.nn as nn
from mi_estimators import CLUBSample_reshape
from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy
import matplotlib.pyplot as plt
# from VQ_Encoder import CPCLoss_sameSeq

torch.manual_seed(137)

# use demo data for simplicity
# make your own validation set as needed
valid_path = "/ceph/home/yangsc21/Python/autovc/SpeechSplit/assets/test_mel/test.pkl"
validation_pt = pickle.load(open(valid_path, "rb"))

MAX_LEN = 128 * 3


class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.lambda_cd = config.lambda_cd
        self.use_l1_loss = config.use_l1_loss
        self.use_VQCPC = config.use_VQCPC
        self.use_VQCPC_2 = config.use_VQCPC_2
        self.use_pitch = config.use_pitch
        self.use_adv = config.use_adv
        self.use_mi = config.use_mi
        self.advloss = nn.CrossEntropyLoss()
        self.device_ids = config.device_ids

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard.
        self.build_model()
        self.build_model2()

        self.cp_mi_net = CLUBSample_reshape(hparams.dim_neck * 2, hparams.dim_neck_3 * 2, 512)
        self.rc_mi_net = CLUBSample_reshape(hparams.dim_neck_2 * 2, hparams.dim_neck * 2, 512)
        self.rp_mi_net = CLUBSample_reshape(hparams.dim_neck_2 * 2, hparams.dim_neck_3 * 2, 512)

        self.optimizer_cp_mi_net = torch.optim.Adam(self.cp_mi_net.parameters(), lr=3e-4)
        self.optimizer_rc_mi_net = torch.optim.Adam(self.rc_mi_net.parameters(), lr=3e-4)
        self.optimizer_rp_mi_net = torch.optim.Adam(self.rp_mi_net.parameters(), lr=3e-4)

        # self.cpc = CPCLoss_sameSeq(n_speakers_per_batch=256, n_utterances_per_speaker=8, n_prediction_steps=6,
        #                            n_negatives=10, z_dim=512+256, c_dim=256)
        #
        # self.cpc_2 = CPCLoss_sameSeq(n_speakers_per_batch=256, n_utterances_per_speaker=8, n_prediction_steps=6,
        #                            n_negatives=10, z_dim=2, c_dim=256)
        #
        # self.optimizer_cpc = torch.optim.Adam(self.cpc.parameters(), self.g_lr, [self.beta1, self.beta2])
        # self.optimizer_cpc_2 = torch.optim.Adam(self.cpc_2.parameters(), self.g_lr, [self.beta1, self.beta2])


        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        # self.G = Generator(self.hparams)

        self.G1 = Generator_MI(self.hparams, self.use_VQCPC, self.use_VQCPC_2)

        self.Interp = InterpLnr(self.hparams)

        self.g_optimizer = torch.optim.Adam(self.G1.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G1, 'G')

        self.G1.to(self.device)
        self.G1 = torch.nn.DataParallel(self.G1, device_ids=self.device_ids, output_device=self.device_ids[0])  # 主要就是这句
        self.Interp.to(self.device)

    def build_model2(self):
        self.G2 = Generator_Decoder(self.hparams, self.use_pitch)
        self.g2_optimizer = torch.optim.Adam(self.G2.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.G2.to(self.device)
        self.G2 = torch.nn.DataParallel(self.G2, device_ids=self.device_ids, output_device=self.device_ids[0])  # 主要就是这句

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # print(name)
        # print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G1.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.g2_optimizer.zero_grad()
        # self.optimizer_cpc.zero_grad()
        # self.optimizer_cpc_2.zero_grad()

    def mi_first_forward(self, content, pitch, rhythm, optimizer_cp_mi_net, optimizer_rc_mi_net, optimizer_rp_mi_net):
        optimizer_cp_mi_net.zero_grad()
        optimizer_rc_mi_net.zero_grad()
        optimizer_rp_mi_net.zero_grad()
        content = content.detach()
        pitch = pitch.detach()
        rhythm = rhythm.detach()
        lld_cp_loss = -self.cp_mi_net.loglikeli(content, pitch)
        lld_rc_loss = -self.rc_mi_net.loglikeli(rhythm, content)
        lld_rp_loss = -self.rp_mi_net.loglikeli(rhythm, pitch)
        lld_cp_loss.backward()
        lld_rc_loss.backward()
        lld_rp_loss.backward()
        optimizer_cp_mi_net.step()
        optimizer_rc_mi_net.step()
        optimizer_rp_mi_net.step()
        return optimizer_cp_mi_net, optimizer_rc_mi_net, optimizer_rp_mi_net, lld_cp_loss, lld_rc_loss, lld_rp_loss

    def mi_second_forward(self, content, pitch, rhythm, x_real_org):
        if self.use_pitch:
            x_identic, mel_outputs_postnet, spk_pred, content_pred, pitch_predict = self.G2(content, pitch, rhythm, x_real_org)
            return x_identic, mel_outputs_postnet, spk_pred, content_pred, pitch_predict
        else:
            x_identic, mel_outputs_postnet, spk_pred, content_pred = self.G2(content, pitch, rhythm, x_real_org)
            return x_identic, mel_outputs_postnet, spk_pred, content_pred

    # =====================================================================================================================

    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        self.cp_mi_net.to(self.device)
        self.rc_mi_net.to(self.device)
        self.rp_mi_net.to(self.device)

        optimizer_cp_mi_net = self.optimizer_cp_mi_net
        optimizer_rc_mi_net = self.optimizer_rc_mi_net
        optimizer_rp_mi_net = self.optimizer_rp_mi_net

        # if self.use_VQCPC:
        #     self.cpc.to(self.device)
        # elif self.use_VQCPC_2:
        #     self.cpc_2.to(self.device)


        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print('Current learning rates, g_lr: {}.'.format(g_lr))

        # Print logs in specified order

        keys = ['G/loss_id', 'G/loss_id_psnt', 'spk_loss', 'content_adv_loss', 'mi_cp_loss', 'mi_rc_loss',
                'mi_rp_loss', 'lld_cp_loss', 'lld_rc_loss', 'lld_rp_loss']

        if self.use_VQCPC or self.use_VQCPC_2:
            keys.append('vq_loss')
            keys.append('cpc_loss')

        if self.use_pitch:
            keys.append('pitch_loss')

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)

            x_real_org = x_real_org.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            self.G1 = self.G1.train()
            self.G2 = self.G2.train()

            '''
            input: 
            x_real_org: (batch, max_len_pad, mel_dim(80))
            f0_org.shape: (batch, max_len_pad, 1)
            '''

            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)  # to (batch, max_len_pad, mel_dim + f0_dim(81))
            x_f0_intrp = self.Interp(x_f0,
                                     len_org)  # len_org: min_len_seq ~ max_len_seq 之间的随机数, (batch) to (batch, max_len_pad, 81)
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:, :, -1])[0]  # to (batch, max_len_pad, 257)
            x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)  # to (batch, max_len_pad, 257+80)

            '''
            x_f0_intrp_org: (batch, max_len_pad, 337)
            x_real_org: (batch, max_len_pad, 80)
            emb_org: (batch, 100)
            '''

            if self.use_VQCPC or self.use_VQCPC_2:
                content, pitch, rhythm, quantized_, x_f0_beforeVQ, c, x_f0_VQ = self.G1(x_f0_intrp_org, x_real_org)
            else:
                content, pitch, rhythm = self.G1(x_f0_intrp_org, x_real_org)

            if self.use_mi:
                for j in range(5):  # mi_iters
                    optimizer_cp_mi_net, optimizer_rc_mi_net, optimizer_rp_mi_net, lld_cp_loss, lld_rc_loss, lld_rp_loss = \
                        self.mi_first_forward(content, pitch, rhythm, optimizer_cp_mi_net, optimizer_rc_mi_net,
                                              optimizer_rp_mi_net)
            else:
                lld_cp_loss = torch.tensor(0.).to(self.device)
                lld_rc_loss = torch.tensor(0.).to(self.device)
                lld_rp_loss = torch.tensor(0.).to(self.device)

            if self.use_pitch:
                x_identic, mel_outputs_postnet, spk_pred, content_pred, pitch_predict = self.mi_second_forward(content, pitch, rhythm,
                                                                                            x_real_org)
            else:
                x_identic, mel_outputs_postnet, spk_pred, content_pred = self.mi_second_forward(content, pitch, rhythm,
                                                                                            x_real_org)

            # x_identic, mel_outputs_postnet, spk_pred, content_pred = self.G(x_f0_intrp_org, x_real_org, emb_org)  # to (batch, max_len_pad, 80)
            g_loss_id = F.mse_loss(x_real_org.to(self.device), x_identic.to(self.device), reduction='mean')
            g_loss_id_psnt = F.mse_loss(x_real_org.to(self.device), mel_outputs_postnet.to(self.device))

            # Backward and optimize.
            if self.use_l1_loss:
                g_loss = g_loss_id + g_loss_id_psnt + \
                         F.l1_loss(x_real_org.to(self.device), x_identic.to(self.device)) + \
                         F.l1_loss(x_real_org.to(self.device), mel_outputs_postnet.to(self.device))

            else:
                g_loss = g_loss_id + g_loss_id_psnt

            x_real_org.requirre_grad = False
            emb_org.requirre_grad = False

            if self.use_adv:
                spk_loss = self.advloss(spk_pred.to(self.device), emb_org)
                content_adv_loss = self.advloss(content_pred, emb_org)
                g_loss += self.lambda_cd * spk_loss + self.lambda_cd * content_adv_loss
            else:
                content_adv_loss = torch.tensor(0.).to(self.device)
                spk_loss = torch.tensor(0.).to(self.device)

            if self.use_mi:
                mi_cp_loss = 0.01 * self.cp_mi_net.mi_est(content, pitch)  # mi_weight
                mi_rc_loss = 0.01 * self.rc_mi_net.mi_est(rhythm, content)  # mi_weight
                mi_rp_loss = 0.01 * self.rp_mi_net.mi_est(rhythm, pitch)  # mi_weight
                g_loss += mi_cp_loss + mi_rc_loss + mi_rp_loss
            else:
                mi_cp_loss = torch.tensor(0.).to(self.device)
                mi_rc_loss = torch.tensor(0.).to(self.device)
                mi_rp_loss = torch.tensor(0.).to(self.device)

            # if self.use_VQCPC:
            #     e_latent_loss = F.mse_loss(x_f0_beforeVQ, quantized_.detach())
            #     vq_loss = 0.25 * e_latent_loss
            #
            #     cpc_loss, accuracy = self.cpc(x_f0_VQ, c)
            #
            #     g_loss += vq_loss + cpc_loss
            #
            # elif self.use_VQCPC_2:
            #     e_latent_loss = F.mse_loss(x_f0_beforeVQ, quantized_.detach())
            #     vq_loss = 0.25 * e_latent_loss
            #
            #     cpc_loss, accuracy = self.cpc_2(x_f0_VQ, c)
            #
            #     g_loss += vq_loss + cpc_loss

            if self.use_pitch:
                zeros = torch.zeros_like(f0_org)
                f0_zero = torch.where(f0_org == -1e10, zeros, f0_org)
                pitch_loss = F.mse_loss(f0_zero.to(self.device), pitch_predict.to(self.device), reduction='mean')
                g_loss += 0.1 * pitch_loss

            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            self.g2_optimizer.step()
            # if self.use_VQCPC:
            #     self.optimizer_cpc.step()
            #
            # elif self.use_VQCPC_2:
            #     self.optimizer_cpc_2.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['spk_loss'] = spk_loss.item()
            loss['content_adv_loss'] = content_adv_loss.item()
            loss['mi_cp_loss'] = mi_cp_loss.item()
            loss['mi_rc_loss'] = mi_rc_loss.item()
            loss['mi_rp_loss'] = mi_rp_loss.item()
            loss['lld_cp_loss'] = lld_cp_loss.item()
            loss['lld_rc_loss'] = lld_rc_loss.item()
            loss['lld_rp_loss'] = lld_rp_loss.item()
            # if self.use_VQCPC or self.use_VQCPC_2:
            #     loss['vq_loss'] = vq_loss.item()
            #     loss['cpc_loss'] = cpc_loss.item()
            if self.use_pitch:
                loss['pitch_loss'] = pitch_loss.item()



            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)
                # if self.use_VQCPC or self.use_VQCPC_2:
                #     print(100 * np.array(accuracy))

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i + 1)

            # Save model checkpoints.
            if (i + 1) >= 150000 and (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))

                checkpoint_state = {
                     'G1': self.G1.state_dict(),
                     'G2': self.G2.state_dict(),
                     'optimizerG1': self.g_optimizer.state_dict(),
                     'optimizerG2': self.g2_optimizer.state_dict(),
                     'cp_mi_net': self.cp_mi_net.state_dict(),
                     'rc_mi_net': self.rc_mi_net.state_dict(),
                     'rp_mi_net': self.rp_mi_net.state_dict(),
                     "optimizer_cp_mi_net": optimizer_cp_mi_net.state_dict(),
                     "optimizer_rc_mi_net": optimizer_rc_mi_net.state_dict(),
                     "optimizer_rp_mi_net": optimizer_rp_mi_net.state_dict(),
                     "epoch": i + 1
                }

                # if self.use_VQCPC:
                #     checkpoint_state["cpc"] = self.cpc.state_dict()
                #     checkpoint_state["optimizer_cpc"] = self.optimizer_cpc.state_dict()
                # if self.use_VQCPC_2:
                #     checkpoint_state["cpc"] = self.cpc_2.state_dict()
                #     checkpoint_state["optimizer_cpc"] = self.optimizer_cpc_2.state_dict()

                torch.save(checkpoint_state, G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Validation.
            if (i + 1) % self.sample_step == 0:
                self.G1 = self.G1.eval()
                self.G2 = self.G2.eval()
                with torch.no_grad():
                    loss_val = []
                    for val_sub in validation_pt:
                        # emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis, :, :], MAX_LEN)
                            # len_org = torch.tensor([val_sub[k][2]]).to(self.device)
                            f0_org = np.pad(val_sub[k][1], (0, MAX_LEN - val_sub[k][2]), 'constant',
                                            constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)

                            if self.use_VQCPC or self.use_VQCPC_2:
                                content, pitch, rhythm, _, _, _, _ = self.G1(x_f0, x_real_pad)  # emb_trg
                            else:
                                content, pitch, rhythm = self.G1(x_f0, x_real_pad)  # emb_trg
                            if self.use_pitch:
                                _, x_identic_val, _, _, _ = self.G2(content, pitch, rhythm, x_real_pad)
                            else:
                                _, x_identic_val, _, _ = self.G2(content, pitch, rhythm, x_real_pad)

                            # x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            g_loss_val = F.mse_loss(x_real_pad, x_identic_val, reduction='sum')
                            loss_val.append(g_loss_val.item())
                val_loss = np.mean(loss_val)
                print('Validation loss: {}'.format(val_loss))
                if self.use_tensorboard:
                    self.writer.add_scalar('Validation_loss', val_loss, i + 1)

            # plot test samples
            if (i + 1) % self.sample_step == 0:
                self.G1 = self.G1.eval()
                self.G2 = self.G2.eval()
                with torch.no_grad():
                    for val_sub in validation_pt:
                        # emb_org_val = torch.from_numpy(val_sub[1]).to(self.device)
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis, :, :], MAX_LEN)
                            # len_org = torch.tensor([val_sub[k][2]]).to(self.device)
                            f0_org = np.pad(val_sub[k][1], (0, MAX_LEN - val_sub[k][2]), 'constant',
                                            constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_org)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
                            x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)

                            # x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                            # x_identic_woF = self.G(x_f0_F, x_real_pad, emb_org_val)
                            # x_identic_woR = self.G(x_f0, torch.zeros_like(x_real_pad), emb_org_val)
                            # x_identic_woC = self.G(x_f0_C, x_real_pad, emb_org_val)

                            if self.use_VQCPC or self.use_VQCPC_2:
                                content, pitch, rhythm, _, _, _, _ = self.G1(x_f0, x_real_pad)  # emb_trg
                            else:
                                content, pitch, rhythm = self.G1(x_f0, x_real_pad)  # emb_trg
                            if self.use_pitch:
                                _, x_identic_val, _, _, _ = self.G2(content, pitch, rhythm, x_real_pad)
                            else:
                                _, x_identic_val, _, _ = self.G2(content, pitch, rhythm, x_real_pad)

                            if self.use_VQCPC or self.use_VQCPC_2:
                                content, pitch, rhythm, _, _, _, _ = self.G1(x_f0_F, x_real_pad)  # emb_trg
                            else:
                                content, pitch, rhythm = self.G1(x_f0_F, x_real_pad)  # emb_trg
                            if self.use_pitch:
                                _, x_identic_woF, _, _, _ = self.G2(content, pitch, rhythm, x_real_pad)
                            else:
                                _, x_identic_woF, _, _ = self.G2(content, pitch, rhythm, x_real_pad)

                            if self.use_VQCPC or self.use_VQCPC_2:
                                content, pitch, rhythm, _, _, _, _ = self.G1(x_f0, torch.zeros_like(x_real_pad))  # emb_trg
                            else:
                                content, pitch, rhythm = self.G1(x_f0, torch.zeros_like(x_real_pad))  # emb_trg
                            if self.use_pitch:
                                _, x_identic_woR, _, _, _ = self.G2(content, pitch, rhythm, x_real_pad)
                            else:
                                _, x_identic_woR, _, _ = self.G2(content, pitch, rhythm, x_real_pad)

                            if self.use_VQCPC or self.use_VQCPC_2:
                                content, pitch, rhythm, _, _, _, _ = self.G1(x_f0_C, x_real_pad)  # emb_trg
                            else:
                                content, pitch, rhythm = self.G1(x_f0_C, x_real_pad)  # emb_trg
                            if self.use_pitch:
                                _, x_identic_woC, _, _, _ = self.G2(content, pitch, rhythm, x_real_pad)
                            else:
                                _, x_identic_woC, _, _ = self.G2(content, pitch, rhythm, x_real_pad)

                            melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                            melsp_out = x_identic_val[0].cpu().numpy().T
                            melsp_woF = x_identic_woF[0].cpu().numpy().T
                            melsp_woR = x_identic_woR[0].cpu().numpy().T
                            melsp_woC = x_identic_woC[0].cpu().numpy().T

                            min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))

                            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
                            im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                            im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                            im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                            im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                            im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                            plt.savefig(f'{self.sample_dir}/{i + 1}_{val_sub[0]}_{k}.png', dpi=150)
                            plt.close(fig)
