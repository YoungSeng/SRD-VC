import os
import argparse
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from hparams import hparams, hparams_debug_string


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    vcc_loader = get_loader(hparams)

    # Solver for training
    solver = Solver(vcc_loader, config, hparams)

    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--lambda_cd', type=float, default=0.1, help='weight for hidden code loss')
    parser.add_argument('--use_l1_loss', type=str2bool, default=True)
    parser.add_argument('--use_VQCPC', type=str2bool, default=False)     # VQCPC after RR
    parser.add_argument('--use_VQCPC_2', type=str2bool, default=False)     # VQCPC after Mel Encoder (rhythm)
    parser.add_argument('--use_pitch', type=str2bool, default=True)
    parser.add_argument('--use_adv', type=str2bool, default=True)
    parser.add_argument('--use_mi', type=str2bool, default=True)


    # Miscellaneous.
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device_ids', type=list, default=[3, 0])
    parser.add_argument('--device_id', type=int, default=3)

    # Directories.
    run_path = 'run_dim2_pitch'
    parser.add_argument('--log_dir', type=str, default=run_path + '/logs')
    parser.add_argument('--model_save_dir', type=str, default=run_path + '/models')
    parser.add_argument('--sample_dir', type=str, default=run_path + '/samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)  # 100
    parser.add_argument('--sample_step', type=int, default=10000)
    parser.add_argument('--model_save_step', type=int, default=50000)  # 20000

    config = parser.parse_args()
    print(config)
    # print(hparams_debug_string())
    main(config)
