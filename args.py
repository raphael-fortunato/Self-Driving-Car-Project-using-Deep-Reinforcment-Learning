import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--num_envs',
            type=int,
            default=12,
            help='number of episodes')
    parser.add_argument(
            '--episodes',
            type=int,
            default=1000,
            help='number of episodes')
    parser.add_argument(
            '--n-cycles',
            type=int,
            default=1,
            help='number of cycles per epoch')
    parser.add_argument(
            '--num_eval_eps',
            type=int,
            default=5,
            help='number of cycles per epoch')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=64,
            help='size of the batch to pass through the network')
    parser.add_argument(
            '--buffer_size',
            type=int,
            default=1e6,
            help='size of the replay buffer')
    parser.add_argument(
            '--gamma',
            type=float,
            default=.99,
            help='the discount ratio')
    parser.add_argument(
            '--polyak',
            type=float,
            default=.95,
            help='polyak ratio')
    parser.add_argument(
            '--epsilon_decay',
            type=float,
            default=0.997,
            help='epsilon decay')
    parser.add_argument(
            '--min_epsilon',
            type=float,
            default=.1,
            help='min epsilon')
    parser.add_argument(
            '--cuda',
            action='store_true',
            help='use the gpu to train the networks')
    args = parser.parse_args()
    return args
