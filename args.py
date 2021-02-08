import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--num_envs',
            type=int,
            default=8,
            help='number of episodes')
    parser.add_argument(
            '--episodes',
            type=int,
            default=100,
            help='number of episodes')
    parser.add_argument(
            '--n-cycles',
            type=int,
            default=50,
            help='number of cycles per epoch')
    parser.add_argument(
            '--num_eval_eps',
            type=int,
            default=5,
            help='number of cycles per epoch')
    parser.add_argument(
            '--n-batches',
            type=int,
            default=40,
            help='number of batch per epoch')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=2096,
            help='size of the batch to pass through the network')
    parser.add_argument(
            '--n-evaluate',
            type=int,
            default=20,
            help='number of evaluate episodes')
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
            '--epsilon',
            type=float,
            default=.98,
            help='start epsilon')
    parser.add_argument(
            '--max_clip_norm',
            type=int,
            default=10,
            help='max gradient norm')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=12,
            help='num workers')
    parser.add_argument(
            '--cuda',
            action='store_true',
            help='use the gpu to train the networks')
    args = parser.parse_args()
    return args
