import argparse

def get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', type=int, default=90, help='number of episodes')
    parser.add_argument('--n-batch', type=int, default=40, help='number of batches to run through')
    parser.add_argument('--n-cycles', type=int, default=50, help='number of cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batch to pass through the network')
    parser.add_argument('--n-evaluate', type=int, default=50, help='number of evaluate episodes')
    parser.add_argument('--n-record', type=int, default=5, help='number of record episodes')
    parser.add_argument('--polyak', type=float, default=.95, help='polyak ratio')
    parser.add_argument('--buffer size', type=int, default=1e6, help='size of the replay buffer')
    parser.add_argument('--l2-norm', type=float, default= 1., help='the l2 regularization of the actor')
    parser.add_argument('--gamma', type=float, default= .99, help='the discount ratio')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2,
            help='the rollouts per mpi')
    parser.add_argument('--cuda', action='store_true',
            help='use the gpu to train the networks')
    args = parser.parse_args()
    return args

