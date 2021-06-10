from argparse import ArgumentParser

from pandas import DataFrame
from torch import randn
from torch.nn import Linear, ReLU, Sequential


def generate_dataset(bs=1000000, n_in=200):
    n_hidden1 = n_in//2
    n_hidden2 = n_in//4
    n_hidden3 = n_in//8

    net = Sequential(Linear(n_in, n_hidden1),
                     ReLU(),
                     Linear(n_hidden1, n_hidden2),
                     ReLU(),
                     Linear(n_hidden2, n_hidden3),
                     ReLU(),
                     Linear(n_hidden3, 1))

    x = randn(bs, n_in)
    y = net(x).detach()
    return x, y


def save_tensor(t, name):
    t = DataFrame(t.numpy()).astype('float32')
    t.to_csv(name, sep='\n', header=False,
             index=False)


def save_dataset(x, y):
    save_tensor(x, 'x.csv')
    save_tensor(y, 'y.csv')


def main(bs=1000000, n_in=200):
    x, y = generate_dataset(bs, n_in)
    save_dataset(x, y)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bs', default=1000000)
    parser.add_argument('--n_in', default=200)
    args = parser.parse_args()

    bs = int(args.bs)
    n_in = int(args.n_in)

    main(bs, n_in)
