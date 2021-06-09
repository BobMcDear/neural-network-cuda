from argparse import ArgumentParser

from pandas import read_csv
from torch import cuda, no_grad, tensor
from torch.nn import Linear, MSELoss, ReLU, Sequential 


def get_data(bs=100000, n_in=100, device='cuda'):
    inp = tensor(read_csv('../data/x.csv', sep='\n', header=None).values.astype('float32')).view(bs, n_in).to(device)
    targ = tensor(read_csv('../data/y.csv', sep='\n', header=None).values.astype('float32')).to(device)
    return inp, targ


def one_epoch(net, inp, targ, loss_func):
    pred = net(inp)
    loss_func(pred, targ).backward()

    for p in net.parameters():
        p.data -= 0.1*p.grad
    

def train(net, inp, targ, n_epochs=10):
    mse = MSELoss()

    for _ in range(n_epochs):
        one_epoch(net, inp, targ, mse)

    with no_grad():
        pred = net(inp)
        loss = mse(pred, targ)
    print(f'The final loss is: {loss}')
    

def main(bs=100000, n_in=100, n_epochs=10):
    n_hidden1 = n_in//2
    n_hidden2 = n_in//4
    n_hidden3 = n_in//8
    device = 'cuda' if cuda.is_available() else 'cpu'

    inp, targ = get_data(bs, n_in, device)
    net = Sequential(Linear(n_in, n_hidden1),
                     ReLU(),
                     Linear(n_hidden1, n_hidden2),
                     ReLU(),
                     Linear(n_hidden2, n_hidden3),
                     ReLU(),
                     Linear(n_hidden3, 1)).to(device)

    train(net, inp, targ, n_epochs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bs', default=100000)
    parser.add_argument('--n_in', default=100)
    parser.add_argument('--n_epochs', default=10)
    args = parser.parse_args()

    bs = args.bs
    n_in = args.n_in
    n_epochs = args.n_epochs

    main(bs, n_in, n_epochs)
