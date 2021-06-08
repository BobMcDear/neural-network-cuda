from pandas import DataFrame
from torch import randn
from torch.nn import Sequential, Linear, ReLU


def generate_dataset():
    net = Sequential(Linear(10, 8),
                    ReLU(),
                    Linear(8, 5),
                    ReLU(),
                    Linear(5, 1))

    x = randn(1000, 10)
    y = net(x).detach()
    return x, y


def save_tensor(t, name):
    t = DataFrame(t).astype('float')
    t.to_csv(name, sep='\n', header=False,
             index=False)


def save_dataset(x, y):
    save_tensor(x, 'x.csv')
    save_tensor(y, 'y.csv')


def main():
    x, y = generate_dataset()
    save_dataset(x, y)


if __name__ == '__main__':
    main()
