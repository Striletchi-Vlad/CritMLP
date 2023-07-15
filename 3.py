# TODO add support for any scale-invariant activation function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


def nu(gamma, af):
    """
    gamma: weight of the residual connections
    activation: activation function
    For now, linear is f(x)=x
    """
    # K* = 0 universality class
    if af in ['tanh', 'sin']:
        return 2/3 * (1 - gamma**4)

    # Scale invariant universality class
    if af in ['relu', 'linear']:
        if af == 'relu':
            A2 = 1/2
            A4 = 1/2
        if af == 'linear':
            A2 = 1
            A4 = 1
        g1 = 1 - gamma**2
        g2 = 4 * gamma**2
        return g1 * (g1 * (3*A4 / A2**2 - 1) + g2)

    # Half-stable universality class
    # TODO implement half-stable universality class
    if af == 'gelu':
        return nu(gamma, 'relu')


def gamma_from_nu(nu, af):
    """
    Reverse engineer the gamma parameter, governing the residual
    connections, given nu.
    """
    # K* = 0 universality class
    if af in ['tanh', 'sin']:
        return np.sqrt(np.sqrt(1 - 3*nu/2))

    # Scale invariant universality class
    if af in ['relu', 'linear']:
        if af == 'relu':
            A2 = 1/2
            A4 = 1/2
        if af == 'linear':
            A2 = 1
            A4 = 1

        const1 = 3*A4 / A2**2 - 1
        a = const1 - 4
        b = 4 - 2*const1
        c = const1 - nu
        d = b**2 - 4*a*c
        x1 = (-b + np.sqrt(d)) / (2*a)
        x2 = (-b - np.sqrt(d)) / (2*a)
        sols = []
        if x1 >= 0 and x1 <= 1:
            sols.append(x1)
        if x2 >= 0 and x2 <= 1:
            sols.append(x2)

        return sols


def optimal_aspect_ratio(gamma, nL, af):
    """
    gamma: weight of the residual connections
    nL: number of layers at the end of the network
    af: activation function
    """
    return (4 / (20 + 3*nL)) * 1/nu(gamma, af)


def gamma_from_ratio(ratio, nL, af):
    """
    Reverse engineer the gamma parameter, governing the residual
    connections, given the optimal aspect ratio.
    """
    c = 4 / (20 + 3*nL)
    res = gamma_from_nu(c/ratio, af)
    if len(res) == 0:
        return 0
    return res[0]


def distribution_hyperparameters(gamma, af):
    """
    gamma: weight of the residual connections
    activation: activation function
    For now, linear is f(x)=x
    """
    Cb = 0
    g1 = 1 - gamma**2
    # Scale invariant universality class
    if af in ['relu', 'linear']:
        if af == 'relu':
            A2 = 1/2
        if af == 'linear':
            A2 = 1
        CW = 1 / A2 * g1
    # K* = 0 universality class
    if af in ['tanh', 'sin']:
        CW = g1
    # Half-stable universality class
    if af == 'gelu':
        Cb = 0.17292239
        CW = 1.98305826
    if af == 'swish':
        Cb = 0.055514317
        CW = 1.98800468

    return Cb, CW


class CritMLP(nn.Module):
    stringToActivations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
        'swish': nn.Hardswish(),
        'linear': lambda x: x
    }

    def __init__(self, in_dim=None, out_dim=None, depth=None, width=None,
                 af='relu'):
        super(CritMLP, self).__init__()
        if depth is not None and width is not None:
            # TODO this should not error, just determine the residual
            # connection weight
            raise ValueError('Either depth or width must be None, CritMLP \
                will infer the other to ensure criticality')
        if out_dim is None:
            raise ValueError('out_dim must be specified')
        if in_dim is None:
            raise ValueError('in_dim must be specified')

        ratio = optimal_aspect_ratio(0, out_dim, af)
        self.gamma = gamma_from_ratio(ratio, out_dim, af)
        print("ratio: ", ratio)
        if depth is None:
            depth = int(width * ratio)
        if width is None:
            width = int(depth / ratio)

        print("depth: ", depth)
        print("width: ", width)

        self.activation = self.stringToActivations[af]

        layers = []
        layers.append(nn.Linear(in_dim, width))
        # layers.append(self.activation)
        for i in range(depth-2):
            layers.append(nn.Linear(width, width))
            # layers.append(self.activation)
        layers.append(nn.Linear(width, out_dim))

        def init_weights(m):
            Cb, CW = distribution_hyperparameters(self.gamma, af)
            print("Cb: ", Cb)
            print("CW: ", CW)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=np.sqrt(CW/width))
                nn.init.normal_(m.bias, mean=0, std=np.sqrt(Cb))
                # nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(
                # m.weight, mode='fan_out', nonlinearity='relu')
                # m.weight, mode='fan_in', nonlinearity='relu')
                pass

        self.seq = nn.Sequential(*layers)
        # self.seq.apply(init_weights)

    def forward(self, x):
        # return self.seq(x)
        # x = self.seq[0](x)
        # x = F.relu(x)
        # x = self.seq[1](x)
        # x = F.relu(x)
        # x = self.seq[-1](x)
        for i in range(len(self.seq)):
            res = x
            x = self.seq[i](x)
            if i != len(self.seq)-1:
                x = F.relu(x)
                if i != 0:
                    # x += res * self.gamma
                    x = x + res * self.gamma
                    pass
        return x


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def train(device, model, optimizer, criterion, X_train, y_train, X_test, y_test):
    num_epochs = 400
    train_losses = []
    val_accs = []
    grad_norms = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train).to(device)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        # grad_norms.append(get_grad_norm(model))

        # Evaluate the model
        with torch.no_grad():
            model.eval()
            outputs = model(X_test).to(device)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            val_accs.append(accuracy)

    return train_losses, val_accs, grad_norms


def main():
    # Test 1: iris dataset
    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # print(optimal_aspect_ratio(0, 3, 'relu'), "d/w")
    # print(distribution_hyperparameters(0, 'relu'))
    # r*  = 0.027, so for d=20, w=740

    # Test 2: wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    # print(optimal_aspect_ratio(0, 3, 'relu'), "d/w")
    # print(distribution_hyperparameters(0, 'relu'))

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y)
    print(X.shape, y.shape)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define the model hyperparameters
    input_size = X.shape[1]
    output_size = len(torch.unique(y))

    model = CritMLP(in_dim=input_size, out_dim=output_size,
                    depth=30, af='relu').to(device)
    print(model)
    opt = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    t1, v1, g1 = train(device, model, opt, criterion,
                       X_train, y_train, X_test, y_test)
    plt.plot(g1, label="m1", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.show()
    plt.plot(t1, label="m1", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(v1, label="m1", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def test():
    print(optimal_aspect_ratio(0, 3, 'relu'))
    print(gamma_from_ratio(0.027, 20, 'relu'))


if __name__ == "__main__":
    main()
    # test()
