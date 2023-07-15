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
    # TODO
    if af == 'gelu':
        return nu(gamma, 'relu')


def optimal_aspect_ratio(gamma, nL, af):
    """
    gamma: weight of the residual connections
    nL: number of layers at the end of the network
    af: activation function
    """
    return (4 / (20 + 3*nL)) * 1/nu(gamma, af)


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
            raise ValueError('Either depth or width must be None, CritMLP \
                will infer the other to ensure criticality')
        if out_dim is None:
            raise ValueError('out_dim must be specified')
        if in_dim is None:
            raise ValueError('in_dim must be specified')

        ratio = optimal_aspect_ratio(0, out_dim, af)
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
        for i in range(depth-2):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, out_dim))

        def init_weights(m):
            Cb, CW = distribution_hyperparameters(0, af)
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
        self.seq.apply(init_weights)

    def forward(self, x):
        return self.seq(x)


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def train(model, optimizer):
    num_epochs = 400
    train_losses = []
    val_accs = []
    grad_norms = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        grad_norms.append(get_grad_norm(model))

        # Evaluate the model
        with torch.no_grad():
            model.eval()
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            val_accs.append(accuracy)

    return train_losses, val_accs, grad_norms


criterion = nn.CrossEntropyLoss()


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

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the model hyperparameters
input_size = X.shape[1]
output_size = len(torch.unique(y))

model = CritMLP(in_dim=input_size, out_dim=output_size, depth=20, af='gelu')
print(model)
opt = optim.Adam(model.parameters(), lr=0.0001)

t1, v1, g1 = train(model, opt)
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
