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
    # Scale invariant universality class
    if af in ['relu', 'linear']:
        if af == 'relu':
            A2 = 1/2
        if af == 'linear':
            A2 = 1
        g1 = 1 - gamma**2
        CW = 1 / A2 * g1
    if af == 'gelu':
        Cb = 0.17292239
        CW = 1.98305826
    if af == 'swish':
        Cb = 0.055514317
        CW = 1.98800468

    return Cb, CW


class MLP1(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim, depth):
        super(MLP1, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        for i in range(depth-2):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(h_dim, out_dim))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=np.sqrt(2/h_dim))
                # nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

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


# This one has named layers, so you can use a variable learning rate
# "depth" is the number of TOTAL layers, including input and output
class MLP2(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, out_dim)
        layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]

        for m in layers:
            nn.init.normal_(m.weight, mean=0, std=np.sqrt(2/h_dim))
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc4(x))
        # x = F.tanh(self.fc5(x))
        x = self.fc6(x)
        return x


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

# Create the MLP model
# m1 is the theoretically best one
m1 = MLP1(input_size, 148, output_size, 4)
m2 = MLP1(input_size, 170, output_size, 4)
m3 = MLP1(input_size, 120, output_size, 4)
# m1 = MLP1(input_size, 185, output_size, 5)
# m2 = MLP1(input_size, 220, output_size, 5)
# m3 = MLP1(input_size, 140, output_size, 5)
# m1 = MLP1(input_size, 222, output_size, 6)
# m2 = MLP1(input_size, 250, output_size, 6)
# m3 = MLP1(input_size, 200, output_size, 6)
# m1 = MLP1(input_size, 259, output_size, 7)
# m2 = MLP1(input_size, 300,  output_size, 7)
# m3 = MLP1(input_size, 200, output_size, 7)
# m1 = MLP1(input_size, 370, output_size, 10)
# m2 = MLP1(input_size, 400,  output_size, 10)
# m3 = MLP1(input_size, 330, output_size, 10)
# m1 = MLP1(input_size, 185, output_size, 5)
# m2 = MLP1(input_size, 185, output_size, 4)
# m3 = MLP1(input_size, 185, output_size, 6)

# m1 = MLP2(input_size, 222, output_size)
# m2 = MLP2(input_size, 250, output_size)
# m3 = MLP2(input_size, 200, output_size)
# m1 = MLP2(input_size, 185, output_size)
# m2 = MLP2(input_size, 195, output_size)
# m3 = MLP2(input_size, 175, output_size)
# m1 = MLP2(input_size, 222, output_size)
# m2 = MLP2(input_size, 259, output_size)
# m3 = MLP2(input_size, 280, output_size)

# Define the loss function and optimizer
# o1 = optim.SGD(m1.parameters(), lr=0.001)
# o2 = optim.SGD(m2.parameters(), lr=0.001)
# o3 = optim.SGD(m3.parameters(), lr=0.001)
o1 = optim.Adam(m1.parameters(), lr=0.001)
o2 = optim.Adam(m2.parameters(), lr=0.001)
o3 = optim.Adam(m3.parameters(), lr=0.001)
# BASE_LR = 0.001
# UNK_CONST = 1.3
# PROD = BASE_LR * UNK_CONST
#
# o1 = optim.Adam(
#     [
#         {"params": m1.fc1.parameters(), "lr": PROD / 1},
#         {"params": m1.fc2.parameters(), "lr": PROD / 2},
#         {"params": m1.fc3.parameters(), "lr": PROD / 3},
#         {"params": m1.fc4.parameters(), "lr": PROD / 4},
#         {"params": m1.fc5.parameters(), "lr": PROD / 5},
#         {"params": m1.fc6.parameters(), "lr": PROD / 6},
#     ],
#     lr=0.001,
# )
#
# o2 = optim.Adam(
#     [
#         {"params": m2.fc1.parameters(), "lr": PROD / 1},
#         {"params": m2.fc2.parameters(), "lr": PROD / 2},
#         {"params": m2.fc3.parameters(), "lr": PROD / 3},
#         {"params": m2.fc4.parameters(), "lr": PROD / 4},
#         {"params": m2.fc5.parameters(), "lr": PROD / 5},
#         {"params": m2.fc6.parameters(), "lr": PROD / 6},
#     ],
#     lr=0.001,
# )
#
# o3 = optim.Adam(
#     [
#         {"params": m3.fc1.parameters(), "lr": PROD / 1},
#         {"params": m3.fc2.parameters(), "lr": PROD / 2},
#         {"params": m3.fc3.parameters(), "lr": PROD / 3},
#         {"params": m3.fc4.parameters(), "lr": PROD / 4},
#         {"params": m3.fc5.parameters(), "lr": PROD / 5},
#         {"params": m3.fc6.parameters(), "lr": PROD / 6},
#     ],
#     lr=0.001,
# )
#

t1, v1, g1 = train(m1, o1)
t2, v2, g2 = train(m2, o2)
t3, v3, g3 = train(m3, o3)
print(m1)
plt.plot(g1, label="m1", color="red")
plt.plot(g2, label="m2", color="green")
plt.plot(g3, label="m3", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.show()
plt.plot(t1, label="m1", color="red")
plt.plot(t2, label="m2", color="green")
plt.plot(t3, label="m3", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.plot(v1, label="m1", color="red")
plt.plot(v2, label="m2", color="green")
plt.plot(v3, label="m3", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
