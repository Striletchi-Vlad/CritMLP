import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from crit_mlp import CritMLP


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def train(device, model, optimizer, criterion,
          X_train, y_train, X_test, y_test):
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
        grad_norms.append(get_grad_norm(model))

        # Evaluate the model
        with torch.no_grad():
            model.eval()
            outputs = model(X_test).to(device)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            val_accs.append(accuracy)

    return train_losses, val_accs, grad_norms


def main():
    # Load wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44)

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                    depth=20, af='relu').to(device)
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


if __name__ == "__main__":
    main()
