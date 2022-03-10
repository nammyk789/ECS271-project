from queue import Full
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from base_model import BaseModel

"""
Define a fully connect neural network
"""


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class FullyConnected(nn.Module):
    def __init__(self, in_features, num_classes, activation=nn.LeakyReLU, num_layers: int = 7):
        assert num_layers % 2 == 1, "Please give an odd number of layers"
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers // 2):
            self.layers.append(nn.Sequential(
                nn.Linear(in_features * (2 ** i), in_features *
                          (2 ** (i + 1)), bias=True),
                activation()
            ))
        for i in reversed(range(num_layers // 2)):
            self.layers.append(nn.Sequential(
                nn.Linear(in_features * (2 ** (i + 1)),
                          in_features * (2 ** i), bias=True),
                activation()
            ))
        self.layers.append(nn.Linear(in_features, num_classes))
        for l in self.layers:
            l.apply(init_weights)

    def forward(self, X):
        for l in self.layers:
            X = l(X)
        return X


class TabularDataset(Dataset):
    """
    Define a simple dataset
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


"""
Define categorical hyperparamer options for the network
"""
loss_criterion_dict = {
    "nn.CrossEntropyLoss": nn.CrossEntropyLoss
}
activation_dict = {
    "nn.PReLU": nn.PReLU,
    "nn.ReLU": nn.ReLU,
    "nn.LeakyReLU": nn.LeakyReLU,
    "nn.Tanh": nn.Tanh,
    "nn.Sigmoid": nn.Sigmoid
}
optimizer_dict = {
    "optim.Adam": optim.Adam,
    "optim.SGD": optim.SGD,
    "optim.Adagrad": optim.Adagrad,
    "optim.RMSprop": optim.RMSprop
}


class NeuralNetworkHyperparam:

    def __init__(self, batch_size: int = 4, lr: float = 0.0003,
                 weight_decay: float = 1e-05, loss_criterion: str = "nn.CrossEntropyLoss",
                 num_layers: int = 7, activation: str = "nn.PReLU", optimizer: str = "optim.Adam"):
        """
        Initialize the neural network hyperparameters.
        The default parameters correspond to the results of the hyperparameter tuning.
        """
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.loss_criterion = loss_criterion_dict[loss_criterion]
        self.activation = activation_dict[activation]
        self.optimizer = optimizer_dict[optimizer]


class NeuralNetwork(BaseModel):
    """
    Define a simple interface for the network. This class initializes and trains a neural network with
    the best hyperparameters found in hyperparameter tuning.
    """

    def train(self, train_X: np.array, train_y: np.array, val_X: np.array, val_y: np.array) -> None:
        hyperparam = NeuralNetworkHyperparam()
        self.model = FullyConnected(
            in_features=train_X.shape[1],
            num_classes=len(np.unique(train_y)),
            activation=hyperparam.activation,
            num_layers=hyperparam.num_layers)
        train_dataset = TabularDataset(train_X, train_y)
        val_dataset = TabularDataset(val_X, val_y)
        self.losses, self.train_accs, self.val_accs = train(
            model=self.model,
            trainset=train_dataset,
            testset=val_dataset,
            loss_criterion=hyperparam.loss_criterion,
            optimizer=hyperparam.optimizer,
            lr=hyperparam.lr,
            weight_decay=hyperparam.weight_decay,
            batch_size=hyperparam.batch_size)

    def predict(self, test_X: np.array) -> np.array:
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(test_X))
            return torch.softmax(outputs, dim=1).numpy()


def calculate_accuracy(dataloader: DataLoader, model: nn.Module):
    """
    Helper function to calculate accuracy
    """
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    return accuracy


def train(model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          loss_criterion=nn.CrossEntropyLoss,
          optimizer=optim.Adam,
          lr: float = 0.001,
          weight_decay: float = 0.001,
          batch_size: int = 4,
          num_epochs: int = 50,
          verbose: bool = False):
    """
    Helper function implementing a complete training loop including validation
    """
    torch.manual_seed(0)
    patience = 0
    max_test_acc = -1.0
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_criterion = loss_criterion()
    losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 4 == 3:    # print every 4 mini-batches
                average_loss = running_loss / 4
                if verbose:
                    print(f'[{epoch + 1}, {i + 1}] loss: {average_loss:.3f}')
                losses.append(average_loss)
                running_loss = 0.0

        # End of epoch, calculate accuracies
        train_accuracy = calculate_accuracy(trainloader, model)
        train_accs.append(train_accuracy)
        if verbose:
            print(
                f'[Epoch {epoch + 1}] Train accuracy:  {train_accuracy:.2f}%')

        test_accuracy = calculate_accuracy(testloader, model)
        test_accs.append(test_accuracy)
        if verbose:
            print(f'[Epoch {epoch + 1}] Test accuracy:  {test_accuracy:.2f}%')

        if test_accuracy > max_test_acc:
            max_test_acc = test_accuracy
            patience = 0
        else:
            patience += 1

        if patience >= 15:
            if verbose:
                print('Patience reached, no improvements for 15 epochs. Early stopping.')
            return losses, train_accs, test_accs
    if verbose:
        print('Finished Training')
    return losses, train_accs, test_accs


def plot(labels, loss_lists, train_acc_lists, test_acc_lists):
    """
    Helper function to plot training results
    """
    for i, loss_list in enumerate(loss_lists):
        plt.plot(loss_list, label=f"{labels[i]}")
    plt.title("Loss Curve")
    leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    leg.get_frame().set_alpha(0.5)
    plt.show()

    for i, train_acc_list in enumerate(train_acc_lists):
        plt.plot(train_acc_list, label=f"{labels[i]}")
    plt.title("Training Set Accuracies")
    leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    leg.get_frame().set_alpha(0.5)
    plt.show()

    for i, test_acc_list in enumerate(test_acc_lists):
        plt.plot(test_acc_list, label=f"{labels[i]}")
    plt.title("Test Set Accuracies")
    leg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    leg.get_frame().set_alpha(0.5)
    plt.show()
