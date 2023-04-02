import matplotlib.pyplot as plt

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


def load_data2(base_path="../data"):
    """ Load the data in PyTorch Tensor. Change the original 0s into -1s.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()

    train_matrix[train_matrix == 0] = -1
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    # print(valid_data)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, layers, p=0.):
        """ Initialize a class AutoEncoder.
        """
        super(AutoEncoder, self).__init__()
        self.dropout = nn.Dropout(p)
        # Define linear functions.
        self.hidden = nn.ModuleList()
        for i, o in zip(layers[:-1], layers[1:]):
            linear = nn.Linear(i, o)
            self.hidden.append(linear)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.
        :return: float
        """
        norm = 0
        for layer in self.hidden:
            norm += torch.norm(layer.weight, 2) ** 2
        return norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        out = inputs
        for layer in self.hidden:
            out = self.dropout(layer(out))
            out = torch.sigmoid(out)
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, criterion):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    training_loss = []
    validation_accuracies = []
    for epoch in range(0, num_epoch):
        train_loss = 0.
        model.train()

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            # Mask the target to only compute the CLE of valid entries
            valid_mask = ~np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            cross_ = F.binary_cross_entropy(output[0][valid_mask],
                                        target[0][valid_mask], reduction="sum")
            cross = -torch.sum(target[0][valid_mask] *
                               torch.log2(output[0][valid_mask]) +
                               (1 - target[0][valid_mask]) *
                               torch.log2(1 - output[0][valid_mask]))

            mse = torch.sum((output - target) ** 2.)

            if criterion == 'mse':
                loss1 = mse
            elif criterion == 'cross':
                loss1 = cross
            else:
                loss1 = 0
            loss = loss1 + lamb * model.\
                get_weight_norm() / 2
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        training_loss.append(train_loss)
        validation_accuracies.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return training_loss, validation_accuracies


def train2(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,
           criterion):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    training_loss = []
    validation_accuracies = []
    for epoch in range(0, num_epoch):
        model.train()
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            # print(inputs)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)
            # print(output)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            # Mask the target to only compute the CLE of valid entries
            valid_mask = ~np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            # Linear transform back to the original scale, need for loss
            # calculation
            target[0][valid_mask] = (target[0][valid_mask] + 1) / 2

            # Different loss functions
            mse = torch.sum((output - target) ** 2.)
            cross_ = F.binary_cross_entropy(output[0][valid_mask],
                                            target[0][valid_mask], reduction="sum")
            cross = -torch.sum(target[0][valid_mask] *
                               torch.log2(output[0][valid_mask]) +
                               (1 - target[0][valid_mask]) *
                               torch.log2(1 - output[0][valid_mask]))
            if criterion == 'mse':
                loss1 = mse
            elif criterion == 'cross':
                loss1 = cross
            else:
                loss1 = 0
            loss = loss1 + lamb * model.get_weight_norm() / 2
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        valid_acc = evaluate(model, zero_train_data, valid_data)
        training_loss.append(train_loss)
        validation_accuracies.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return training_loss, validation_accuracies


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    """
    Train a variety of models with different learning rates and different number
    of neurons in the hidden layer. Plot the validation accuracies in plots.
    """
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    for i in [0.1, 0.01, 0.001]:
        for k in [10, 50, 100, 200, 500]:
            # Set model hyperparameters.
            model = AutoEncoder(1774, k)
            # Set optimization hyperparameters.
            lr = i
            num_epoch = 200
            lamb = 0

            _, validation_accuracies = train(model, lr, lamb, train_matrix,
                                             zero_train_matrix, valid_data,
                                             num_epoch)
            indices = [i for i in range(num_epoch)]
            plt.scatter(indices, validation_accuracies, label="k={}".format(k),
                        alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy")
        plt.title("Validation accuracy progression with lr {} for different "
                  "number of neurons".format(lr))
        plt.legend(loc="lower right")
        plt.show()


def plot_and_report(layers, dropout, momentum, lamb, lr, num_epoch):
    """
    Plot the training loss curve and the validation curve for a fully specified
    model in a duo axis plot. Report the final testing accuracy of the model.
    """
    zero_train_matrix, train_matrix, valid_data, test_data = load_data2()
    indices = [i for i in range(num_epoch)]
    model = AutoEncoder(layers, dropout)
    training_loss, validation_accuracies = \
        train2(model, lr, lamb, train_matrix, zero_train_matrix, valid_data,
              num_epoch, 'mse')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(indices, validation_accuracies, '-', color=(139/255, 0, 0))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation accuracy', color=(139/255, 0, 0))
    ax1.tick_params('y', colors=(139/255, 0, 0))

    ax2.plot(indices, training_loss, '-', color=(0, 0, 139/255))
    ax2.set_ylabel('Training loss', color=(0, 0, 139/255))
    ax2.tick_params('y', colors=(0, 0, 139/255))

    plt.title('Training loss and validation accuracy as training progresses')
    plt.figtext(0.5, 0., "Model hyperparams: Hidden layers {}"
                         ", dropout {}; "
                         "Optimization hyperparams: lr {}, lamb {}, momentum {}; \n"
                         "Criterion: MSE; Activation: tanh; "
                         "Weight initialization: Xavier".
                format(layers[1:-1], dropout, lr, lamb, momentum), ha="center",
                color='grey')
    print("Test accuracy: {}".format(evaluate(model, zero_train_matrix,
                                              test_data)))
    plt.show()


def tune_dropout(layers, dropout):
    """
    Plot the training loss curve and the validation curve for a fully specified
    model in a duo axis plot. Report the final testing accuracy of the model.
    """
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    lr = 0.01
    lamb = 0.
    num_epoch = 200
    indices = [i for i in range(num_epoch)]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for p in dropout:
        model = AutoEncoder(layers, p)
        training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
                                                     zero_train_matrix, valid_data,
                                                     num_epoch, 'mse')

        ax1.plot(indices, validation_accuracies, '-', label="dropout={}".format(p), alpha=0.7)
        ax2.plot(indices, training_loss, '-', label="dropout={}".format(p), alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation accuracy')
    ax1.tick_params('y')
    ax1.legend(loc="lower right")

    ax2.set_ylabel('Training loss')
    ax2.tick_params('y')

    plt.title('Training loss and validation accuracy as training progresses for different dropouts')
    plt.figtext(0.5, 0.01, "Net: {}, "
                "Optimization hyperparamters: lr {}, lamb {}".
                format(layers, lr, lamb), ha="center", color='grey')
    # print("Test accuracy: {}".format(evaluate(model, zero_train_matrix,
    #                                           test_data)))
    plt.show()


def comparison():
    layers = [1774, 60, 50, 1774]
    lr = 0.01
    lamb = 0.
    num_epoch = 200
    indices = [i for i in range(num_epoch)]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # p = 0.
    # model = AutoEncoder(layers, p)
    # training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
    #                                              zero_train_matrix, valid_data,
    #                                              num_epoch)
    #
    # ax1.plot(indices, validation_accuracies, '-', label="dropout={}".format(p), alpha=0.7)
    # ax2.plot(indices, training_loss, '-', label="dropout={}".format(p), alpha=0.7)

    p = 0.3
    model = AutoEncoder(layers, p)
    training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
                                                 zero_train_matrix, valid_data,
                                                 num_epoch, 'mse')
    ax1.plot(indices, validation_accuracies, 'x', label="dropout={}".format(p), alpha=0.7, color=(139/250, 0, 0))
    ax2.plot(indices, training_loss, '-', label="dropout={}".format(p), alpha=0.7, color=(139/250, 0, 0))

    p = 0
    model = AutoEncoder([1774, 50, 1774], p)
    training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
                                                 zero_train_matrix, valid_data,
                                                 num_epoch, 'mse')
    ax1.plot(indices, validation_accuracies, 'x', label="baseline model", alpha=0.7, color=(0, 0, 139/250))
    ax2.plot(indices, training_loss, '-', label="baseline model", alpha=0.7, color=(0, 0, 139/250))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('xxx Validation accuracy xxx')
    ax1.tick_params('y')
    ax1.legend(loc="lower right")
    ax2.set_ylabel('--- Training loss ---')
    ax2.tick_params('y')

    plt.title('Training loss and validation accuracy comparison for dropout vs baseline model')
    plt.figtext(0.5, 0.01, "Net: {}, Optimization hyperparamters: lr {}, lamb {}".
                format(layers, lr, lamb), ha="center", color='grey')
    plt.show()


def compare_criteria():
    layers = [1774, 50, 50, 1774]
    lr = 0.01
    lamb = 0.
    num_epoch = 200
    indices = [i for i in range(num_epoch)]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    p = 0.3

    model = AutoEncoder(layers, p)
    training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
                                                 zero_train_matrix, valid_data,
                                                 num_epoch, 'mse')
    ax1.plot(indices, validation_accuracies, 'x', label="{}".format('mse'), alpha=0.7, color=(139/250, 0, 0))
    ax2.plot(indices, training_loss, '-', label="{}".format('mse'), alpha=0.7, color=(139/250, 0, 0))

    model = AutoEncoder(layers, p)
    training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
                                                 zero_train_matrix, valid_data,
                                                 num_epoch, 'cross')
    ax1.plot(indices, validation_accuracies, 'x', label="cross", alpha=0.7, color=(0, 0, 139/250))
    ax2.plot(indices, training_loss, '-', label="cross", alpha=0.7, color=(0, 0, 139/250))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('xxx Validation accuracy xxx')
    ax1.tick_params('y')
    ax1.legend(loc="lower right")
    ax2.set_ylabel('--- Training loss ---')
    ax2.tick_params('y')

    plt.title('Training loss and validation accuracy comparison for MSE and Cross-entropy loss criteria')
    plt.figtext(0.5, 0.01, "Net: {} dropout: {}, Optimization hyperparamters: lr {}, lamb {}".
                format(layers, p, lr, lamb), ha="center", color='grey')
    plt.show()


def compare_data_structure():
    layers = [1774, 60, 50, 1774]
    lr = 0.01
    lamb = 0.
    num_epoch = 200
    indices = [i for i in range(num_epoch)]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    p = 0.3
    model = AutoEncoder(layers, p)
    training_loss, validation_accuracies = train(model, lr, lamb, train_matrix,
                                                 zero_train_matrix, valid_data,
                                                 num_epoch, 'mse')
    ax1.plot(indices, validation_accuracies, 'x', label="{}".format('Original sata structure'), alpha=0.7, color=(139/250, 0, 0))
    ax2.plot(indices, training_loss, '-', label="{}".format('Original data structure'), alpha=0.7, color=(139/250, 0, 0))

    zero_train_matrix, train_matrix, valid_data, test_data = load_data2()
    model = AutoEncoder(layers, p)
    training_loss, validation_accuracies = train2(model, lr, lamb, train_matrix,
                                                  zero_train_matrix, valid_data,
                                                  num_epoch, "mse")
    ax1.plot(indices, validation_accuracies, 'x', label="Alternative data structure", alpha=0.7, color=(0, 0, 139/250))
    ax2.plot(indices, training_loss, '-', label="Alternative data structure", alpha=0.7, color=(0, 0, 139/250))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('xxx Validation accuracy xxx')
    ax1.tick_params('y')
    ax1.legend(loc="lower right")
    ax2.set_ylabel('--- Training loss ---')
    ax2.tick_params('y')

    plt.title('Training loss and validation accuracy comparison for Alternative data structure')
    plt.figtext(0.5, 0.01, "Net: {} dropout: {}, Optimization hyperparamters: lr {}, lamb {}".
                format(layers, p, lr, lamb), ha="center", color='grey')
    plt.show()


if __name__ == "__main__":
    # main()
    # tune_dropout([1774, 60, 50, 1774], [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    # tune_lamb(50, 0.01, 50)
    # compare_criteria()
    # compare_data_structure()
    plot_and_report([1774, 50, 50, 1774], 0.3, 0., 0., 0.1, 100)
