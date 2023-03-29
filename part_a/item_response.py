import matplotlib.pyplot as plt
from utils import *
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    # for index, t in enumerate(theta):
    #     log_lklihood += \
    #         sum([(data["is_correct"][j] *
    #               np.log(sigmoid(t - beta[data["question_id"][j]])) +
    #               (1 - data["is_correct"][j]) *
    #               np.log(1 - sigmoid(t - beta[data["question_id"][j]])))
    #              if data["user_id"][j] == index else 0 for j in
    #              range(len(data["user_id"]))])
    for i in range(theta.shape[0]):
        indices_j = [j for j in range(len(data["user_id"])) if data["user_id"][j] == i]
        log_lklihood += sum([(data["is_correct"][j] * np.log(sigmoid(theta[i] - beta[data["question_id"][j]]))
                              + (1 - data["is_correct"][j]) * np.log(1 - sigmoid(theta[i] - beta[data["question_id"][j]]))) for j in indices_j])
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    grad_theta = []
    for i in range(theta.shape[0]):
        indices_j = [j for j in range(len(data["user_id"])) if data["user_id"][j] == i]
        s = sum([(data["is_correct"][j] - sigmoid(theta[i] - beta[data["question_id"][j]])) for j in indices_j])
        grad_theta.append(s)
    grad_theta = np.array(grad_theta)
    theta += lr * grad_theta

    grad_beta = []
    for j in range(beta.shape[0]):
        indices_i = [i for i in range(len(data["question_id"])) if data["question_id"][i] == j]
        s = sum([(data["is_correct"][i] - sigmoid(theta[data["user_id"][i]] - beta[j])) for i in indices_i])
        grad_beta.append(-s)
    grad_beta = np.array(grad_beta)
    beta += lr * grad_beta

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    tnllklst = []
    vnllklst = []

    for i in range(iterations):
        training_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        valid_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        tnllklst.append(training_neg_lld)
        vnllklst.append(valid_neg_lld)
        val_acc_lst.append(score)
        print("Iteration{} \t Training NLLK: {} \t Validation NLLK: {} \t "
              "Validation score: {}".format(i, training_neg_lld, valid_neg_lld,
                                            score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # You may change the return values to achieve what you want.
    return theta, beta, tnllklst, vnllklst, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    """
    Train the IRT on the data and plot the training and validation negative log-
    likelihoods, validation accuracies, and report the final test and validation
    accuracy. Return the trained parameters beta and theta.
    """
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    ite = 100
    lr = 0.005
    iterations = [i for i in range(ite)]

    t, b, tnllk, vnllk, vscore = irt(train_data, val_data, lr, ite)

    test_acc = evaluate(test_data, t, b)
    val_acc = evaluate(val_data, t, b)
    print("Final test accuracy: {}".format(test_acc))
    print("Final validation accuracy: {}".format(val_acc))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(iterations, tnllk, '-', color=(158/250, 24/250, 196/250), alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training negative log-likelihood', color=(158/250, 24/250,
                                                              196/250))
    ax1.tick_params('y', colors=(158/250, 24/250, 196/250))

    ax2.plot(iterations, vnllk, '-', color=(0/250, 139/250, 0/250), alpha=0.7)
    ax2.set_ylabel('Validation negative log-likelihood', color=(0/250, 139/250,
                                                                0/250))
    ax2.tick_params('y', colors=(0/250, 139/250, 0/250))
    ax1.set_ylim(min(tnllk), max(tnllk) + 1000)
    plt.title("Training and Validation negative log-likelihoods progression")
    plt.figtext(0.45, 0.001, "Learning rate: {}".format(lr), color="grey")
    plt.show()

    plt.scatter(y=vscore, x=iterations, color=(0/250, 139/250, 0/250))
    plt.ylabel("Validation score")
    plt.xlabel("Iterations")
    plt.title("Validation accuracy progression".format(lr))
    plt.figtext(0.45, 0.01, "Learning rate: {}".format(lr), color="grey")
    plt.show()

    return b, t


def plot_pcij(beta, theta, js):
    """
    Plot the probability of getting the given questions in <js> right against
    students and against students abilities. Use the beta and theta in we
    trained.

    @param beta: vector
    @param theta: vector
    @js: a list of questions of interest
    """
    students = [i for i in range(542)]
    x = np.linspace(-5, 5, 100)
    for ji in js:
        beta_ji = beta[ji]
        probability = np.exp(theta - beta_ji) / (1 + np.exp(theta - beta_ji))
        plt.scatter(np.array(students), probability, alpha=0.7,
                    label="{}-th question with difficulty {}".format(ji,
                                                                     round(beta_ji, 3)))
    plt.xlabel("Student number")
    plt.ylabel("Probability")
    plt.title("Probabilities of getting the given questions right against students")
    plt.legend()
    plt.show()
    for ji in js:
        beta_ji = beta[ji]
        probability = np.exp(x - beta_ji) / (1 + np.exp(x - beta_ji))
        plt.scatter(x, probability, alpha=0.7,
                    label="{}-th question with difficulty {}".format(ji,
                                                                     round(beta_ji, 3)))
    plt.xlabel("Student ability theta")
    plt.ylabel("Probability")
    plt.title("Probabilities of getting the given questions right against students ability")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    b, t = main()
    plot_pcij(b, t, [34, 855, 1773])
