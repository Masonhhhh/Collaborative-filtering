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
        # question_ids = [data["question_id"][j] for j in indices_j]
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

    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
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
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    ite = 100
    lr = 0.005
    iterations = [i for i in range(ite)]

    t, b, tnllk, vnllk, vscore = irt(train_data, val_data, lr, ite)


    plt.scatter(y=tnllk, x=iterations, label="Training neg-log-llh")
    plt.scatter(y=vnllk, x=iterations, label="Validation neg-log-llh")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Negative log-likelihood")
    plt.title("Training and Validation Negative Log-Likelihood Curves for "
              "learning rate {}".format(lr))
    plt.legend(loc="upper right")
    plt.show()

    plt.scatter(y=vscore, x=iterations)
    plt.ylabel("Validation score")
    plt.xlabel("Number of Iterations")
    plt.title("Validation Score Curve for learning rate {}".format(lr))
    plt.show()


    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    test_acc = evaluate(test_data, t, b)
    val_acc = evaluate(val_data, t, b)
    print("Final test accuracy: {}".format(test_acc))
    print("Final validation accuracy: {}".format(val_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################

    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
