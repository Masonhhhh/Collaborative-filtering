from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("User-based Validation Accuracy for k={}: {}".format(k, acc))
    return acc


# The assumption of item-based collaborative filtering is that if question A
# was answered correctly and incorrectly by the same students as question B,
# question A's correctness by a specific student would be the same as that of
# question B
def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    matrix_transpose = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix_transpose).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item-based Validation Accuracy for k={}: {}".format(k, acc))
    return acc


def item_main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    indices = [1, 6, 11, 16, 21, 26]
    accs = [knn_impute_by_item(sparse_matrix, val_data, k)
            for k in indices]

    plt.scatter(x=indices, y=accs)
    plt.xlabel("k")
    plt.ylabel("accuracies")
    plt.title("Item-based KNN validation accuracies for different k's")

    k_star = indices[accs.index(np.max(np.array(accs)))]
    print("k* for highest validation accuracies: {}".format(k_star))
    matrix_star = KNNImputer(n_neighbors=k_star).fit_transform(sparse_matrix.T)
    print("Test accuracy: {}".format(sparse_matrix_evaluate(test_data,
                                                            matrix_star.T)))

    plt.show()


def user_main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print("Sparse matrix:")
    # print(sparse_matrix)
    # print("Shape of sparse matrix:")
    # print(sparse_matrix.shape)

    indices = [1, 6, 11, 16, 21, 26]
    accs = [knn_impute_by_user(sparse_matrix, val_data, k)
            for k in indices]

    plt.scatter(x=indices, y=accs)
    plt.xlabel("k")
    plt.ylabel("accuracies")
    plt.title("User-based KNN validation accuracies for different k's")

    k_star = indices[accs.index(np.max(np.array(accs)))]
    print("k* for highest validation accuracies: {}".format(k_star))
    matrix_star = KNNImputer(n_neighbors=k_star).fit_transform(sparse_matrix)
    print("Test accuracy: {}".format(sparse_matrix_evaluate(test_data,
                                                            matrix_star)))

    plt.show()

    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def compare():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    indices = [i for i in range(1, 51, 5)]

    accs_item = [knn_impute_by_item(sparse_matrix, val_data, k)
                 for k in indices]
    plt.scatter(x=indices, y=accs_item, label="Item-based")

    accs_user = [knn_impute_by_user(sparse_matrix, val_data, k)
                 for k in indices]
    plt.scatter(x=indices, y=accs_user, label="User-based")

    plt.xlabel("k")
    plt.ylabel("accuracies")
    plt.title("User- and Item-based KNN validation accuracies for different k's")
    plt.legend(loc="lower right")

    max_index = accs_user.index(np.max(np.array(accs_user)))
    plt.annotate(f'Max: ({str(max_index)}, {round(max(accs_user),3)})',
                 xy=(indices[max_index], accs_user[max_index]), xytext=(0, 15),
                 textcoords='offset points', ha='center', va='top')
    max_index_1 = accs_item.index(np.max(np.array(accs_item)))
    plt.annotate(f'Max: ({str(max_index_1)}, {round(max(accs_item),3)})',
                 xy=(indices[max_index_1], accs_item[max_index_1]), xytext=(0, 15),
                 textcoords='offset points', ha='center', va='top')

    plt.show()


def main():
    user_main()


# The disadvantages of KNN in this scenario are:
# The assumptions of KNN in this task might not be necessarily true in general:
# every student differs in their skill sets and abilities to perform on various
# kinds of questions, and every question have attributes that account for
# different underlying thought processes.
# KNN algorithm is slow in its prediction stage, and it would be potentially
# even slower had the data been even larger.
if __name__ == "__main__":
    main()
