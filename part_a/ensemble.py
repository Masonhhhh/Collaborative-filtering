# DONE: complete this file.
from part_a.item_response import *


def bootstrap(data, size):
    length = len(data["user_id"])
    sample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    for i in np.random.randint(length, size=size):
        sample["user_id"].append(data["user_id"][i])
        sample["question_id"].append(data["question_id"][i])
        sample["is_correct"].append(data["is_correct"][i])

    return sample


def train(train_data, val_data, lr, num_epoch):
    print("Training IRT......")
    theta, beta, _, _, _ = irt(train_data, val_data, lr, num_epoch)
    print("Training IRT finished.")
    return theta, beta


def predict(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def evaluate(data, pred1, pred2, pred3):
    pred = []
    for i in range(len(pred1)):
        result = int(pred1[i]) + int(pred2[i]) + int(pred3[i])
        result = ((result / 3) >= 0.5)
        pred.append(result)

    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.005
    num_epoch = 25

    val_preds = []
    test_preds = []

    for _ in range(3):
        sample = bootstrap(train_data, len(train_data["user_id"]))
        theta, beta = train(sample, val_data, lr, num_epoch)

        val_pred = predict(val_data, theta, beta)
        test_pred = predict(test_data, theta, beta)

        val_preds.append(val_pred)
        test_preds.append(test_pred)

    val_acc = evaluate(val_data, val_preds[0], val_preds[1], val_preds[2])
    test_acc = evaluate(test_data, test_preds[0], test_preds[1], test_preds[2])

    print("Validation Accuracy is: {} \nTest Accuracy is: {}".format(val_acc, test_acc))


if __name__ == "__main__":
    main()
