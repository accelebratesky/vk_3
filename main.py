import matplotlib.pyplot as plt
from dataset import Dataset
from model import LinearRegression
import numpy as np

if __name__ == "__main__":
    train_set = Dataset("./data/train.csv")
    test_set = Dataset("./data/test.csv")
    x_train, y_train = train_set.get_features()
    x_test, y_test = test_set.get_features()

    model = LinearRegression(lr=0.01)
    model.train(x_train, y_train, 2000)

    y_test_pred = model.predict(x_test)
    test_mse = np.sum((y_test_pred - y_test)**2) / x_test.shape[0]
    print("\nTest MSE loss:", test_mse)

    plt.subplot(1, 2, 1)
    plt.scatter(model.epoch_record, model.loss_history, s=20)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(bottom=0, top=200)
    # plt.yscale("log")
    plt.title("Linear Regression")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(x_train, y_train, c="blue", label="train data", s=30)
    plt.scatter(x_test, y_test, c="orange", label="test data", s=30)
    x_line = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, c="red", lw=2, label="fit line")
    plt.xlabel("hours")
    plt.ylabel("score")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()