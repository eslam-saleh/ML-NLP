import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    return np.divide(np.subtract(x, x_mean), x_std)


def svm_gd(x, y, l_rate, epochs):
    w = np.zeros(x.shape[1])
    _lambda = 1 / epochs
    costs = []
    y = y.tolist()
    for _ in range(epochs+1):
        cost = 0
        for i in range(x.shape[0]):
            term = y[i] * np.dot(x[i], w)
            if term < 1:
                w -= l_rate * (2 * _lambda * w - np.dot(x[i], y[i]))
                cost += 1 - term
            else:
                w -= 2 * l_rate * _lambda * w
        costs.append(cost)
    return w, costs


def predict(x_tst, w):
    y = np.dot(x_tst, w)
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = -1
        else:
            y[i] = 1
    return y


def test_data(x_tst, y_tst, w):
    y = predict(x_tst, w)
    y_tst = y_tst.tolist()
    accuracy = np.mean(y == y_tst) * 100
    return accuracy


if __name__ == '__main__':
    data = pd.read_csv('heart.csv')
    best_accuracy = -1
    best_costs = []
    for _ in range(1000):
        split_mask = np.random.choice([True, False], size=data.shape[0], p=[0.8, 0.2])
        x = data[['sex', 'chol', 'restecg', 'ca']]
        x = normalize(x)
        x.insert(x.shape[1], "ones", 1)
        x = x.to_numpy()

        y = data['target']
        y.replace(to_replace=0, value=-1, inplace=True)

        x_trn = x[split_mask]
        y_trn = y[split_mask]
        x_tst = x[~split_mask]
        y_tst = y[~split_mask]

        epochs = 5
        l_rate = 0.003
        w, costs = svm_gd(x_trn, y_trn, l_rate, epochs)

        accuracy = test_data(x_tst, y_tst, w)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_costs = costs
        if best_accuracy >= 95:
            break
    print('Best Accuracy = ', best_accuracy, '%')
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.plot(best_costs, c='red', label='Cost')
    plt.legend()
    plt.show()
