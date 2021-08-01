import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cost_function(thetas, xs, ys):
    predictor = 1 / (1 + np.exp(-np.dot(xs, thetas)))  # hypothesis
    j = ys * np.log(predictor) + (1 - ys) * np.log(1 - predictor)
    return np.sum(j) / (-len(ys))


def gradient_descent(xs, ys, iteration, ratio):
    thetas = np.zeros(xs.shape[1])
    cost = [cost_function(thetas, xs, ys)]
    for i in range(iteration):
        h = 1 / (1 + np.exp(-np.dot(xs, thetas)))
        thetas -= ratio * np.dot(xs.T, (h - ys)) / ys.shape[0]
        cost.append(cost_function(thetas, xs, ys))  # calc cost in every iteration
    return thetas, cost


if __name__ == "__main__":
    url = 'heart.csv'
    data = pd.read_csv(url)
    x = data[['trestbps', 'chol', 'thalach', 'oldpeak']]
    y = data['target']
    ratio = 0.1  # we used 0.1 as the learning rate so change this to see diff.
    iterations = 200

    mu = np.mean(x, axis=0)  # normalization
    sigma = np.std(x, axis=0)
    x = np.divide(np.subtract(x, mu), sigma)
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    theta, cost = gradient_descent(x, y, iterations, ratio)

    plt.figure(figsize=(12, 8))
    plt.ylabel("Cost Function")
    plt.xlabel('Iteration')
    plt.scatter(range(0, len(cost)), cost)
    plt.show()

    h = 1 / (1 + np.exp(-np.dot(x, theta)))
    for i in range(len(h)):
        h[i] = 1 if h[i] >= 0.5 else 0
    acc = np.sum([list(y)[i] == h[i] for i in range(len(list(y)))]) / len(list(y))

    print("Accuracy: ", acc * 100)
