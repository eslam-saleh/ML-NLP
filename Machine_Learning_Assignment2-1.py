import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Node:
    def __init__(self, name):
        self.name = name
        self.children = {}


def Entropy(x):
    attributes, count = np.unique(x, return_counts=True)
    return np.sum([(-count[k] / float(len(x))) * np.log2(count[k] / float(len(x))) for k in range(len(attributes))])


def InformationGain(feat, x):
    children, attributes = np.unique(feat, return_inverse=True)
    children_for_feature = 0
    for k in range(len(children)):
        children_for_feature += Entropy(x[attributes == k]) * len(x[attributes == k]) / len(x)

    return Entropy(x) - children_for_feature


def getPossibleValue(node, value):
    if not (value[node.name] in node.children):
        return
    child = node.children[value[node.name]]
    return child if type(child) != Node else getPossibleValue(child, value)


def CreateTree(row1, column1, levels):
    isIdentical, counts = np.unique(column1, return_counts=True)
    if len(isIdentical) < 2:
        return isIdentical[0], 1
    if len(levels) == 0:
        return isIdentical[np.argmax(counts)], 1
    result = []
    for lvl in levels:
        result.append(InformationGain(row1[lvl], column1))

    best_node = Node(levels[np.argmax(result)])
    updatedlvls = []
    for k in levels:
        if k != levels[np.argmax(result)]:
            updatedlvls.append(k)
    length = 0
    for intersection in np.unique(row1[levels[np.argmax(result)]]):
        subset = row1[levels[np.argmax(result)]] == intersection
        best_node.children[intersection], current_level = CreateTree(row1[subset], column1[subset], updatedlvls)
        length += current_level
    return best_node, length + 1


if __name__ == '__main__':
    url = "house-votes-84.data.txt"
    data = pd.read_csv(url)

    for ranges in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
        print("\n=================[%" + str(ranges * 100) + "]=================")
        trainAcc, testAcc, treeAcc = [], [], []
        for i in range(5):
            train = data.sample(frac=ranges)  # Random values for both sets
            test = data.drop(train.index)
            if ranges != 0.25:
                values = data.columns[1:]
                for lbl in values:
                    counter = data[lbl].value_counts()
                    if counter['y'] < counter['n']:
                        data[lbl].replace({'?': 'n'}, inplace=True)
                    else:
                        data[lbl].replace({'?': 'y'}, inplace=True)
            root, depth = CreateTree(train.iloc[:, 1:], train.iloc[:, 0], train.iloc[:, 1:].columns.tolist())
            sumTrain = 0.0
            for j in range(len(train)):
                if getPossibleValue(root, train.iloc[j, 1:]) == train.iloc[j, 0]:
                    sumTrain += 1
            sumTrain /= len(train)
            trainAcc.append(sumTrain)
            sumTest = 0.0
            for j in range(len(test)):
                if getPossibleValue(root, test.iloc[j, 1:]) == test.iloc[j, 0]:
                    sumTest += 1
            sumTest /= len(test)
            testAcc.append(sumTest)
            treeAcc.append(depth)

            if ranges == 0.25:
                print("Tree Size = " + str(depth))
                print("Training Acc of " + str(ranges) + " = " + str(trainAcc[i] * 100))
                print("Testing  Acc of " + str(1 - ranges) + " = " + str(testAcc[i] * 100))

        print("\nThe min of tree size: " + str(np.min(treeAcc)))
        print("The max of tree size: " + str(np.max(treeAcc)))
        print("The mean of tree size: " + str(np.mean(treeAcc)))
        print("The min of train Acc = " + str(np.min(trainAcc) * 100))
        print("The max of train Acc = " + str(np.max(trainAcc) * 100))
        print("The mean of train Acc = " + str(np.mean(trainAcc) * 100))
        print("The min of test Acc = " + str(np.min(testAcc) * 100))
        print("The max of test Acc = " + str(np.max(testAcc) * 100))
        print("The mean of test Acc = " + str(np.mean(testAcc) * 100))

        # plt.ylabel('Acc')
        # plt.xlabel('Iteration')
        # plt.plot(trainAcc, c='red', label='Acc')
        # plt.legend()
        # plt.show()
