import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def plotting(tfidf, lbls):
    tfs_reduced = TruncatedSVD(n_components=2, random_state=7).fit_transform(tfidf)
    tfs_embedded = TSNE(n_components=2).fit_transform(tfs_reduced)
    plt.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], c=ListedColormap(('red', 'blue'))(lbls))
    plt.title('Performing text classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def addUserInput():
    text = input("Enter Text: ")
    data = model.predict(vectorizer.transform([text]))
    print(data)
    plotting(x_train, y_train)


if __name__ == '__main__':
    corpusnegative = []
    corpuspositive = []
    negatives = []
    positives = []
    all_pos_and_neg = []
    labels = []

    for name in os.listdir('neg'):
        filename = os.path.join('neg', name)
        currfile = open(filename, "r").read()
        corpusnegative.append(currfile)
        negatives.append(0)

    for name in os.listdir('pos'):
        filename = os.path.join('pos', name)
        currfile = open(filename, "r").read()
        corpuspositive.append(currfile)
        positives.append(1)

    for i in negatives:
        labels.append(i)
    for i in positives:
        labels.append(i)

    for i in corpusnegative:
        all_pos_and_neg.append(i)
    for i in corpuspositive:
        all_pos_and_neg.append(i)

    vectorizer = TfidfVectorizer()
    x_train, x_test, y_train, y_test = train_test_split(all_pos_and_neg, labels, test_size=0.3, random_state=40)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    # print(result)
    acc = accuracy_score(y_test, result)
    print("Logistic Regression accuracy is : %f" % (acc * 100))
    addUserInput()
