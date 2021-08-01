from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

corpusnegative = []
corpuspositive = []
negatives = []
positives = []


def tf_idf(txt):
    all_pos_and_neg = []
    labels = []

    for c in negatives:
        labels.append(c)
    for c in positives:
        labels.append(c)

    for c in corpusnegative:
        all_pos_and_neg.append(c)
    for c in corpuspositive:
        all_pos_and_neg.append(c)

    vectorizer = TfidfVectorizer()
    x_train, x_test, curr_y_train, curr_y_test = train_test_split(all_pos_and_neg, labels, test_size=0.3,
                                                                  random_state=40)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, curr_y_train)
    result = model.predict(x_test)
    acc = accuracy_score(curr_y_test, result)
    print()
    print("TF-IDF accuracy is : %f" % (acc * 100))
    print()
    model.predict(vectorizer.transform([txt]))


def ignore_letters(txt):
    tokens = []
    for sent in nltk.sent_tokenize(txt):
        for word in nltk.word_tokenize(sent):
            if len(word) <= 2:
                continue
            tokens.append(word.lower())
    return tokens


def get_vectors(model, tagged_docs):
    sentences = tagged_docs
    Ys, Xs = zip(*[(doc.tags, model.infer_vector(doc.words, steps=20)) for doc in sentences])
    return Ys, Xs


def get_avg(curr):
    avg = []
    for c in curr:
        avg.append((c - min(curr)) / (max(curr) - min(curr)))
    return avg


def format_lines(txt):
    txt = re.sub(r'\s+[a-zA-Z]\s+', ' ', txt)
    txt = re.sub(r'\^[a-zA-Z]\s+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt, flags=re.I)
    txt = re.sub(r'^b\s+', '', txt)
    txt = txt.lower()
    return txt


if __name__ == '__main__':
    data = []
    text = input("Enter Text: ")
    text = text.lower()
    text = format_lines(text)
    data.append(text)

    postivetexts = glob.glob("txt_sentoken/pos/*.txt")
    negativetexts = glob.glob("txt_sentoken/neg/*.txt")
    newnegtive = []
    newpostive = []
    step = []
    targets = []

    for file in negativetexts:
        f = open(file, "r")
        lines = f.read()
        corpusnegative.append(lines)
        negatives.append(0)
        lines = format_lines(lines)
        newnegtive.append(lines)

    for file in postivetexts:
        f = open(file, "r")
        lines = f.read()
        corpuspositive.append(lines)
        positives.append(1)
        lines = format_lines(lines)
        newpostive.append(lines)

    for i in newnegtive:
        step.append(i)
        targets.append("neg")
    for i in newpostive:
        step.append(i)
        targets.append("pos")

    train_set, test_set, goal_train, goal_test = train_test_split(step, targets, train_size=0.7, random_state=1)
    taggedtrain = []
    taggedtest = []

    c1 = 0
    c2 = 0
    while c1 < len(train_set):
        taggedtrain.append(TaggedDocument(ignore_letters(train_set[c1]), goal_train[c1]))
        c1 = c1 + 1
    while c2 < len(test_set):
        taggedtest.append(TaggedDocument(ignore_letters(test_set[c2]), goal_test[c2]))
        c2 = c2 + 1

    new_model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=10, alpha=0.025, min_alpha=0.025)
    new_model.build_vocab(taggedtrain)
    for epoch in range(20):
        new_model.train(utils.shuffle(taggedtrain), total_examples=len(taggedtrain), epochs=1)
        new_model.alpha = new_model.alpha - 0.002
        new_model.min_alpha = new_model.alpha

    y_train, Xtrain = get_vectors(new_model, taggedtrain)
    X_train = []
    for i in Xtrain:
        X_train.append(get_avg(i))
    y_test, Xtest = get_vectors(new_model, taggedtest)
    X_test = []
    for i in Xtest:
        X_test.append(get_avg(i))

    logistic = LogisticRegression(C=1.0, fit_intercept=True, intercept_scaling=1, max_iter=random.randint(100, 1000),
                                  random_state=0, tol=random.uniform(0000.1, 0.01), verbose=0)
    logistic.fit(X_train, y_train)
    pred = logistic.predict(X_test)

    guessed = 0
    for c1, c2 in zip(y_test, pred):
        if c1 == c2:
            guessed += 1
    logistic_accuracy = float(guessed) / len(y_test)

    svm = SVC(C=1.0, max_iter=random.randint(100, 1000), kernel="linear", tol=random.uniform(0000.1, 0.01))
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)

    guessed = 0
    for c1, c2 in zip(y_test, pred):
        if c1 == c2:
            guessed += 1
    svm_accuracy = float(guessed) / len(y_test)

    print()
    print("Logistic Regression algorithm accuracy is : %f" % (logistic_accuracy * 100))
    print()

    print()
    print("SVM algorithm accuracy is : %f" % (svm_accuracy * 100))
    print()

    tf_idf(text)

    str = ""
    for i in data:
        str += i
    Y = ignore_letters(str)
    X = new_model.infer_vector(Y, steps=20)
    prediction = logistic.predict([list(get_avg(X))])
    print()
    print("Result is ", prediction)
