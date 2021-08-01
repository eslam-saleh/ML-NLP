'''
Eslam Saleh 20170046
Abd-Elrahman Ahmed 20170140
CS-IS-1
'''

import random
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


def create_model():
    rand1 = random.randint(1, 3)
    new_model = Sequential()
    for i in range(1, rand1):
        new_model.add(Conv2D(random.choice([32, 64, 128]), (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        rand2 = random.randint(2, 5)
        new_model.add(MaxPooling2D((rand2, rand2)))
    new_model.add(Flatten())
    new_model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    new_model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model


if __name__ == '__main__':
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    trainX = (trainX.astype('float32')) / 255.0
    testX = (testX.astype('float32')) / 255.0
    for j in range(1, 5):
        print('\nModel ' + str(j) + ' :')
        kfold = KFold(5, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(trainX):
            model = create_model()
            tmptrainX, tmptrainY, tmptestX, tmptestY = trainX[train_ix], trainY[train_ix], trainX[test_ix], trainY[
                test_ix]
            _, acc = model.evaluate(tmptestX, tmptestY, verbose=0)
            print('> %.3f' % (acc * 100.0))
