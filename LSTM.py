import tensorflow as tf
import json
import numpy as np
from gensim import corpora

def nntrain(x_train, y_train, x_test, y_test):
    # design network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, nb_epoch=500, batch_size=1, verbose=2)

    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    # model.add(tf.keras.layers.Dense(1))
    # model.compile(loss='mae', optimizer='adam')
    # # fit network
    # history = model.fit(x_train, y_train, epochs=50, batch_size=72, validation_data=(x_test, y_test), verbose=2,
    #                     shuffle=False)
def getdata(file):
    xtrainsent = []
    ytrainsent = []
    for i in range(1,3):
        file='data/data'+str(i)+'.json'
        with open(file,'r') as load_f:
            load_dict = json.load(load_f)
        print(len(load_dict[0]))
        for i in range(len(load_dict[0])):
            xtrainsent.append(load_dict[0][str(i)][0:8])
            ytrainsent.append(load_dict[0][str(i)][8:10])

    dictionary = corpora.Dictionary(xtrainsent)
    corpus = [dictionary.doc2bow(text) for text in xtrainsent]
    print(corpus[1])  # [(0, 1), (1, 1), (2, 1)]
    print(corpus[1])  # [(0, 1), (1, 1), (2, 1)]

    xtrainsent= np.array(xtrainsent)
    ytrainsent= np.array(ytrainsent)
    xtrainsent=xtrainsent.reshape((xtrainsent.shape[0], 1,xtrainsent.shape[1]))
    return xtrainsent,ytrainsent

def train(xtrainsent,ytrainsent):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(xtrainsent.shape[1], xtrainsent.shape[2])))
    model.add(tf.keras.layers.Dense(ytrainsent.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrainsent, ytrainsent, epochs=10, batch_size=32, verbose=2)
    # model.save('lstm.model')
    testx=[["8336", "4522", "11338", "6316", "2850", "12419", "8048", "11479"]]
    testx = np.array(testx)
    testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))

    testPredict = model.predict(testx)
    # testPredict = scaler.inverse_transform(testPredict)
    print(testPredict)
    return model


x,y =getdata("data/data1.json")
model=train(x,y)