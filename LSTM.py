import tensorflow as tf
import json
import numpy as np
from gensim.models.word2vec import Word2Vec

def onehot(num):
    list=[0]*20000
    list[num]=1
    return list

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
        map(int, load_dict)
        for i in range(len(load_dict[0])):
            # xtrainsent.append(load_dict[0][str(i)][0:8])
            # ytrainsent.append(load_dict[0][str(i)][8:10])
            xlist = load_dict[0][str(i)][0:9]
            Xlist = [int(i) for i in xlist]
            xtrainsent.append(Xlist)
            ylist = load_dict[0][str(i)][9:10]
            Ylist = [int(i) for i in ylist]
            ytrainsent.append(Ylist)

    ytrainsent2=[]
    for y in ytrainsent:
        # print(onehot(y[0]))
        ytrainsent2.append(onehot(y[0]))

    # print("x+y:")
    # print(xtrainsent)
    # print(ytrainsent2)

    xtrainsent= np.array(xtrainsent)
    ytrainsent2= np.array(ytrainsent2)
    xtrainsent=xtrainsent.reshape((xtrainsent.shape[0], 1,xtrainsent.shape[1]))
    return xtrainsent,ytrainsent2

def train(xtrainsent,ytrainsent):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(xtrainsent.shape[1], xtrainsent.shape[2])))
    model.add(tf.keras.layers.Dense(ytrainsent.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrainsent, ytrainsent, epochs=10, batch_size=32, verbose=2)
    # model.save('lstm.model')

    # next_index = sample(preds=preds, temperatue=temperature)
    # next_char = char[next_index]
    return model

def test(model):
    testx = [["10734", "4320", "7556", "10694", "9153", "1259", "15893", "8780", "4436"]]
    testx = np.array(testx)
    testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))

    # testPredict = model.predict(testx)
    preds = model.predict(testx, verbose=0)[0]
    print(preds)
    print(max(preds))
    print(np.argmax(preds))

x,y =getdata("data/data1.json")
#

model=train(x,y)
# model.save('lstm.model')
test(model)
# model = Word2Vec.load('lstm.model')

