import tensorflow as tf
import json
import numpy as np
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
def train(file):
    load_dict=[]
    with open(file,'r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict[0]))

    xtrainsent = []
    ytrainsent = []
    # sentences = [["i","love","haha","tree"],["i","love","tree","haha"],["love","flawer","hoho"],["you","love","flawer","hoho"]]
    for i in range(len(load_dict[0])):
    # for i in range(5):
    #     print(load_dict[0][str(i)])
        xtrainsent.append(load_dict[0][str(i)][0:5])
        ytrainsent.append(load_dict[0][str(i)][5:10])

    xtrainsent= np.array(xtrainsent)
    ytrainsent= np.array(ytrainsent)
    xtrainsent=xtrainsent.reshape((xtrainsent.shape[0], 1,xtrainsent.shape[1]))
    # ytrainsent=ytrainsent.reshape((ytrainsent.shape[0], 1,ytrainsent.shape[1]))
    # ytrainsent.reshape((ytrainsent.shape[0], 1, ytrainsent.shape[1]))
    print(xtrainsent.shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(xtrainsent.shape[1], xtrainsent.shape[2])))
    model.add(tf.keras.layers.Dense(ytrainsent.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrainsent, ytrainsent, epochs=10, batch_size=3, verbose=2)


train("data/data1.json")