import time
import math
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.linalg import qr

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, BatchNormalization, merge, LocallyConnected1D, Flatten, Conv1D
from keras import backend as K
from keras import regularizers
from keras.objectives import mse
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV


dataDir = '/media/yanglu/TOSHIBA/data/featuresAndResponseDataframeCSVs/2018_6_15_fiveFoldCSVs'
#dataTypeList = ['LRnoMotifs', 'LSnoMotifs', 'LTnoMotifs'];
dataTypeList = ['LRnoMotifs'];

num_epochs = 200;
batch_size = 10;
filterNum = 1;
bias = True;
activation='relu';
iterNum = 10;

def calc_selectedfeat(origin_vec, knockoff_vec, q_thres):
    W = np.fabs(origin_vec)-np.fabs(knockoff_vec); print(W.shape)
	t = np.concatenate(([0],np.sort(np.fabs(W))));
	ratio = np.zeros(origin_vec);
	for j in range(origin_vec): ratio[j] = 1.0*len(np.where(W <= -t[j])[0]) / np.max((1, len(np.where(W >= t[j])[0]) ))
    
    T = np.inf;
    arr = np.where(ratio <= q_thres)[0];
    if len(arr) > 0:
        id = np.min(arr);
        T = t[id];
    
    qualifiedIndices = np.where(np.fabs(W) >= T)[0];
    return qualifiedIndices;
    

def show_layer_info(layer_name, layer_out):
    # print('[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())))
    pass

def build_DNN(p, coeff=0):

    input = Input(name='input', shape=(p, 2));
    show_layer_info('Input', input);

    local1 = LocallyConnected1D(filterNum,1, use_bias=bias, kernel_initializer=Constant(value=0.1))(input);
    show_layer_info('LocallyConnected1D', local1);

    local2 = LocallyConnected1D(1,1, use_bias=bias, kernel_initializer='glorot_normal')(local1);
    show_layer_info('LocallyConnected1D', local2);

    flat = Flatten()(local2);
    show_layer_info('Flatten', flat);

    dense1 = Dense(p, activation=activation,use_bias=bias, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1(coeff))(flat);
    show_layer_info('Dense', dense1);

    dense2 = Dense(p, activation=activation, use_bias=bias, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1(coeff))(dense1);
    show_layer_info('Dense', dense2);

    out_ = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(dense2)
    show_layer_info('Dense', out_)

    model = Model(inputs=input, outputs=out_)
    # model.compile(loss='mse', optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def train_DNN(model, X, y, myCallback):
    num_sequences = len(y);
    num_positives = np.sum(y);
    num_negatives = num_sequences - num_positives;

    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=1, class_weight={True: num_sequences / num_positives, False: num_sequences / num_negatives}, callbacks=[myCallback]);
    return model;

def test_DNN(model, X, y):
    return roc_auc_score(y, model.predict(X));

def predict_DNN(model, X):
    return model.predict(X);

def plot_roc(model, X, y):
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=model.predict(X));
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, linestyle='-', label='auc={:.4f}'.format(roc_auc));
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.legend(loc='lower right', fontsize=16)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show();

class My_Callback(keras.callbacks.Callback):
    def __init__(self, outputDir, pVal):
        self.outputDir = outputDir;
        self.pVal = pVal;
        print(self.outputDir);

    def on_epoch_end(self, epoch, logs={}):
        if ((epoch+1) % 100) != 0: return;

        h_local1_weight = np.array(self.model.layers[1].get_weights()[0]);
        h_local2_weight = np.array(self.model.layers[2].get_weights()[0]);

        print('h_local1_weight = ' + str(h_local1_weight.shape))
        print('h_local2_weight = ' + str(h_local2_weight.shape))
        h0 = np.zeros((self.pVal, 2));
        h0_abs = np.zeros((self.pVal, 2));

        for pIdx in range(self.pVal):
            h0[pIdx, :] = np.matmul(h_local1_weight[pIdx, :, :], h_local2_weight[pIdx, :, :]).flatten();
            h0_abs[pIdx, :] = np.matmul(np.fabs(h_local1_weight[pIdx, :, :]), np.fabs(h_local2_weight[pIdx, :, :])).flatten();

        print('h0 = ' + str(h0.shape))
        print('h0_abs = ' + str(h0_abs.shape))

        h1 = np.array(self.model.layers[4].get_weights()[0]);
        h2 = np.array(self.model.layers[5].get_weights()[0]);
        h3 = np.array(self.model.layers[6].get_weights()[0]);

        print('h1 = ' + str(h1.shape))
        print('h2 = ' + str(h2.shape))
        print('h3 = ' + str(h3.shape))

        W1 = h1;
        W_curr = h1;
        W2 = np.matmul(W_curr, h2);
        W_curr = np.matmul(W_curr, h2);
        W3 = np.matmul(W_curr, h3);

        print('W1 = ' + str(W1.shape))
        print('W2 = ' + str(W2.shape))
        print('W3 = ' + str(W3.shape))
        v0_h0 = h0[:, 0].reshape((self.pVal, 1));
        v1_h0 = h0[:, 1].reshape((self.pVal, 1));
        v0_h0_abs = h0_abs[:, 0].reshape((self.pVal, 1));
        v1_h0_abs = h0_abs[:, 1].reshape((self.pVal, 1));

        #v1 = np.vstack((v0_h0_abs, v1_h0_abs)).T;
        #v2 = np.vstack((np.sum(np.square(np.multiply(v0_h0_abs, np.fabs(W2))), axis=1).reshape((self.pVal, 1)), np.sum(np.square(np.multiply(v1_h0_abs, np.fabs(W2))), axis=1).reshape((self.pVal, 1)))).T;
        v3 = np.vstack((np.sum(np.square(np.multiply(v0_h0_abs, np.fabs(W3))), axis=1).reshape((self.pVal, 1)), np.sum(np.square(np.multiply(v1_h0_abs, np.fabs(W3))), axis=1).reshape((self.pVal, 1)))).T;

        v5 = np.vstack((np.sum(np.multiply(v0_h0, W3), axis=1).reshape((self.pVal, 1)),
                        np.sum(np.multiply(v1_h0, W3), axis=1).reshape((self.pVal, 1)))).T;

        with open(os.path.join(self.outputDir, 'result_epoch'+ str(epoch+1) +'_featImport.csv'), "a+") as myfile:
            myfile.write(','.join([str(x) for x in v3.flatten()]) + '\n');
        with open(os.path.join(self.outputDir, 'result_epoch'+ str(epoch+1) +'_featWeight.csv'), "a+") as myfile:
            myfile.write(','.join([str(x) for x in v5.flatten()]) + '\n');

for dataType in dataTypeList:

    x_url = os.path.join(dataDir, dataType, 'X_knockoff.csv'); print(x_url)
    y_url = os.path.join(dataDir, dataType, 'Y.csv');

    xvalues = pd.read_csv(x_url, header=None).values.astype(float);
    yvalues = pd.read_csv(y_url, header=None).values.astype(int);

    pVal = xvalues.shape[1] / 2;
    n = xvalues.shape[0];

    X_origin = xvalues[:, 0:pVal];
    X_knockoff = xvalues[:, pVal:];

    print(xvalues.shape)
    print(yvalues.shape)

    x3D_train = np.zeros((n, pVal, 2));
    x3D_train[:, :, 0] = X_origin;
    x3D_train[:, :, 1] = X_knockoff;
    label_train = yvalues;

    coeff = 0.05 * np.sqrt(2.0 * np.log(pVal) / n);
    outputDir = os.path.join(dataDir, dataType, 'result_2layer_epoch' + str(num_epochs) + '_batch' + str(batch_size) + '_knockoff_all');
    try:
        os.stat(outputDir)
    except:
        os.mkdir(outputDir)


    for iter in range(iterNum):
        print('iter=' + str(iter))
        np.random.seed(iter)

        DNN = build_DNN(pVal, coeff);

        myCallback = My_Callback(outputDir, pVal);
        DNN = train_DNN(DNN, x3D_train, label_train, myCallback);

