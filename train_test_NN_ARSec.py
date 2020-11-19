# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 07:41:39 2020

@author: Kunal
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from util import *
from sklearn.metrics import accuracy_score
from NN_ARSec_model import *
import argparse

parser = argparse.ArgumentParser(
    description='train_test_model')

parser.add_argument('--data_path', default="./ProcessedData/", type=str,
                    help='Path to data files X_train, y_train, X_test, y_test')
# parser.add_argument('--labels', default="./ProcessedData/labels.csv", type=str,
#                     help='Path to label for the data')
parser.add_argument('--model_path', default="./trained_model/", type=str,
                    help='Output path for trained model where model is placed')
parser.add_argument('--num_epochs', default=20, type=int,
                    help='Number of epochs the model should be trained for')
parser.add_argument('--batch_size', default=50, type=int,
                    help='The batch size for training the model')
args = parser.parse_args()


X_train = np.loadtxt(args.data_path+"X_train.csv", delimiter=',')
y_train = np.loadtxt(args.data_path+"y_train.csv", delimiter=',')
X_test = np.loadtxt(args.data_path+"X_test.csv", delimiter=',')
y_test = np.loadtxt(args.data_path+"y_test.csv", delimiter=',')

model = return_NN_ARSec()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size)

model.save(args.model_path+f"trained_model_{args.num_epochs}.h5")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, np.round(y_pred))
print('Accuracy on X_test is: %.5f' % (accuracy*100))

cm_NN = confusion_matrix(y_test, np.round(y_pred))
plot_confusion_matrix(cm_NN,
                      target_names = ["Honest", "Malicious"],
                      title = "training Confusion Matrix", normalize=True,
                      xLabel = "Predicted Category", yLabel = "Actual Category")
