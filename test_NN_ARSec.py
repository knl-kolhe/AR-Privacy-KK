# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 04:22:13 2020

@author: Kunal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from util import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import argparse
from tensorflow import keras


parser = argparse.ArgumentParser(
    description='test_model')

parser.add_argument('--data_path', default="./ProcessedData/", type=str,
                    help='Path to data files X_train, y_train, X_test, y_test')
parser.add_argument('--model_path', default="./trained_model/trained_model_20.h5", type=str,
                    help='Path for trained model')
args = parser.parse_args()

X_test = np.loadtxt(args.data_path+"X_test.csv", delimiter=',')
y_test = np.loadtxt(args.data_path+"y_test.csv", delimiter=',')

model = keras.models.load_model(args.model_path)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, np.round(y_pred))
print('Accuracy on X_test is: %.5f' % (accuracy*100))

cm_NN = confusion_matrix(y_test, np.round(y_pred))
plot_confusion_matrix(cm_NN,
                      target_names = ["Honest", "Malicious"],
                      title = "training Confusion Matrix", normalize=True,
                      xLabel = "Predicted Category", yLabel = "Actual Category")
