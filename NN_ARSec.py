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


train = np.loadtxt("./ProcessedData/data.csv", delimiter=',')
label_csv = np.loadtxt("./ProcessedData/labels.csv", delimiter=',')

label = np.zeros(label_csv.shape[0])
for i in range(label_csv.shape[0]):
    label[i] = sum(label_csv[i,:])-1

scaler = MinMaxScaler()

for i in range(train.shape[0]):
    train[i,:] = scaler.fit_transform(train[i,:].reshape(-1,1))[:,0]

X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=540, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=50)

_, accuracy = model.evaluate(X_test, y_test)

result = model.predict(X_test)
print('Accuracy: %.2f' % (accuracy*100))