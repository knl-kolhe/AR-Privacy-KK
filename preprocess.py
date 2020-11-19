# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:54:28 2020

@author: Kunal
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from util import return_cpu

parser = argparse.ArgumentParser(
    description='preprocessing data')

parser.add_argument('--raw_data', default="./RawData/", type=str,
                    help='The folder which contains all the CPU traces')
parser.add_argument('--op_folder', default="./ProcessedData/", type=str,
                    help='The folder which contains the preprocessed data in csv format')
parser.add_argument('--scale_data', action='store_true', 
                    help='To scale the data horizontally into a range of 0 to 1 or not.')
parser.add_argument('--create_dataset', action='store_true', 
                    help='Include flag to create a structured dataset (Fixed rows x Fixed Columns)\
                     which can be used for training. Do not include flag if you just want to parse all values')
parser.add_argument('--per_segment', default=30, type=int,
                    help='The length of each tuple in the dataset will be 18 * per_segment')
parser.add_argument('--test_size', default=0.33, type=float,
                    help='The size of the test set. Default 33% of all data is test data')
args = parser.parse_args()


if args.create_dataset:
    num_samples_ = 5
    per_segment_ = args.per_segment
else:
    num_samples_ = 1
    per_segment_ = args.per_segment


files = [filename for filename in os.listdir(args.raw_data) if filename.endswith(".csv")]

final_labels = []
final_data = []
for filename in files:
    temp = filename.split("-")[:2]
    label = [0]*5
    #naming scheme needs more work but for now this is fine 5th October 2020
    if "Barcode" in temp:
        label[0] = 1
    if "Object" in temp:
        label[1] = 1
    if "Text" in temp:
        label[2] = 1
    if "Face" in temp:
        label[3] = 1   
    if "FaceFilter" in temp:
        label[4] = 1   
    print(filename,temp)
    final_data.extend(return_cpu(args.raw_data+filename, num_samples=num_samples_,per_segment = per_segment_, dataset=args.create_dataset))
    # if(not len(final_data[-1])==18*per_segment_):
    #     raise Exception(f"Length of returned array not equal to {18*per_segment_}")
    
    for i in range(num_samples_):
        final_labels.append(label)

final_data = np.array(final_data, dtype=object)
final_labels = np.array(final_labels)

# train = np.loadtxt(args.data, delimiter=',')
# label_csv = np.loadtxt(args.labels, delimiter=',')

label = np.zeros(final_labels.shape[0])
for i in range(final_labels.shape[0]):
    label[i] = sum(final_labels[i,:])-1

print(f"Option for create dataset: {args.create_dataset}")

if args.create_dataset:
    final_data_scaled = final_data
    if args.scale_data:    
        scaler = MinMaxScaler()    
        for i in range(final_data_scaled.shape[0]):
            final_data_scaled[i,:] = scaler.fit_transform(final_data_scaled[i,:].reshape(-1,1))[:,0]
    
    X_train, X_test, y_train, y_test = train_test_split(final_data_scaled, label, test_size=args.test_size)
    
    
    np.savetxt(args.op_folder+"data.csv", final_data_scaled, delimiter=",")
    np.savetxt(args.op_folder+"labels.csv", final_labels, delimiter=",")
    
    np.savetxt(args.op_folder+"X_train.csv", X_train, delimiter=",")
    np.savetxt(args.op_folder+"y_train.csv", y_train, delimiter=",")
    
    np.savetxt(args.op_folder+"X_test.csv", X_test, delimiter=",")
    np.savetxt(args.op_folder+"y_test.csv", y_test, delimiter=",")

else:
    parsed_cpu = pd.DataFrame()
    # parsed_cpu = pd.concat([parsed_cpu,pd.DataFrame(final_data[1])], axis=1)
    for array in final_data:
        parsed_cpu = pd.concat([parsed_cpu, pd.DataFrame(array)], axis=1)
    parsed_cpu.to_csv(args.op_folder+"parsed_cpu_values.csv", index = False, header = False)



    



