# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:54:28 2020

@author: Kunal
"""

import pandas as pd
import numpy as np
import os
import re
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='preprocessing data')

parser.add_argument('--raw_data', default="./RawData/", type=str,
                    help='The folder which contains all the CPU traces')
parser.add_argument('--op_folder', default="./ProcessedData/", type=str,
                    help='The folder which contains the preprocessed data in csv format')
parser.add_argument('--scale_data', default=True, type=bool,
                    help='To scale the data horizontally into a range of 0 to 1 or not.')
parser.add_argument('--create_dataset', default=True, type=bool,
                    help='False to just parse all CPU values from csv trace files and output to csv with labels')
parser.add_argument('--per_segment', default=30, type=int,
                    help='the length of each tuple in the dataset will be 18 * per_segment')
args = parser.parse_args()


if args.create_dataset:
    num_samples_ = 5
    per_segment_ = args.per_segment
else:
    num_samples_ = 1
    per_segment_ = args.per_segment
    

def return_cpu(filename: str, num_samples: int=5, per_segment: int=30, dataset = True) -> np.array:
    
    with open(filename) as file_data:
        file_data = file_data.read()
        file_data = file_data.split("\n")
        while 'TRIAL' not in file_data[0]:
            file_data.pop(0)
        while not file_data[-1].split(',')[0].isnumeric():
            file_data.pop()
        for i, each_row in enumerate(file_data):
            file_data[i] = each_row.split(",")
        
    temp_df = pd.DataFrame(file_data[1:], columns=file_data[0])
        
    for i, val in enumerate(temp_df['CPU_PERC']):
        if re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
            temp_df['CPU_PERC'].iloc[i] = temp_df['MEM_PERC'].iloc[i]
    
    cpu_arr = temp_df['CPU_PERC']
    
    
    
    def return_sections(cpu_arr):
        section_len = int(len(cpu_arr)/18)
        i=0
        while i<18:
            yield cpu_arr[section_len*i:section_len*(i+1)]
            i+=1
#        for i in range(0,len(cpu_arr)-section_len,section_len):
#            yield cpu_arr[i:i+section_len]
    
#    num_samples = 5
    return_arr = []
    for i in range(num_samples):
        selected = []
#        per_segment = 30
        if(per_segment>int(len(cpu_arr)/18)):
            replace_flag = True
        else:
            replace_flag = False        
        for section in return_sections(cpu_arr):
            if dataset:
                choice = np.random.choice(section, per_segment, replace=replace_flag)
    #            print(len(choice), len(section))
                selected.extend(choice)                
            else:
                selected.extend(section)
                # print(len(section))
        return_arr.append(selected)
    return np.array(return_arr, dtype=np.float64)

files = [filename for filename in os.listdir(args.raw_data) if filename.endswith(".csv")]

final_labels = []
final_data = []
for filename in files:
    temp = filename.split("-")[:2]
    label = [0]*4
    #naming scheme needs more work but for now this is fine 5th October 2020
    if "Barcode" in temp:
        label[0] = 1
    if "Object" in temp:
        label[1] = 1
    if "Text" in temp:
        label[2] = 1
    if "Face" in temp:
        label[3] = 1        
    final_data.extend(return_cpu(args.raw_data+filename, num_samples=num_samples_,per_segment = per_segment_, dataset=args.create_dataset))
    # if(not len(final_data[-1])==18*per_segment_):
    #     raise Exception(f"Length of returned array not equal to {18*per_segment_}")
    print(filename,temp)
    for i in range(num_samples_):
        final_labels.append(label)

final_data = np.array(final_data, dtype=object)
final_labels = np.array(final_labels)

# train = np.loadtxt(args.data, delimiter=',')
# label_csv = np.loadtxt(args.labels, delimiter=',')

label = np.zeros(final_labels.shape[0])
for i in range(final_labels.shape[0]):
    label[i] = sum(final_labels[i,:])-1

if args.create_dataset:
    final_data_scaled = final_data
    if args.scale_data:    
        scaler = MinMaxScaler()    
        for i in range(final_data_scaled.shape[0]):
            final_data_scaled[i,:] = scaler.fit_transform(final_data_scaled[i,:].reshape(-1,1))[:,0]
    
    X_train, X_test, y_train, y_test = train_test_split(final_data_scaled, label, test_size=0.33)
    
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



    



