# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:54:28 2020

@author: Kunal
"""

import pandas as pd
import numpy as np
import os
import re

def return_cpu(filename: str, num_samples: int=5, per_segment: int=30) -> np.array:
    
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
            choice = np.random.choice(section, per_segment, replace=replace_flag)
#            print(len(choice), len(section))
            selected.extend(choice)
        return_arr.append(selected)
    return np.array(return_arr, dtype=np.float64)

files = [filename for filename in os.listdir("./RawData") if filename.endswith(".csv")]

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
    num_samples_ = 5
    per_segment_ = 30
    final_data.extend(return_cpu("./RawData/"+filename, num_samples=num_samples_))
    if(not len(final_data[-1])==18*per_segment_):
        raise Exception(f"Length of returned array not equal to {18*per_segment_}")
    print(filename,temp)
    for i in range(num_samples_):
        final_labels.append(label)

final_data = np.array(final_data)
final_labels = np.array(final_labels)

np.savetxt("./ProcessedData/data.csv", final_data, delimiter=",")
np.savetxt("./ProcessedData/labels.csv", final_labels, delimiter=",")






    



