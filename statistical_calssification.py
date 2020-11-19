# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:54:41 2020

@author: Kunal
"""

from util import *
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser(
    description='preprocessing data')

parser.add_argument('--raw_CPU', default="./ProcessedData/parsed_cpu_values.csv", type=str,
                    help='The path to the file which contains the parsed cpu values for the traces.')
parser.add_argument('--labels', default="./ProcessedData/labels.csv", type=str,
                    help='The path to the csv file containing all the labels for the respective CPU trace')
args = parser.parse_args()


data = pd.read_csv(args.raw_CPU,header=None)
labels = pd.read_csv(args.labels,header=None)
labels = np.sum(labels,1)-1
labels = labels[np.mod(np.arange(labels.size),5)==0]

pred_ks = []
pred_t = []
total = data.shape[1]

for i in range(total):
    temp = remove_nan(data.iloc[:,i])
    ksEdge, tEdge, ks_cm, t_cm = stat_test(temp[:,0], "Honest App")
    
    if is_mal(ksEdge):        
        pred_ks.append(1)
    else:
        pred_ks.append(0)
    
    if is_mal(tEdge):        
        pred_t.append(1)
    else:
        pred_t.append(0)
    

ksTest_cm = confusion_matrix(labels, pred_ks)
plot_confusion_matrix(ksTest_cm, ["Honest", "Mal"],title="KS-Test for homogeneity test", normalize='all', xLabel="Predicted", yLabel="Actual")

tTest_cm = confusion_matrix(labels, pred_t)
plot_confusion_matrix(tTest_cm, ["Honest", "Mal"], title = "T-Test for homogeneity test", normalize='all', xLabel="Predicted", yLabel="Actual")

# ksEdge, tEdge, ks_cm, t_cm = stat_test(honest_cpu[:,0], "Honest App")
# ksEdge, tEdge, ks_cm, t_cm = stat_test(mal_cpu[:,0], "Malicious App")