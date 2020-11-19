# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:57:58 2020

@author: Kunal
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from sklearn.metrics import plot_confusion_matrix
import re

def cal_p1(arr):
    return int(len(arr)/3)-5

def cal_p2(arr):
    return int(len(arr)*2/3)-5

def remove_nan(arr):
    arr = np.array(arr,dtype=np.float)
    to_delete = np.where(np.isnan(arr))#arr.index[np.isnan(arr)].tolist()    
    arr = np.delete(arr, to_delete)
    arr = arr.reshape((-1,1))
    # scaler = MinMaxScaler()
    # arr = scaler.fit_transform(arr)
    return arr 

def outlier(data):
    dataSort = list(data)
    dataSort.sort()
    #calculate interquartile ranges
    Q2 = np.median(dataSort)
    Q1 = np.median(dataSort[:int(len(dataSort)/2)])
    Q3 = np.median(dataSort[int(len(dataSort)/2):])
    IQR = Q3-Q1
    #calculate range of values within Inter Quartile range*1.5
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    outlierRemovedData = []
    # removedIndexes = []

    for index,count in enumerate(data):
        if (count>lower and count < upper):
            # removedIndexes.append(index)
            outlierRemovedData.append(count)          

    return outlierRemovedData

def plot_ecdf(data1,data2,label1,label2):
    data1 = np.maximum(data1, np.ones(len(data1))*1e-6)
    data2 = np.maximum(data2, np.ones(len(data2))*1e-6)
    for d in data2:
        if d==0:
            print(d)
    data1ECDF = ECDF(data1)
    data2ECDF = ECDF(data2)
    minVal = np.floor(np.minimum(np.amin(data1),np.amin(data2)))
    maxVal = np.ceil(np.maximum(np.amax(data1),np.amax(data2)))
    scale = np.linspace(minVal,maxVal)
    plt.step(scale,data1ECDF(scale),label=label1, where='post')
    plt.step(scale,data2ECDF(scale),label=label2, where='post')
    plt.title("ECDF plot")
    plt.legend()
    plt.show()
    
def plot_data(data,display_str):
    timeScale=np.arange(0,len(data),1)
    plt.plot(timeScale,data,label=display_str)
    plt.legend()
    plt.title('Graph')
    plt.show() 
    
def ks_test(data1,data2,label1,label2):
    # data1 = outlier(data1)
    # data2 = outlier(data2)
    
    # plot_ecdf(data1, data2, label1,label2)
    ks_result = stats.ks_2samp(data1, data2)
    if ks_result[1]>0.05:    
        # print(f"The K-S statistic for the 2 distributions is {ks_result[0]:0.2f}\
        #       and the p-value is {ks_result[1]} which is more than 0.05, so both the samples\
        #     come from the same distribution")
        # print(f"K-S Test p-value:{ks_result[1]*100:0.01f}%, Same")
        return ks_result[1]#float(f"{ks_result[1]:0.3f}")
    else:
        
        #     print(f"The K-S statistic for the 2 distributions is {ks_result[0]:0.2f}\
        # and the p-value is {ks_result[1]} which is less than 0.05, so both the samples\
        # come from different distributions")
        # print(f"K-S Test p-value:{ks_result[1]*100:0.01f}%, Diff")
        return ks_result[1]#float(f"{ks_result[1]:0.3f}")
        
def t_test(data1,data2):
    # baselineData = list(data[:100])
    # honestData = list(data[100:180])
    # maliciousData = list(data[180:])
    
    # data1 = outlier(data1)
    # data2 = outlier(data2)
    # maliciousData = outlier(maliciousData) 
    
    # data1 = np.random.choice(data1, size = min(len(data1),len(data2))) 
    # data2 = np.random.choice(data2, size = min(len(data1),len(data2))) 
    # print(len(data1), len(data2))
    
    t_result = stats.ttest_ind(data1, data2)
    if t_result[1]>0.05:    
        # print(f"The T-Test statistic for the 2 distributions is {t_result[0]:0.2f} and the p-value is {t_result[1]} which is more than 0.05, so\
        # both the samples come from the same distribution")
        # print(f"T-Test p-value:{t_result[1]*100:0.01f}%, Same")
        return t_result[1]#float(f"{t_result[1]:0.3f}")
    else:
        # print(f"The T-Test statistic for the 2 distributions is {t_result[0]:0.2f} and the p-value is {t_result[1]} which is less than 0.05, so\
        # both the samples come from different distributions")
        # print(f"T-Test p-value:{t_result[1]*100:0.01f}%, Diff")
        return t_result[1]#float(f"{t_result[1]:0.3f}")
    
        
def stat_test(cpu_vals,string):    
    change1 = int(len(cpu_vals)/3)
    change2 = int(len(cpu_vals)*2/3)
    baselineData = list(cpu_vals[:change1])
    honestData = list(cpu_vals[change1:change2])
    maliciousData = list(cpu_vals[change2:])
    
    # print(stats.shapiro(baselineData))
    # print(stats.shapiro(honestData))
    # print(stats.shapiro(maliciousData))
    
    # plt.hist(baselineData,bins=20)
    # plt.show()
    
    # plt.hist(honestData,bins=20)
    # plt.show()
    
    # plt.hist(maliciousData,bins=20)
    # plt.show()
    
    # plot_data(cpu_vals,"Honest App CPU")
    #statistical test to confirm whether baseline and barcode part is same or not
    skip = 10
    end = -10
    
    baselineData = baselineData[skip:end]
    honestData = honestData[skip:end]
    maliciousData = maliciousData[skip:end]
    ksReturn = []
    tReturn = []
    ks_cm = np.zeros((3,3))
    np.fill_diagonal(ks_cm, 1)
    t_cm = np.zeros((3,3))
    np.fill_diagonal(t_cm, 1)
    
    # print(f"\n{string}: Noise vs Part1")
    temp = ks_test(baselineData,honestData,"Noise Data","Part1 Data")
    ks_cm[0,1], ks_cm[1,0] = temp, temp
    # if temp>0.05:
    ksReturn.append(("B","P1",temp))
    
    temp = t_test(baselineData,honestData)
    t_cm[0,1], t_cm[1,0] = temp, temp
    # if temp>0.05:
    tReturn.append(("B","P1",temp))
    #---------------------------------------------------------------------------
    # print(f"\n{string}: Noise vs Part2")
    temp = ks_test(baselineData,maliciousData,"Noise Data","Part2 Data")
    ks_cm[0,2], ks_cm[2,0] = temp, temp
    # if temp>0.05:
    ksReturn.append(("B","P2",temp))
        
    temp = t_test(baselineData,maliciousData)  
    t_cm[0,2], t_cm[2,0] = temp, temp
    # if temp>0.05:
    tReturn.append(("B","P2",temp))
    #---------------------------------------------------------------------------
    # print(f"\n{string}: Part1 vs Part2")
    temp = ks_test(honestData,maliciousData,"Part1 Data","Part2 Data")
    ks_cm[2,1], ks_cm[1,2] = temp, temp
    # if temp>0.05:    
    ksReturn.append(("P1","P2",temp))
        
    temp = t_test(honestData,maliciousData)
    t_cm[2,1], t_cm[1,2] = temp, temp
    # if temp>0.05:
    tReturn.append(("P1","P2",temp))
    
    return ksReturn, tReturn, ks_cm, t_cm

def drawGraph(edges,label="Homogeneity Test"):
    G = nx.Graph()
    G.add_node("B")
    G.add_node("P1")
    G.add_node("P2")
    filtered = []
    for tuples in edges:
        if tuples[2]>0.05:
            filtered.append((tuples[0],tuples[1],float(f"{tuples[2]:0.3f}")))
    G.add_weighted_edges_from(filtered)
    pos = {"B":(0,0),"P1":(1,2),"P2":(2,0)}
    nx.draw_networkx(G,pos, node_size=1200, node_color = "Cyan", with_labels=True,font_size=20)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, font_size=20)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.1
    plt.xlim(x_min - x_margin, x_max + x_margin)
    y_max = max(y_values)
    y_min = min(y_values)
    y_margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.title(label,fontsize=20)
    plt.show()
    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          xLabel = "Video Sections",
                          yLabel = "Video Sections"):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize==True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif normalize=='all':
        cm = cm.astype('float') / cm.sum()


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()
    
def is_mal(graph):
    for tuples in graph:
        if tuples[2]>0.05:
            return False
    return True

def return_cpu(filename: str, num_samples: int=1, per_segment: int=30, dataset = True) -> np.array:
    
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
            if re.match(r'^-?\d+(?:\.\d+)?$', temp_df['MEM_PERC'].iloc[i]) is not None:
                temp_df['CPU_PERC'].iloc[i] = temp_df['MEM_PERC'].iloc[i]
            else:
                temp_df['CPU_PERC'].iloc[i] = temp_df['CPU_PERC'].iloc[i-1]
    
    cpu_arr = temp_df['CPU_PERC']
    
    
    
    def return_sections(cpu_arr):
        section_len = int(len(cpu_arr)/18)
        i=0
        while i<18:
            yield cpu_arr[section_len*i:section_len*(i+1)]
            i+=1

    return_arr = []
    for i in range(num_samples):
        selected = []
        if(per_segment>int(len(cpu_arr)/18)):
            replace_flag = True
        else:
            replace_flag = False        
        for section in return_sections(cpu_arr):
            if dataset:
                choice = np.random.choice(section, per_segment, replace=replace_flag)
                selected.extend(choice)                
            else:
                selected.extend(section)
        return_arr.append(selected)
    return np.array(return_arr, dtype=np.float64)