# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 08:43:37 2020

@author: Kunal
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from scipy import ndimage
from util import *
# from combine import *
# from sklearn.metrics import plot_confusion_matrix

   

# df_hon = pd.read_csv("./Videos Dataset/Barcode-Text/CombinedData/Barcode-Text-Data.csv")
df_hon = pd.read_csv("./Videos Dataset/Object-Text/CombinedData/Combined-headless_fb-Object-Text-13pt5min.csv")
df_mal = pd.read_csv("./Videos Dataset/Object-Text/CombinedData/Combined-headless_fb_mal-Object-Text-13pt5min.csv")


# honest_cpu = df_hon["Honest-Barcode-Text-Handheld-NonHeadless-CPU"]
# mal_cpu = df_hon["Mal-Barcode-Text-Handheld-NonHeadless-CPU"]
key = "trial1"
honest_cpu = df_hon[key]
mal_cpu = df_mal[key]



honest_cpu = remove_nan(honest_cpu)
# honest_mem = remove_NaN(honest_mem)
mal_cpu = remove_nan(mal_cpu)
# mal_mem = remove_NaN(mal_mem)


# smooth_honest_cpu = ndimage.gaussian_filter1d(honest_cpu,2)
# smooth_mal_cpu = ndimage.gaussian_filter1d(mal_cpu,2)

# CPU values Graph honest
plt.plot(np.arange(0,len(honest_cpu),1),honest_cpu,label='Honest')
# plt.plot(np.arange(0,len(honest_cpu),1),smooth_honest_cpu,label='Smooth honest')
plt.axvline(x=int(len(honest_cpu)/3),ymin=0,ymax=5,c='g')
plt.axvline(x=int(len(honest_cpu)*2/3),ymin=0,ymax=5,c='g')
plt.legend()
plt.title('Honest app processor values')
plt.xlabel('Seconds')
plt.ylabel('% utilization')
plt.show()

# CPU values Graph Malicious
timeScale=np.arange(0,len(mal_cpu),1)
plt.plot(timeScale,mal_cpu,label='Malicious')
# plt.plot(timeScale,smooth_mal_cpu,label='Smooth Malicious')
plt.axvline(x=int(len(mal_cpu)/3),ymin=0,ymax=5,c='g')
plt.axvline(x=int(len(mal_cpu)*2/3),ymin=0,ymax=5,c='g')
plt.legend()
plt.title('Malicious app processor values')
plt.xlabel('Seconds')
plt.ylabel('% utilization')
plt.show()

# CPU values Graph Honest vs Malicious
plt.plot(np.arange(0,len(honest_cpu),1),honest_cpu,label='Honest')
plt.plot(np.arange(0,len(mal_cpu),1),mal_cpu,label='Malicious')
plt.axvline(x=int(len(honest_cpu)/3),ymin=0,ymax=5,c='g')
plt.axvline(x=int(len(honest_cpu)*2/3),ymin=0,ymax=5,c='g')
plt.legend()
plt.title('Processor values')
plt.xlabel('Seconds')
plt.ylabel('% utilization')
plt.show()

# Memory values Graph Honest vs Malicious
# plt.plot(np.arange(0,len(honest_mem),1),honest_mem,label='Honest')
# plt.plot(np.arange(0,len(mal_mem),1),mal_mem,label='Malicious')
# # plt.axvline(x=25,ymin=0,ymax=5,c='g')
# # plt.ylim(0,100)
# plt.legend()
# plt.title('Memory values')
# plt.xlabel('Seconds')
# plt.ylabel('% utilization')
# plt.show()

ksEdge, tEdge, ks_cm, t_cm = stat_test(honest_cpu[:,0], "Honest App")
drawGraph(ksEdge, "Honest App KS-Test")
drawGraph(tEdge, "Honest App T-Test")
plot_confusion_matrix(ks_cm,
                      target_names = ["Baseline", "Part1", "Part2"],
                      title = "Honest App KS-Test")
plot_confusion_matrix(t_cm,
                      target_names = ["Baseline", "Part1", "Part2"],
                      title = "Honest App T-Test")



ksEdge, tEdge, ks_cm, t_cm = stat_test(mal_cpu[:,0], "Malicious App")
drawGraph(ksEdge,"Malicious App KS-Test")
drawGraph(tEdge, "Malicious App T-Test")
plot_confusion_matrix(ks_cm,
                      target_names = ["Baseline", "Part1", "Part2"],
                      title = "Malicious App KS-Test")
plot_confusion_matrix(t_cm,
                      target_names = ["Baseline", "Part1", "Part2"],
                      title = "Malicious App T-Test")

# print("Smoothened using Gaussian sigma=2")
# stat_test(smooth_honest_cpu, "Honest App")
# stat_test(smooth_mal_cpu, "Malicious App")




