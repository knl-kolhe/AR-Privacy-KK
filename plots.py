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
import argparse

parser = argparse.ArgumentParser(
    description='preprocessing data')

parser.add_argument('--hon_CPU', default="./RawData/Barcode-9min.2020-10-07.170856.csv", type=str,
                    help='The path to the raw trace from an honest app.')
parser.add_argument('--mal_CPU', default="./RawData/Barcode-Text-9min.2020-10-07.164217.csv", type=str,
                    help='The path to the raw trace from a malicious app.')
parser.add_argument('--stat_example', action='store_true',
                    help="Include flag to get the statistical analysis of the respective traces")
args = parser.parse_args()


   
honest_cpu = return_cpu(args.hon_CPU, dataset=False)[0,:]
mal_cpu = return_cpu(args.mal_CPU, dataset=False)[0,:]

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

if(args.stat_example):
    ksEdge, tEdge, ks_cm, t_cm = stat_test(honest_cpu, "Honest App")
    drawGraph(ksEdge, "Honest App KS-Test")
    drawGraph(tEdge, "Honest App T-Test")
    plot_confusion_matrix(ks_cm,
                          target_names = ["Baseline", "Part1", "Part2"],
                          title = "Honest App KS-Test")
    plot_confusion_matrix(t_cm,
                          target_names = ["Baseline", "Part1", "Part2"],
                          title = "Honest App T-Test")
    
    
    
    ksEdge, tEdge, ks_cm, t_cm = stat_test(mal_cpu, "Malicious App")
    drawGraph(ksEdge,"Malicious App KS-Test")
    drawGraph(tEdge, "Malicious App T-Test")
    plot_confusion_matrix(ks_cm,
                          target_names = ["Baseline", "Part1", "Part2"],
                          title = "Malicious App KS-Test")
    plot_confusion_matrix(t_cm,
                          target_names = ["Baseline", "Part1", "Part2"],
                          title = "Malicious App T-Test")





