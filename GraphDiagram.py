# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:48:52 2020

@author: Kunal
"""

import networkx as nx

G = nx.Graph()
G.add_node("N")
G.add_node("P1")
G.add_node("P2")
pos = {"N":(0,0),"P1":(30,60),"P2":(60,0)}
nx.draw(G,pos, node_size=800, node_color = "Cyan", with_labels=True)