#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:47:03 2022

@author: ys
"""

import json
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
import random
import math
import scipy
from scipy.stats import poisson
import time
import copy
import statistics
import operator
import itertools
from itertools import combinations, product
import gurobipy as gp
from gurobipy import GRB
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')



def folium_map_result(depot_lat, depot_lon, decision_nodes, area_non_walk_df, depot_point, region_name, vehicle_ub, result_Vars_v):
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=2)
    colors = ['red', 
              'blue', 
              'green', 
              'purple', 
              'orange', 
              'pink',
              'gray', 
              'beige', 
              'white',
              'brown',
              'olive',
              'cyan']


    re_branch_nodes=decision_nodes 
    iterator=0
    for j in range(len(re_branch_nodes)):
        cur_all_nodes=re_branch_nodes[j]
        cur_color=colors[iterator]
        for i in range(len(cur_all_nodes)):
            #print(cur_color)
            cur_loc=(area_non_walk_df['pad_nodes_lat'][cur_all_nodes[i]], area_non_walk_df['pad_nodes_lon'][cur_all_nodes[i]])
            #print(cur_loc)
            folium.CircleMarker(location=cur_loc,radius=5,fill=True, popup=cur_loc, fill_color=cur_color, color='black', fill_opacity=1).add_to(m)
        iterator+=1

    folium.Marker(location=depot_point,radius=10,fill=True, popup=depot_point, fill_color='black', color='black', fill_opacity=1).add_to(m)
    
    map_name=region_name+"_1size_"+str(vehicle_ub)+"_"+str(result_Vars_v)+".html"
    m.save(map_name) 

def folium_map_result_zone(num_branches_here, depot_lat, depot_lon, decision_nodes, area_non_walk_df, depot_point, region_name, vehicle_ub, result_Vars_v):
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=2)
    colors = ['blue', 
              'orange', 
              'green', 
              'red', 
              'purple', 
              'brown',
              'pink', 
              'gray', 
              'olive',
              'cyan',
              'lightcoral',
              'burlywood',
              'khaki',
              'yellowgreen',
              'lime',
              'mediumaquamarine',
              'paleturquoise',
              'cadetblue',
              'steelblue',
              'lightstellblue',
              'midnightblue',
              'dartslateblue',
              'blueviolet',
              'violet',
              'magenta',
              'deeppink',
              'palevioletred',
              'dodgerblue',
              'chartreuse',
              'chocolate',
              'darkorchid',
              'snow',
              'lightsalmon',
              'saddlebrown',
              'lawngreen',
              'oldlace',
              'gold',
              'moccasin',
              'cadetblue',
              'darkslategrey',
              'fuchsia',
              'thistle',
              'seagreen',
              'burlywood']


    re_branch_nodes=decision_nodes 
    iterator=0
    for j in range(len(re_branch_nodes)):
        cur_all_nodes=re_branch_nodes[j]
        cur_color=colors[iterator]
        for i in range(len(cur_all_nodes)):
            #print(cur_color)
            cur_loc=(area_non_walk_df['pad_nodes_lat'][cur_all_nodes[i]], area_non_walk_df['pad_nodes_lon'][cur_all_nodes[i]])
            #print(cur_loc)
            folium.CircleMarker(location=cur_loc,radius=5,fill=True, popup=cur_loc, fill_color=cur_color, color='black', fill_opacity=1).add_to(m)
        iterator+=1

    folium.Marker(location=depot_point,radius=10,fill=True, popup=depot_point, fill_color='black', color='black', fill_opacity=1).add_to(m)
    
    map_name=region_name+"_zone_fullsize_"+str(vehicle_ub)+"_"+str(result_Vars_v)+"_"+str(num_branches_here)+".html"
    m.save(map_name) 

