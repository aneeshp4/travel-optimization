#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:32:56 2022

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

def order_node_and_zone_generation_one_branch(branch_all_nodes_input, area_non_walk_df, depot_node, G, longest_distance_from_depot):

    def remove_dummy_nodes():
        branch_all_nodes=branch_all_nodes_input
        num_real_nodes=area_non_walk_df.shape[0]+1
        branch_all_nodes_after_deleting=[]
        
        for i in range(len(branch_all_nodes)):
            cur_branch_old=branch_all_nodes[i]
            cur_branch_new=[]
            for j in range(len(cur_branch_old)):
                cur_node_old=cur_branch_old[j]
                if cur_node_old>=num_real_nodes:
                    continue
                else:
                    cur_branch_new.append(cur_node_old)
            if len(cur_branch_new)>0:
                branch_all_nodes_after_deleting.append(cur_branch_new)
           
        return branch_all_nodes_after_deleting
    
    branch_all_nodes_after_deleting=remove_dummy_nodes()
    
    
    all_index_after_rebranching=branch_all_nodes_after_deleting
    re_index_after_rebranching=[]
    for i in range (len(all_index_after_rebranching)):
        cur_branch=[]
        for j in range(len(all_index_after_rebranching[i])):
            cur_branch.append(all_index_after_rebranching[i][j]-1)
        re_index_after_rebranching.append(cur_branch)
      
    all_index_after_rebranching =re_index_after_rebranching  
                               
    all_nodes_df=area_non_walk_df
    def get_distance_sort_list_after():
        path_lengths={}
        for j in range(len(all_index_after_rebranching)):
            path_lengths[j]=[]
            cur_orig_node=depot_node
            near_points=all_index_after_rebranching[j]
            for i in range(len(near_points)):
                cur_dest=(all_nodes_df.iloc[near_points[i]]['pad_nodes_lat'], all_nodes_df.iloc[near_points[i]]['pad_nodes_lon'])
                cur_dest_node=ox.get_nearest_node(G, cur_dest)
                cur_short_path_length=nx.shortest_path_length(G, cur_orig_node, cur_dest_node, weight='length') 
                path_lengths[j].append(cur_short_path_length)
        return path_lengths
                    
    path_lengths_after=get_distance_sort_list_after()
    
    
    def sort_distance_node_after():
        distance_beyond=[1000000 for i in range(all_nodes_df.shape[0])]
        assignment_list=[1000000 for i in range(all_nodes_df.shape[0])]
        assignment_key=0
        for j in range(len(path_lengths_after)):
            cur_branch=path_lengths_after[j]
            cur_branch_node=all_index_after_rebranching[j]
            for i in range(len(cur_branch)):
                cur_dist=cur_branch[i]
                cur_index=cur_branch_node[i]

                distance_beyond[cur_index]=cur_dist
                assignment_list[cur_index]=assignment_key
            assignment_key+=1
        
        return distance_beyond, assignment_list

    distance_beyond_after, assignment_list_after=sort_distance_node_after()
    orde_node_df_after=pd.DataFrame(list(zip(assignment_list_after,distance_beyond_after)), columns=['group_index','distance'])

    def order_distance_node_after():
        ranked_node_df=orde_node_df_after.sort_values(by=['distance'], ascending=True)
        ranked_node_df=ranked_node_df.reset_index()
        ranked_node_df = ranked_node_df.rename(columns={'index': 'node_index'})
        node_order_list=[[] for i in range(len(path_lengths_after))]
        node_distance_order_list=[[] for i in range(len(path_lengths_after))]
        for i in range (ranked_node_df.shape[0]):
            cur_group=int(ranked_node_df.iloc[i]['group_index'])
            cur_node=int(ranked_node_df.iloc[i]['node_index'])
            cur_dist=ranked_node_df.iloc[i]['distance']
            if cur_group<1000000:
                node_order_list[cur_group].append(cur_node)
                node_distance_order_list[cur_group].append(cur_dist)
            
        return node_order_list, node_distance_order_list

    node_order_list_after, node_distance_order_list_after=order_distance_node_after()
    
     
    
    return node_order_list_after, node_distance_order_list_after




def get_combine_node_order(across_branch_nodes_input, start_num_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot):

    all_node_order=[]
 
    for i in range(len(across_branch_nodes_input)):
        cur_decision_branch=across_branch_nodes_input[i]
        node_order_list_decision, node_distance_order_list_decision=order_node_and_zone_generation_one_branch(cur_decision_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot)
        all_node_order.append(node_order_list_decision)
        
    
    return all_node_order
    


def get_final_decision_zone_list(all_decision_list_result_input):
    
    decision_zone_list=all_decision_list_result_input[0]
    for i in range(1, len(all_decision_list_result_input)):
        cur_decision_list=all_decision_list_result_input[i]
        for j in range(len(cur_decision_list)):
            cur_zone=cur_decision_list[j]
            if not cur_zone in decision_zone_list:
                decision_zone_list.append(cur_zone)
                
    return decision_zone_list




