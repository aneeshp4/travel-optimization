#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:27:17 2022

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
from  shortest_path_tree import *
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')



##################################################################################################################


def get_parent_string(this_node_for_parent, tree_nodes, node_parent):
    
    parent_string=[]
    if this_node_for_parent==0:
        reached_depot=1
    else:
        reached_depot=0

    while reached_depot==0:
        cur_index=tree_nodes.index(this_node_for_parent)
        parent_string.append(node_parent[cur_index])
        this_node_for_parent=node_parent[cur_index]
        if this_node_for_parent==0:
            reached_depot=1
            
    return parent_string
 


def fine_division(a, dist_limit, area_non_walk_df, G, combined_shortest_dummy_coord):
    node_point_amount=area_non_walk_df.shape[0]+1
    initial_range=len(a)
    initial_nodes=a.copy() 
    
    initial_range=len(a)
    initial_nodes=a.copy()   
    some_change=0
    
    all_first_nodes={}
    all_second_nodes={}
    all_distance_fs={}
    max_distance_fs={}
    for i in range(0, initial_range):
        
        print(i)
        first_nodes=[]
        second_nodes=[]
        distance_fs=[]
        
        cur_branch_nodes=initial_nodes[i]
        print(len(cur_branch_nodes))
        
        
        for j in range(len(cur_branch_nodes)-1):
            for k in range(j+1,len(cur_branch_nodes)):
                if cur_branch_nodes[j]<node_point_amount and cur_branch_nodes[k]<node_point_amount:
                    if cur_branch_nodes[j]<cur_branch_nodes[k]:
                        first_nodes.append(cur_branch_nodes[j])
                        second_nodes.append(cur_branch_nodes[k])
                        cur_first_node=ox.get_nearest_node(G, combined_shortest_dummy_coord[cur_branch_nodes[j]])
                        cur_second_node=ox.get_nearest_node(G, combined_shortest_dummy_coord[cur_branch_nodes[k]])
                    else:
                        first_nodes.append(cur_branch_nodes[k])
                        second_nodes.append(cur_branch_nodes[j])
                        cur_first_node=ox.get_nearest_node(G, combined_shortest_dummy_coord[cur_branch_nodes[k]])
                        cur_second_node=ox.get_nearest_node(G, combined_shortest_dummy_coord[cur_branch_nodes[j]])

                    try:            
                        cur_short_path_length=nx.shortest_path_length(G, cur_first_node, cur_second_node, weight='length') 
                        distance_fs.append(cur_short_path_length)
                    except nx.NetworkXNoPath:   
                        try:            
                            cur_short_path_length=nx.shortest_path_length(G,cur_second_node,  cur_first_node, weight='length') 
                            distance_fs.append(cur_short_path_length)
                        except nx.NetworkXNoPath:
                            distance_fs.append(dist_limit-1)

        all_first_nodes[i]= first_nodes
        all_second_nodes[i]= second_nodes
        all_distance_fs[i]= distance_fs
        
        if len(distance_fs)==0:
            max_distance_fs[i]=0
        else:
            max_distance_fs[i]=max(distance_fs)
        
        
      
    return all_first_nodes, all_second_nodes, all_distance_fs, max_distance_fs

##################################################################################################################


def one_more_division(first_nodes, second_nodes, distance_fs, dist_limit, tree_nodes, node_parent, node_children):
    if len(distance_fs)>0:              
        if max(distance_fs)>dist_limit: 
            position_max=distance_fs.index(max(distance_fs))
            max_first_node=first_nodes[position_max]
            max_second_node=second_nodes[position_max]
            
            first_parent_string=get_parent_string(max_first_node, tree_nodes, node_parent)
            second_parent_string=get_parent_string(max_second_node, tree_nodes, node_parent)
            common_parent=0
            for l in second_parent_string:
                if l in first_parent_string:
                    common_parent=l
                    break
            first_parent_index=first_parent_string.index(common_parent)
            first_child=0
            if first_parent_index==0:
                first_child=max_first_node
            else:
                first_child=first_parent_string[first_parent_index-1]
                
            second_parent_index=second_parent_string.index(common_parent)
            second_child=0
            if second_parent_index==0:
                second_child=max_second_node
            else:
                second_child=second_parent_string[second_parent_index-1]
            
                
            common_index=tree_nodes.index(common_parent)
            node_children[common_index].remove(second_child)
            out_index=tree_nodes.index(second_child)
            node_parent[out_index]=0
            node_children[0].append(second_child)
    branch_size_new, branch_all_nodes_new = branch_size_of_tree(0, tree_nodes, node_children)
    
    return node_children, node_parent, branch_size_new, branch_all_nodes_new  

##################################################################################################################

def get_division_one_more(all_first_nodes_input, all_second_nodes_input, all_distance_fs_input, max_distance_fs_input, dist_limit, tree_nodes, node_children, node_parent, branch_size, branch_all_nodes):
    
    #dist_limit=3352
    this_one_more=0
    cur_max_pair_branch_index=max(max_distance_fs_input.items(), key=operator.itemgetter(1))[0]
    cur_max_pair_dist=max_distance_fs_input[cur_max_pair_branch_index]
    if cur_max_pair_dist<=dist_limit:
        print("end num branch: "+ str(len(max_distance_fs_input)))
        node_children_out=node_children
        node_parent_out=node_parent
        branch_size_new_out=branch_size
        branch_all_nodes_new_out=branch_all_nodes
        pass
    else:
        this_one_more=1
        node_children_a, node_parent_a, branch_size_new_a, branch_all_nodes_new_a=one_more_division(all_first_nodes_input[cur_max_pair_branch_index], all_second_nodes_input[cur_max_pair_branch_index], all_distance_fs_input[cur_max_pair_branch_index], dist_limit, tree_nodes, node_parent, node_children)
        node_children_out=node_children_a
        node_parent_out=node_parent_a
        branch_size_new_out=branch_size_new_a
        branch_all_nodes_new_out=branch_all_nodes_new_a
    

    return node_children_out, node_parent_out, branch_size_new_out, branch_all_nodes_new_out, this_one_more, cur_max_pair_branch_index

##################################################################################################################

def generate_all_branch_choices(tree_nodes, node_children, branch_all_nodes, area_non_walk_df, G, combined_shortest_dummy_coord, node_parent, branch_size):
    across_first_nodes={}
    across_second_nodes={}
    across_distance_fs={}
    across_previous_branch_root={}
    across_current_branch_root={}
    across_previous_branch_split={}
    across_branch_nodes={}
    
    #dist_limit=3352
    dist_limit=1700
    start_index=0
    flag_more=1
    while flag_more==1:
        
        previous_children_of_root=node_children[0].copy()
        #current_num_branch=len(branch_size)
        all_first_nodes, all_second_nodes, all_distance_fs, max_distance_fs= fine_division(branch_all_nodes, dist_limit, area_non_walk_df, G, combined_shortest_dummy_coord) 
        across_first_nodes[start_index]=all_first_nodes
        across_second_nodes[start_index]=all_second_nodes
        across_distance_fs[start_index]=all_distance_fs
        across_previous_branch_root[start_index]=previous_children_of_root
        
        across_branch_nodes[start_index]=branch_all_nodes
        node_children, node_parent, branch_size, branch_all_nodes, this_one_more, cur_max_pair_branch_index=get_division_one_more(all_first_nodes, all_second_nodes, all_distance_fs, max_distance_fs,dist_limit, tree_nodes, node_children, node_parent, branch_size, branch_all_nodes)
        after_children_of_root=node_children[0].copy()
        across_current_branch_root[start_index]=after_children_of_root
        across_previous_branch_split[start_index]=cur_max_pair_branch_index
        
        
        
        start_index=start_index+1
        flag_more=this_one_more

    return across_first_nodes, across_second_nodes, across_distance_fs, across_previous_branch_root, across_current_branch_root, across_previous_branch_split, across_branch_nodes

##################################################################################################################
##################################################################################################################

