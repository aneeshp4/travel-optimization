#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 13:16:31 2022

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
from itertools import chain, combinations
from itertools import combinations, product
import gurobipy as gp
from gurobipy import GRB
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


# =============================================================================
# def remove_dummy_nodes(branch_all_nodes_input, area_non_walk_df):
#     branch_all_nodes=branch_all_nodes_input
#     num_real_nodes=area_non_walk_df.shape[0]
#     branch_all_nodes_after_deleting=[]
#     
#     for i in range(len(branch_all_nodes)):
#         cur_branch_old=branch_all_nodes[i]
#         cur_branch_new=[]
#         for j in range(len(cur_branch_old)):
#             cur_node_old=cur_branch_old[j]
#             if cur_node_old>=num_real_nodes:
#                 continue
#             else:
#                 cur_branch_new.append(cur_node_old)
#         if len(cur_branch_new)>0:
#             branch_all_nodes_after_deleting.append(cur_branch_new)
#        
#     return branch_all_nodes_after_deleting
# 
# 
# def get_across_branch_nodes_after_deleting(across_branch_nodes_in, area_non_walk_df):
# 
#     across_branch_nodes_after_deleting_out=[]
#     for i in range(len(across_branch_nodes_in)):
#         regional_nodes_for_branches=across_branch_nodes_in[i]
#         branch_all_nodes_after_deleting=remove_dummy_nodes(regional_nodes_for_branches, area_non_walk_df)
#         across_branch_nodes_after_deleting_out.append(branch_all_nodes_after_deleting)
#         
#     return across_branch_nodes_after_deleting_out
# 
# across_branch_nodes_real=get_across_branch_nodes_after_deleting(across_branch_nodes, area_non_walk_df)
# 
# =============================================================================

def order_node_one_branch(branch_all_nodes_input, area_non_walk_df, depot_node, G, longest_distance_from_depot):

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
    
    
    
    
    #zone generation
    

    def get_nodes_considered(longest_distance_from_depot):
        considered_nodes=[] 
        for i in range(len(node_distance_order_list_after)):
            this_branch=node_distance_order_list_after[i]
            j=0
            while j <len(this_branch):
                if this_branch[j]>longest_distance_from_depot:
                    break
                else:
                    j=j+1       
            considered_nodes.append(node_order_list_after[i][:j])
        
        return considered_nodes
    considered_nodes=get_nodes_considered(longest_distance_from_depot)


    
    return node_order_list_after, node_distance_order_list_after


def generate_node_to_node_distances(area_non_walk_df, depot_point,outer_bound_dist, not_reachable_dist):
    G_temp=ox.graph.graph_from_point(depot_point, dist=outer_bound_dist*3, dist_type='network', network_type='drive', simplify=False)
    
    first_column=[]
    first_node=0
    cur_orig=(area_non_walk_df.iloc[first_node]['pad_nodes_lat'], area_non_walk_df.iloc[first_node]['pad_nodes_lon'])
    cur_orig_node=ox.get_nearest_node(G_temp, cur_orig)
    for i in range(area_non_walk_df.shape[0]):
        if i==first_node:
            first_column.append(0)
        else:
            try:
                cur_dest=(area_non_walk_df.iloc[i]['pad_nodes_lat'], area_non_walk_df.iloc[i]['pad_nodes_lon'])
                cur_dest_node=ox.get_nearest_node(G_temp, cur_dest)
                cur_short_path_length=nx.shortest_path_length(G_temp, cur_orig_node, cur_dest_node, weight='length') 
                first_column.append(cur_short_path_length)
            except nx.NetworkXNoPath:
                first_column.append(not_reachable_dist)

    ntn_df = pd.DataFrame(first_column, columns=[first_node])
    
    
    for it in range(1,area_non_walk_df.shape[0]):
        cur_column=[]
        cur_node=it
        cur_orig=(area_non_walk_df.iloc[cur_node]['pad_nodes_lat'], area_non_walk_df.iloc[cur_node]['pad_nodes_lon'])
        cur_orig_node=ox.get_nearest_node(G_temp, cur_orig)
        for i in range(area_non_walk_df.shape[0]):
            if i==cur_node:
                cur_column.append(0)
            else:
                
                try:
                    cur_dest=(area_non_walk_df.iloc[i]['pad_nodes_lat'], area_non_walk_df.iloc[i]['pad_nodes_lon'])
                    cur_dest_node=ox.get_nearest_node(G_temp, cur_dest)
                    cur_short_path_length=nx.shortest_path_length(G_temp, cur_orig_node, cur_dest_node, weight='length') 
                    cur_column.append(cur_short_path_length)
                except nx.NetworkXNoPath:
                    cur_column.append(not_reachable_dist)

        ntn_df[cur_node] = cur_column
        print(cur_node)
    
    ntn_df.to_csv ('note_to_node_distances.csv', index=None)
    


    return ntn_df


def get_ntn_df_diff(not_reachable_dist):
    ntn_df_read = pd.read_csv ('note_to_node_distances.csv')
    ntn_df_diff=ntn_df_read.copy()
    for i in range(ntn_df_read.shape[0]):
        for j in range(i+1, ntn_df_read.shape[0]):
            if ntn_df_read.iloc[i][j]<not_reachable_dist and ntn_df_read.iloc[j][i]<not_reachable_dist:
                ntn_df_diff.iloc[i][j]=abs(ntn_df_read.iloc[i][j]-ntn_df_read.iloc[j][i])
                ntn_df_diff.iloc[j][i]=0
            else:
                ntn_df_diff.iloc[i][j]=0
                ntn_df_diff.iloc[j][i]=0

    return ntn_df_diff


def get_ntn_df_avg(not_reachable_dist, compensate_dist):
    ntn_df_read = pd.read_csv ('note_to_node_distances.csv')
    ntn_df_avg=ntn_df_read.copy()  
    for i in range(ntn_df_read.shape[0]):
        for j in range(i+1, ntn_df_read.shape[0]):
            if ntn_df_read.iloc[i][j]<not_reachable_dist and ntn_df_read.iloc[j][i]<not_reachable_dist:
                ntn_df_avg.iloc[i][j]=(ntn_df_read.iloc[i][j]+ntn_df_read.iloc[j][i])/2
                ntn_df_avg.iloc[j][i]=(ntn_df_read.iloc[i][j]+ntn_df_read.iloc[j][i])/2
            elif ntn_df_read.iloc[i][j]>not_reachable_dist:
                ntn_df_avg.iloc[i][j]=ntn_df_avg.iloc[j][i]+compensate_dist
            else:
                ntn_df_avg.iloc[j][i]==ntn_df_avg.iloc[i][j]+compensate_dist
    
    
    return ntn_df_avg

def sub_lists (l):
    lists = [[]]
    for i in range(len(l) + 1):
        for j in range(i):
            lists.append(l[j: i])
    return lists

# =============================================================================
# def powerset(iterable):
#     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#     s = list(iterable)
#     return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
# =============================================================================

def build_zones_one_branch(node_order_list_zone_branch, max_multual_dist, ntn_df_avg, demand_limit, area_non_walk_df):
    
    zone_this_branch=[]
    for i in range(len(node_order_list_zone_branch)):
        cur_node=node_order_list_zone_branch[i]
        potential_nodes=[cur_node]
        potential_ex_cur=[]
        
        for j in range(i+1, len(node_order_list_zone_branch)):
            test_node=node_order_list_zone_branch[j]
            if_test_in=1
            for k in range(len(potential_nodes)):
                in_node=potential_nodes[k]
                #print(test_node)
                #print(in_node)
                if ntn_df_avg.iloc[test_node][in_node]>max_multual_dist:
                    if_test_in=0
                    break           
            if if_test_in==1:
                potential_nodes.append(test_node)
                potential_ex_cur.append(test_node)
        
        potential_ex_sub=sub_lists(potential_ex_cur)
        #potential_ex_sub=powerset(potential_ex_cur)
        for l in range(len(potential_ex_sub)):
            test_potential_zone=[cur_node]
            test_potential_zone.extend(potential_ex_sub[l])
            total_zone_demand=0
            for m in range(len(test_potential_zone)):
                demand_node=test_potential_zone[m]
                total_zone_demand=total_zone_demand+area_non_walk_df.iloc[demand_node]['pad_nodes_all_pickup_amount']+area_non_walk_df.iloc[demand_node]['pad_nodes_all_dropoff_amount']
            if total_zone_demand<=demand_limit or len(test_potential_zone)==1:
                zone_this_branch.append(test_potential_zone)
                
    return zone_this_branch




def zone_generation_process(max_multual_dist, ntn_df_avg, demand_limit, across_branch_nodes_input, area_non_walk_df, depot_node, G, longest_distance_from_depot):
    
    zones_all_decisions=[]
    for i in range(len(across_branch_nodes_input)):
        cur_decision_branch=across_branch_nodes_input[i]
        node_order_list_after, node_distance_order_list_after=order_node_one_branch(cur_decision_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot)
        zones_this_decision=[]
        for j in range(len(node_order_list_after)):
            zone_this_branch_out=build_zones_one_branch(node_order_list_after[j], max_multual_dist, ntn_df_avg, demand_limit, area_non_walk_df)
            zones_this_decision.extend(zone_this_branch_out)
    
        zones_all_decisions.append(zones_this_decision)
        
    return zones_all_decisions




def get_final_decision_zone_list(all_decision_list_result_input):
    
    decision_zone_list=all_decision_list_result_input[0]
    for i in range(1, len(all_decision_list_result_input)):
        cur_decision_list=all_decision_list_result_input[i]
        for j in range(len(cur_decision_list)):
            cur_zone=cur_decision_list[j]
            if not cur_zone in decision_zone_list:
                decision_zone_list.append(cur_zone)
                
    return decision_zone_list


def check_nodes_covered(final_list_in):
    
    overall_count_list=[]
    for i in range(len(final_list_in)):
        for j in range(len(final_list_in[i])):
            cur_node=final_list_in[i][j]
            if cur_node not in overall_count_list:
                overall_count_list.append(cur_node)
    return overall_count_list
        
        
        
        
        
        
        