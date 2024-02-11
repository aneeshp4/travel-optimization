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

def order_node_and_zone_generation_one_branch(max_multual_dist, ntn_df_avg, branch_all_nodes_input, area_non_walk_df, depot_node, G, longest_distance_from_depot):

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


    def get_all_sim_sets():
        branches=[]
        num_nodes_each_branch=[] 
        for i in range(len(considered_nodes)):
            this_branch_len=len(considered_nodes[i])
            num_nodes_each_branch.append(this_branch_len)
            branches.append(i)        
        return branches, num_nodes_each_branch

                    
    branches, num_nodes_each_branch=get_all_sim_sets()


    def get_location_index(branches, num_nodes_each_branch):
        point_locations={}
        point_demand={}
        for i in range(len(branches)):
            point_locations[i]=considered_nodes[i]
            cur_all_demand=[]
            for j in range(0,num_nodes_each_branch[i]):
                cur_total_dem=area_non_walk_df['pad_nodes_all_pickup_amount'][considered_nodes[i][j]]+area_non_walk_df['pad_nodes_all_dropoff_amount'][considered_nodes[i][j]]
                cur_all_demand.append(cur_total_dem)
            point_demand[i]=cur_all_demand
        return point_locations, point_demand

    point_locations, point_demand=get_location_index(branches, num_nodes_each_branch)

    def get_reverse_point_locations(point_locations):
        rev_point_location=[]
        for i in range(len(point_locations)):
            cur_branch_loc=point_locations[i]
            cur_reverse=cur_branch_loc[::-1]
            rev_point_location.append(cur_reverse)
        return rev_point_location
            
    rev_point_location=get_reverse_point_locations(point_locations)

    def possible_breaks_along_branch(branch_index, point_locations_l, max_zone_dem):
        
        cur_branch_loc=point_locations_l[branch_index]
        cur_branch_dem=point_demand[branch_index]
        cur_num_point=len(cur_branch_loc)
        
        next_break_point={}
        for i in range(0,cur_num_point):
            cur_point_dem=cur_branch_dem[i]
            dem_so_far=cur_point_dem
            if dem_so_far>max_zone_dem:
                end_point_here=1
                next_break_point[i]=[i]
            else:
                end_point_here=0 
                
            cur_list=[]
            next_point_index=i
            cur_total_dem=cur_point_dem
            
            while end_point_here==0 and next_point_index+1<cur_num_point:
                next_point_index=next_point_index+1

                next_point_dem=cur_branch_dem[next_point_index]
                cur_total_dem=cur_total_dem+next_point_dem
                
                if cur_total_dem>max_zone_dem:
                    if len(cur_list)==0:
                        cur_list.append(next_point_index-1)
                    next_break_point[i]=cur_list
                    end_point_here=1
                elif cur_total_dem==max_zone_dem:
                    if next_point_index<cur_num_point:
                        cur_list.append(next_point_index)
                    next_break_point[i]=cur_list
                    end_point_here=1
                else:
                    if len(cur_list)==0:
                        cur_list.append(next_point_index-1)   
                    if next_point_index<cur_num_point:
                        cur_list.append(next_point_index)

                    next_break_point[i]=cur_list                
                if next_point_index+1>=cur_num_point:
                    end_point_here=1
            next_break_point[cur_num_point-1]=[cur_num_point-1]
        return next_break_point










    max_zone_dem=100
    min_zone_dem=1
    def get_all_demand_groups():
        all_sz=[]
        all_sz_demand=[]
        for i in range(len(branches)):
            cur_next_break_point=possible_breaks_along_branch(i, point_locations, max_zone_dem)
            #print("forward_"+ str(i)+": "+str(cur_next_break_point))
            cur_rev_break_point=possible_breaks_along_branch(i, rev_point_location, max_zone_dem)  
            #print("backward_"+ str(i)+": "+str(cur_rev_break_point))
            cur_branch_sz=[]
            cur_branch_sz_demand=[]
            
            cur_all_point=[]
            for j in range(len(cur_next_break_point)):  
                cur_all_point_index=cur_next_break_point[j]
                cur_all_point=[]
                mutual_distance_exceed=0
                if len(cur_next_break_point[j])>1:              
                    cur_all_point_array=point_locations[i][cur_next_break_point[j][0]:(cur_next_break_point[j][-1]+1)]
                else:
                    cur_all_point_array=point_locations[i][cur_next_break_point[j][0]:(cur_next_break_point[j][0]+1)]            
                
                for ml in range(len(cur_all_point_index)):
                    cur_all_point.append(point_locations[i][j:(cur_next_break_point[j][ml]+1)])

                for cp in range(len(cur_all_point)):
                    cur_all_demand=0
                    for k in range(len(cur_all_point[cp])):
                        cur_all_demand=cur_all_demand+area_non_walk_df['pad_nodes_all_pickup_amount'][cur_all_point[cp][k]]+area_non_walk_df['pad_nodes_all_dropoff_amount'][cur_all_point[cp][k]]
                        
                    if cur_all_demand>=min_zone_dem:
                        dont_include=0
                        for count in range(len(cur_branch_sz)): 
                            
                            if set(cur_all_point[cp])==set(cur_branch_sz[count]):
                                dont_include=1
                          
                        if dont_include==0:   
                            cur_branch_sz.append(cur_all_point[cp])
                            cur_branch_sz_demand.append(cur_all_demand)


            for l in range(len(cur_rev_break_point)):
                cur_all_point_index=cur_rev_break_point[l]
                cur_all_point=[]
                if len(cur_rev_break_point[l])>1:
                    cur_all_point_array=rev_point_location[i][cur_rev_break_point[l][0]:(cur_rev_break_point[l][-1]+1)]
                else:
                    cur_all_point_array=rev_point_location[i][cur_rev_break_point[l][0]:(cur_rev_break_point[l][0]+1)]            
     
                for ml in range(len(cur_all_point_index)):
                    cur_all_point.append(rev_point_location[i][l:(cur_rev_break_point[l][ml]+1)])
                
                
                for cp in range(len(cur_all_point)):
                    cur_all_demand=0
                    for kl in range(len(cur_all_point[cp])):
                        cur_all_demand=cur_all_demand+area_non_walk_df['pad_nodes_all_pickup_amount'][cur_all_point[cp][kl]]+area_non_walk_df['pad_nodes_all_dropoff_amount'][cur_all_point[cp][kl]]
                        
                    if cur_all_demand>=min_zone_dem:
                        dont_include=0
                        for count in range(len(cur_branch_sz)):  
                            if set(cur_all_point[cp])==set(cur_branch_sz[count]):
                                dont_include=1
                                
                        
                        if dont_include==0:   
                            cur_branch_sz.append(cur_all_point[cp])
                            cur_branch_sz_demand.append(cur_all_demand)

            all_sz.append(cur_branch_sz)
            all_sz_demand.append(cur_branch_sz_demand)        
            
        
        return all_sz, all_sz_demand
    all_sz, all_sz_demand=get_all_demand_groups()
    
    return node_order_list_after, node_distance_order_list_after, all_sz, all_sz_demand



##################################################################################
##################################################################################

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

def check_within_mutual_distance(this_zone_nodes, ntn_df_avg, max_multual_dist):
    exceed_distance=0
    for i in range(len(this_zone_nodes)-1):
        for j in range(i+1, len(this_zone_nodes)):
            first_int=int(this_zone_nodes[i])
            second_int=int(this_zone_nodes[j])
            cur_distance=ntn_df_avg.iloc[first_int][second_int]
            if cur_distance>max_multual_dist:
                exceed_distance=1
                break
    
    return exceed_distance
##################################################################################
##################################################################################



def combine_all_zones(max_multual_dist, ntn_df_avg, across_branch_nodes_input, start_num_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot):
    
    def get_sz_to_list(all_sz_d):
        all_sz_to_list=[]
        for i in range(len(all_sz_d)):
            cur_branch_sz=all_sz_d[i]
            for j in range(len(cur_branch_sz)):
                cur_sz=cur_branch_sz[j]
                all_sz_to_list.append(cur_sz)
        return all_sz_to_list
    
    
    all_decision_lists=[]
    all_decision_lists_dict={}
    all_node_order=[]
    all_node_distance=[]
    for i in range(len(across_branch_nodes_input)):
        cur_decision_branch=across_branch_nodes_input[i]
        node_order_list_decision, node_distance_order_list_decision, all_sz_decision, all_sz_demand_decision=order_node_and_zone_generation_one_branch(max_multual_dist, ntn_df_avg, cur_decision_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot)
        all_node_order.append(node_order_list_decision)
        all_node_distance.append(node_distance_order_list_decision)
        decision_list=get_sz_to_list(all_sz_decision)
        

        zone_filtered=[]
        for j in range(len(decision_list)):
            cur_zone=decision_list[j]
            exceed_check=check_within_mutual_distance(cur_zone, ntn_df_avg, max_multual_dist)
            if exceed_check==0 or len(cur_zone)==1:
                zone_filtered.append(cur_zone) 
            
            
        all_decision_lists.append(zone_filtered)
        #all_decision_lists_dict[i]=decision_list
        

    return all_node_order, all_decision_lists



def get_final_decision_zone_list(all_decision_list_result_input):
    
    decision_zone_list=all_decision_list_result_input[0]
    for i in range(1, len(all_decision_list_result_input)):
        cur_decision_list=all_decision_list_result_input[i]
        for j in range(len(cur_decision_list)):
            cur_zone=cur_decision_list[j]
            if not cur_zone in decision_zone_list:
                decision_zone_list.append(cur_zone)
                
    return decision_zone_list




