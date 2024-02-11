#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:00:57 2023

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
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def further_division_by_demand_kmean(ntn_df_avg,max_multual_dist, high_demand_zone, demand_limit):
    

    after_divide_zones=[]
    
    node_amount=high_demand_zone.shape[0]
    if node_amount==1:
        after_divide_zones=[high_demand_zone['node_index'].tolist()]

    else:
    
        number_of_zones=2
        data=high_demand_zone
        
        x=data.iloc[:,3:5]
        kmeans = KMeans(number_of_zones)
        kmeans.fit(x)
        identified_clusters = kmeans.fit_predict(x)
        data_with_clusters = data.copy()
        data_with_clusters['network_cluster_km'] = identified_clusters 
        km_cluster_value_count=data_with_clusters['network_cluster_km'].value_counts()
        km_cluster_group_num=len(km_cluster_value_count)
        
        for km_index in range(0, km_cluster_group_num):
            index_km_group=data_with_clusters.index[data_with_clusters['network_cluster_km']==km_index].tolist()
    
            cur_list=data_with_clusters.iloc[index_km_group]['node_index'].tolist()
            #print(cur_list)
            
            cur_list_demand=0
            for cd in range(len(cur_list)):
                cur_list_demand=cur_list_demand+data['pad_nodes_all_pickup_amount'][index_km_group[cd]]+data['pad_nodes_all_dropoff_amount'][index_km_group[cd]]
                
            
            exceeded_dist=check_within_mutual_distance(cur_list, ntn_df_avg, max_multual_dist)
            #print(cur_list_demand)
            if cur_list_demand>demand_limit or exceeded_dist==1:
                to_subdivide=data_with_clusters.iloc[index_km_group][:]
                to_subdivide.reset_index(drop=True, inplace=True)
                
                #print(to_subdivide)
                divided_sub_zone=further_division_by_demand_kmean(ntn_df_avg,max_multual_dist,to_subdivide, demand_limit)
                #print(divided_sub_zone)
                
                after_divide_zones.extend(divided_sub_zone)
    
            else:
                after_divide_zones.append(cur_list)

    #print(after_divide_zones)
    #exit()
    
    
    return after_divide_zones

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
def check_further_division(ntn_df_avg, max_multual_dist, demand_limit, new_cur_dbscan_rein):
    km_result_groups=[]
    node_demand=sum(new_cur_dbscan_rein['pad_nodes_all_pickup_amount'])+sum(new_cur_dbscan_rein['pad_nodes_all_dropoff_amount'])
    node_amount=new_cur_dbscan_rein.shape[0]

    cur_zone=[new_cur_dbscan_rein.iloc[:]['node_index'].tolist()]
    exceeded_dist=check_within_mutual_distance(cur_zone, ntn_df_avg, max_multual_dist)
        
    if (node_demand<=demand_limit and exceeded_dist==0) or node_amount==1:
        km_result_groups=[new_cur_dbscan_rein.iloc[:]['node_index'].tolist()]
      
    else:
        km_result_groups=further_division_by_demand_kmean(ntn_df_avg, max_multual_dist, new_cur_dbscan_rein, demand_limit)
        
    return km_result_groups

def dbkm_for_one_branch(ntn_df_avg, max_multual_dist, demand_limit,nodes_for_this_branch, area_non_walk_df, G, depot_point):
    #print(nodes_for_this_branch)
    #print(len(nodes_for_this_branch))
    zones_this_branch=[]
    cur_dbscan_df=area_non_walk_df[area_non_walk_df.index.isin(nodes_for_this_branch)]
    cur_dbscan_rein=cur_dbscan_df.reset_index()
    cur_dbscan_rein=cur_dbscan_rein.rename(columns={"index":"node_index"})
    nn_list=[]
    for i in range(cur_dbscan_rein.shape[0]):
        nn_list.append(ox.get_nearest_node(G, (cur_dbscan_rein['pad_nodes_lat'][i], cur_dbscan_rein['pad_nodes_lon'][i]), method='balltree'))
    cur_dbscan_rein['nn']=nn_list
    max_bound_dist=5000
    G_max=ox.graph.graph_from_point(depot_point, dist=max_bound_dist, dist_type='bbox', network_type='drive', simplify=False)

    nodes_unique = pd.Series(cur_dbscan_rein['nn'].unique())
    nodes_unique.index = nodes_unique.values
    def network_distance_matrix(u, G_max, vs=nodes_unique):
        dists = [nx.dijkstra_path_length(G_max, source=u, target=v, weight='length') for v in vs]
        return pd.Series(dists, index=vs)


    node_dm = nodes_unique.apply(network_distance_matrix, G_max=G_max)
    node_dm = node_dm.astype(int)
    ndm = node_dm.reindex(index=cur_dbscan_rein['nn'], columns=cur_dbscan_rein['nn'])

    eps_set=500
    db = DBSCAN(eps=eps_set, min_samples=2, metric='precomputed')
    cur_dbscan_rein['network_cluster'] = db.fit_predict(ndm)
    
    
    
    dbscan_cluster_value_count=cur_dbscan_rein['network_cluster'].value_counts()
    dbscan_cluster_group_num=len(dbscan_cluster_value_count)

    index_minus_one=cur_dbscan_rein.index[cur_dbscan_rein['network_cluster']==-1].tolist()
    if len(index_minus_one)==0:
        dbscan_cluster_group_num=dbscan_cluster_group_num+1
    for index in index_minus_one:
        if len([cur_dbscan_rein['node_index'][index]])>0:
            zones_this_branch.append([cur_dbscan_rein['node_index'][index]])
    for gi in range(0,dbscan_cluster_group_num-1):
        new_cur_dbscan_rein=cur_dbscan_rein.loc[cur_dbscan_rein['network_cluster'] == gi]
        new_cur_dbscan_rein=new_cur_dbscan_rein.reset_index()
        new_cur_dbscan_rein=new_cur_dbscan_rein.drop(columns=["index"])
        
        km_result_groups_out=check_further_division(ntn_df_avg,max_multual_dist, demand_limit,new_cur_dbscan_rein)
        if len(zones_this_branch)>0:
            zones_this_branch.extend(km_result_groups_out)
        else:
            zones_this_branch=km_result_groups_out
    
    

        
    overall_count_list=[]
    for i in range(len(zones_this_branch)):
        for j in range(len(zones_this_branch[i])):
            cur_node=zones_this_branch[i][j]
            if cur_node not in overall_count_list:
                overall_count_list.append(cur_node)
    
    #print(overall_count_list)
    #print(len(overall_count_list))
    return zones_this_branch



def zones_for_cur_choice(ntn_df_avg,max_multual_dist, demand_limit, nodes_branches_this_choice, area_non_walk_df, G, depot_point):
    zone_list_cur_choice=[]
    
    for i in range(len(nodes_branches_this_choice)):
        cur_branch_nodes=nodes_branches_this_choice[i]
        cur_branch_nodes_adjust = [bn-1  for bn in cur_branch_nodes]
        #print("for another branch: "+str(i))
        zones_this_branch_out=dbkm_for_one_branch(ntn_df_avg,max_multual_dist, demand_limit, cur_branch_nodes_adjust, area_non_walk_df, G, depot_point)
        zone_list_cur_choice.extend(zones_this_branch_out)
    
    return zone_list_cur_choice


def remove_dummy_nodes(branch_all_nodes_input, area_non_walk_df):
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

def zones_all_choices(demand_limit, ntn_df_avg, max_multual_dist,across_branch_nodes_in, area_non_walk_df, G, depot_point):
    
    zone_list_cur_choice_save={}
    #branch_all_nodes_after_collect={}
    for i in range(len(across_branch_nodes_in)):
        regional_nodes_for_branches=across_branch_nodes_in[i]
        branch_all_nodes_after_deleting=remove_dummy_nodes(regional_nodes_for_branches, area_non_walk_df)
        #branch_all_nodes_after_collect[i]=branch_all_nodes_after_deleting
        zone_list_cur_choice_out= zones_for_cur_choice(ntn_df_avg,max_multual_dist,demand_limit, branch_all_nodes_after_deleting, area_non_walk_df, G, depot_point)
        
        zone_filtered=[]
        for j in range(len(zone_list_cur_choice_out)):
            cur_zone=zone_list_cur_choice_out[j]
            exceed_check=check_within_mutual_distance(cur_zone, ntn_df_avg, max_multual_dist)
            if exceed_check==0 or len(cur_zone)==1:
                zone_filtered.append(cur_zone)
                
        zone_list_cur_choice_save[i]=zone_filtered   
        
    
    return zone_list_cur_choice_save



def plot_zones(zone_list_cur_choice_save_in, area_non_walk_df, depot_point, depot_lat, depot_lon):
    
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    for z in range(len(zone_list_cur_choice_save_in)):
        number_of_zones=len(zone_list_cur_choice_save_in[z])
        point_colors=get_colors(number_of_zones)
        
        m = folium.Map(location=[depot_lat, depot_lon], zoom_start=2)
        for i in range(0, number_of_zones):
            cur_color=point_colors[i]
            for j in range(len(zone_list_cur_choice_save_in[z][i])):
                node_index_j=zone_list_cur_choice_save_in[z][i][j]
                #print(node_index_j)
                cur_loc=(area_non_walk_df['pad_nodes_lat'][node_index_j],area_non_walk_df['pad_nodes_lon'][node_index_j] )
                folium.CircleMarker(location=cur_loc,radius=5,fill=True, popup=cur_loc, fill_color=cur_color, color=cur_color,  fill_opacity=1).add_to(m)

            
        folium.Marker(location=depot_point,radius=10,fill=True, popup=cur_loc, fill_color='black', color='white', fill_opacity=1).add_to(m)

        map_name="hanover_inn"+"_dbkm_"+str(z+2)+"_branhes_"+str(len(point_colors))+"_zones.html"

        m.save(map_name)
    
    
 
    
    
def check_nodes_covered(final_list_in):
    
    overall_count_list=[]
    for i in range(len(final_list_in)):
        for j in range(len(final_list_in[i])):
            for k in range(len(final_list_in[i][j])):
                cur_node=final_list_in[i][j][k]
                if cur_node not in overall_count_list:
                    overall_count_list.append(cur_node)
    return overall_count_list




def get_final_decision_zone_list(all_decision_list_result_input):
    
    decision_zone_list=all_decision_list_result_input[0]
    for i in range(1, len(all_decision_list_result_input)):
        cur_decision_list=all_decision_list_result_input[i]
        for j in range(len(cur_decision_list)):
            cur_zone=cur_decision_list[j]
            if not cur_zone in decision_zone_list:
                decision_zone_list.append(cur_zone)
                
    return decision_zone_list

def check_final_nodes_covered(final_list_in):
    
    overall_count_list=[]
    for i in range(len(final_list_in)):
        for j in range(len(final_list_in[i])):
            cur_node=final_list_in[i][j]
            if cur_node not in overall_count_list:
                overall_count_list.append(cur_node)
    return overall_count_list

