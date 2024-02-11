#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:11:04 2022

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
from itertools import combinations, product, chain
import gurobipy as gp
from gurobipy import GRB
from  vrp_simulation import *
from districting_small import *
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

start_timer = time.time()

###########################################
depot_lat=43.70235275
depot_lon=-72.28830002
region_name="hanover_inn"
###########################################
def setup_parameters():
    global outer_bound_dist, demand_file_name, depot_point, G, depot_node,inner_bount_dist, G_inner

    #outer_bound_dist=3218
    outer_bound_dist=1500
    inner_bount_dist=800
    
    depot_point=(depot_lat, depot_lon)
    G=ox.graph.graph_from_point(depot_point, dist=outer_bound_dist, dist_type='bbox', network_type='drive', simplify=False)
    depot_node=ox.get_nearest_node(G, depot_point) 
    
    G_inner = ox.graph_from_point(depot_point, dist=inner_bount_dist, dist_type='bbox', network_type='drive', simplify=False)

    return outer_bound_dist, depot_point, G, depot_node, inner_bount_dist, G_inner

outer_bound_dist, depot_point, G, depot_node, inner_bount_dist, G_inner=setup_parameters()

upper_valley_nodes_name="overall_demand_instance_v0.csv"
upper_valley_nodes_df=pd.read_csv(upper_valley_nodes_name)
upper_valley_nodes_df=upper_valley_nodes_df.rename(columns = {'Unnamed: 0':'full_index'})
##################################################################################################################

def get_area_nodes(region_name, upper_valley_nodes_df):
    demand_df=upper_valley_nodes_df

    area_all_nodes=[]
    area_non_walk_nodes=[]    
    for i in range(demand_df.shape[0]):
        cur_point=(demand_df.iloc[i]['pad_nodes_lat'], demand_df.iloc[i]['pad_nodes_lon'])
        cur_node=ox.get_nearest_node(G, cur_point) 
        depot_node=ox.get_nearest_node(G, depot_point) 
        
        if G.has_node(cur_node):
            try:            
                cur_short_path_length=nx.shortest_path_length(G, depot_node, cur_node, weight='length') 
                if cur_short_path_length<outer_bound_dist:
                    distance_within=1
                else:
                    distance_within=0
            except nx.NetworkXNoPath:
                distance_within=0

        node_in_area=G.has_node(cur_node) and distance_within
        node_in_walk=G_inner.has_node(cur_node)
        node_in_non_walk=node_in_area and (not node_in_walk)
        
        if node_in_area:
            area_all_nodes.append(i)
        if node_in_non_walk:
            area_non_walk_nodes.append(i)
            
        
    area_all_df=demand_df.iloc[area_all_nodes]
    area_non_walk_df=demand_df.iloc[area_non_walk_nodes]

    area_all_df_name="area_all_df_"+region_name+".csv"
    area_non_walk_df_name="area_non_walk_df_"+region_name+".csv"
    
    #area_all_df.to_csv(area_all_df_name)
    area_non_walk_df.to_csv(area_non_walk_df_name)
        
    
    return area_all_df, area_non_walk_df

area_all_df, area_non_walk_df=get_area_nodes(region_name, upper_valley_nodes_df)

# =============================================================================
# cur_point=(area_non_walk_df.iloc[12]['pad_nodes_lat'], area_non_walk_df.iloc[12]['pad_nodes_lon'])
# cur_node=ox.get_nearest_node(G, cur_point)
# depot_node=ox.get_nearest_node(G, depot_point) 
# nx.shortest_path_length(G, depot_node, cur_node, weight='length')
# =============================================================================
##################################################################################################################

area_non_walk_df=area_non_walk_df.reset_index()
area_non_walk_df=area_non_walk_df.drop(columns=['index'])


m = folium.Map(location=[depot_lat, depot_lon], zoom_start=2)
for i in range(area_non_walk_df.shape[0]):
    cur_loc=(area_non_walk_df['pad_nodes_lat'][i], area_non_walk_df['pad_nodes_lon'][i])
    folium.CircleMarker(location=cur_loc,radius=5,fill=True, popup=cur_loc, fill_color='green', color='black', fill_opacity=1).add_to(m)
folium.Marker(location=depot_point,radius=10,fill=True, popup=depot_point, fill_color='black', color='black', fill_opacity=1).add_to(m)

map_name=region_name+"_1size_test_start"+".html"
m.save(map_name) 



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

nodes_small=[i for i in range(area_non_walk_df.shape[0])]
zones_small=powerset(nodes_small)
for i in range(len(zones_small)):
    zones_small[i]=list(zones_small[i])
  
    
demand_limit=200
within_demand_zones=[]
for i in range(len(zones_small)):
    cur_zone=zones_small[i]
    cur_demand=0
    for j in range(len(cur_zone)):
        cur_demand=cur_demand+area_non_walk_df['pad_nodes_all_pickup_amount'][cur_zone[j]]+area_non_walk_df['pad_nodes_all_dropoff_amount'][cur_zone[j]]
    if cur_demand<demand_limit:
        within_demand_zones.append(cur_zone)
    

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

max_multual_dist=1700
not_reachable_dist=100000
compensate_dist=10000
ntn_df_avg=get_ntn_df_avg(not_reachable_dist, compensate_dist)
mutual_filtered=[]
for i in range(len(within_demand_zones)):
    cur_zone=within_demand_zones[i]
    if_exceeded=check_within_mutual_distance(cur_zone, ntn_df_avg, max_multual_dist)
    if if_exceeded==0 or len(cur_zone)==1:
        mutual_filtered.append(cur_zone)

final_decision_zone_list=mutual_filtered

longest_distance_from_depot=outer_bound_dist
def test_node_group_cost():
    
    cur_df = pd.DataFrame(columns=['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions','a_count_not_good_versions', 'a_count_not_good_versions_avg']) 
    for i in range(len(final_decision_zone_list)):
        cur_within=final_decision_zone_list[i]

        print(cur_within)
        cur_fleet_size=1
        
        this_df=fixRvarFS(0,area_non_walk_df, G, depot_node, longest_distance_from_depot, cur_fleet_size, cur_within)
        cur_df=cur_df.merge(this_df, on=['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions','a_count_not_good_versions', 'a_count_not_good_versions_avg'], how='outer') 
        cur_df = pd.DataFrame(cur_df)
        filename = "cost_cs_hanover_1size_exact.csv"
        cur_df.to_csv(filename)   
test_node_group_cost() 
    
    
cs_cost_read=pd.read_csv ('cost_cs_hanover_1size_exact.csv')
cs_cost_read=cs_cost_read.drop("Unnamed: 0",axis=1)
cs_cost_read=cs_cost_read.rename(columns={"avg_total_obj_cost_all_instances_with_pax_wait": "cs_value"})
cs_input=cs_cost_read['cs_value'].values


cF=80

def small_districting_model(cs_input,nodes_small, zones_small, vehicle_upper_bound):
    N_set=nodes_small
    Z_set=[i for i in range(len(cs_input))]
    
    scale_ratio_cost=30
    c_n={}
    for j in range(len(N_set)):
        c_n[j]=scale_ratio_cost*(area_non_walk_df['pad_nodes_all_pickup_amount'][j]+area_non_walk_df['pad_nodes_all_dropoff_amount'][j])
    
    n_in_z={}
    for j in N_set:
        n_in_z[j]=[]
        for k in range(len(zones_small)):
            if j in zones_small[k]:
                n_in_z[j].append(k)
    
    
    model=gp.Model('small_solver_selection')
    y_z = model.addVars(Z_set, vtype=GRB.BINARY, name='y_z')
    x_n = model.addVars(N_set, vtype=GRB.BINARY, name='x_n')
    v = model.addVar(lb=0,vtype=GRB.INTEGER,name='v')
    
    
    model.ModelSense=GRB.MINIMIZE
    model.setObjective(sum(cs_input[i]*y_z[i] for i in Z_set)+sum(c_n[i]*(1-x_n[i]) for i in N_set) + cF*v)
    
    sethterms=sum(c_n[i]*(1-x_n[i]) for i in N_set) + cF*v
    
    
    model.addConstr(sum(y_z[i] for i in Z_set) <= v)

    for j in N_set: 
        model.addConstr(sum(y_z[k] for k in n_in_z[j]) >= x_n[j])
        model.addConstr(sum(y_z[k] for k in n_in_z[j]) <= x_n[j])

    model.addConstr(v<=vehicle_upper_bound)
    
    model.optimize()
    model.printAttr('X')
    
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)
    
    result_v=[]
    for name, val in zip(names, values):
        result_v.append(int(val))
        

    
    result_Vars_yz=[0 for z in Z_set]
    for z in Z_set:
        wx=int(result_v[z])
        result_Vars_yz[z]=wx  
            
    
            
    result_Vars_xn=[0 for n in N_set]
    before1=len(Z_set)
    for n in N_set:
        wx=int(result_v[before1+n])
        result_Vars_xn[n]=wx   
            
            
    result_Vars_v=0
    before2=before1+len(N_set)
    wx=int(result_v[before2])
    result_Vars_v=wx

    indices = [i for i, x in enumerate(result_Vars_yz) if x == 1]  
    selected_zone=[zones_small[i] for i in indices]

    obj = model.getObjective().getValue()
    seth_value=sethterms.getValue()
    
    return result_Vars_yz, result_Vars_xn, result_Vars_v, selected_zone, obj,seth_value


def evaluate_zones(selected_zones_test):
    
        
    cur_df = pd.DataFrame(columns=['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions', 'a_count_not_good_versions', 'a_count_not_good_versions_avg']) 
    for i in range(len(selected_zones_test)):
        cur_within=selected_zones_test[i]

        print(cur_within)
        cur_fleet_size=1
        
        this_df=fixRvarFS(1, area_non_walk_df, G, depot_node, longest_distance_from_depot, cur_fleet_size, cur_within)
        cur_df=cur_df.merge(this_df, on=['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions','a_count_not_good_versions', 'a_count_not_good_versions_avg'], how='outer') 
        cur_df = pd.DataFrame(cur_df)
    total_cost_report=cur_df['avg_total_obj_cost_all_instances_with_pax_wait'].sum()

    return total_cost_report

num_vehicle_list=[]
obj_function_value_list=[]
evaluation_cost=[]
num_vehicle_used_list=[]
evaluation_demand=[]
for i in range(1,13):
    vehicle_ub=i
    
    result_Vars_yz, result_Vars_xn, result_Vars_v, selected_zone, obj,seth_value=small_districting_model(cs_input,nodes_small,final_decision_zone_list, vehicle_ub)

    folium_map_result(depot_lat, depot_lon, selected_zone, area_non_walk_df, depot_point, region_name, vehicle_ub, result_Vars_v)

    num_vehicle_list.append(i)
    obj_function_value_list.append(obj)
    
    evaluate_cost_single=evaluate_zones(selected_zone)+seth_value
    
    eval_demands=[]
    for i in range(len(selected_zone)):
        cur_demand=0
        this_zone=selected_zone[i]
        for j in range(len(this_zone)):
            cur_demand=cur_demand+area_non_walk_df['pad_nodes_all_pickup_amount'][this_zone[j]]+area_non_walk_df['pad_nodes_all_dropoff_amount'][this_zone[j]]
        eval_demands.append(cur_demand)
        
    evaluation_demand.append(str(eval_demands))
    evaluation_cost.append(evaluate_cost_single)
    num_vehicle_used_list.append(result_Vars_v)
    
    list_of_tuples = list(zip(num_vehicle_list, obj_function_value_list,evaluation_cost, num_vehicle_used_list, evaluation_demand))
    df_obj = pd.DataFrame(list_of_tuples, columns = ['num_vehicle_max', 'obj_function_value_list','evaluation_cost','num_vehicle_used','evaluation_demand'])
    
    df_obj.to_csv("evaluation_costs_1size_absolute_exact_"+str(demand_limit)+".csv")



##################################################################################################################
##################################################################################################################
##################################################################################################################


