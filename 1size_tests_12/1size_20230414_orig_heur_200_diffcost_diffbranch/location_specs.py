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
from  shortest_path_tree import *
from  branch_decisions import *
from  order_node_zone_generation_orig import *
from  vrp_simulation import *
from  districting import *
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

##################################################################################################################
##################################################################################################################

cur_intersection_points=get_intersection_points_select(depot_point, outer_bound_dist,inner_bount_dist)

combined_shortest_dummy_nodes, combined_shortest_dummy_coord=combine_shortest_path_nodes(depot_node, depot_point, area_non_walk_df, G, cur_intersection_points)
shortest_path_routes=get_shortest_path_routes(G, combined_shortest_dummy_nodes, depot_node)
shortest_path_with_nodes, shortest_path_node_index, all_nodes_id=get_shortest_path_with_nodes(combined_shortest_dummy_nodes, shortest_path_routes)
tree_nodes, node_parent, node_children, list_lens= get_shortest_path_tree(shortest_path_node_index) 
branch_size, branch_all_nodes = branch_size_of_tree(0, tree_nodes, node_children)

##################################################################################################################

combined_shortest_dummy_coord, cur_intersection_points=filter_intersection_structure(area_non_walk_df, combined_shortest_dummy_coord, depot_point, tree_nodes, node_children)

combined_shortest_dummy_nodes, combined_shortest_dummy_coord=combine_shortest_path_nodes(depot_node, depot_point, area_non_walk_df, G, cur_intersection_points)
shortest_path_routes=get_shortest_path_routes(G, combined_shortest_dummy_nodes, depot_node)
shortest_path_with_nodes, shortest_path_node_index, all_nodes_id=get_shortest_path_with_nodes(combined_shortest_dummy_nodes, shortest_path_routes)
tree_nodes, node_parent, node_children, list_lens= get_shortest_path_tree(shortest_path_node_index)
branch_size, branch_all_nodes = branch_size_of_tree(0, tree_nodes, node_children)
##################################################################################################################

initial_num_branch=len(branch_all_nodes)
across_first_nodes, across_second_nodes, across_distance_fs, across_previous_branch_root, across_current_branch_root, across_previous_branch_split, across_branch_nodes=generate_all_branch_choices(tree_nodes, node_children, branch_all_nodes, area_non_walk_df, G, combined_shortest_dummy_coord, node_parent, branch_size)
##################################################################################################################

longest_distance_from_depot=outer_bound_dist
max_multual_dist=1700
not_reachable_dist=100000
compensate_dist=10000
max_zone_dem_input=200
ntn_df_avg=get_ntn_df_avg(not_reachable_dist, compensate_dist)
all_node_order_result, all_decision_list_result=combine_all_zones(max_zone_dem_input,max_multual_dist, ntn_df_avg, across_branch_nodes, initial_num_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot)
final_decision_zone_list=get_final_decision_zone_list(all_decision_list_result)    

##################################################################################################################

zone_data = {'total': ['final_decision_zone_list'],
        'length': [len(final_decision_zone_list)]}
zone_df = pd.DataFrame(zone_data)
zone_filename_time = "number_of_zones.csv"
zone_df.to_csv(zone_filename_time)

print("zone_generated.")
##################################################################################################################


def test_node_group_cost():
    
    cur_df = pd.DataFrame(columns=['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions','a_count_not_good_versions', 'a_count_not_good_versions_avg']) 
    for i in range(len(final_decision_zone_list)):
        cur_within=final_decision_zone_list[i]

        print(cur_within)
        cur_fleet_size=1
        
        this_df=fixRvarFS(0,area_non_walk_df, G, depot_node, longest_distance_from_depot, cur_fleet_size, cur_within)
        cur_df=cur_df.merge(this_df, on=['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions','a_count_not_good_versions', 'a_count_not_good_versions_avg'], how='outer') 
        cur_df = pd.DataFrame(cur_df)
        filename = "cost_cs_hanover_1size_heur.csv"
        cur_df.to_csv(filename)   
test_node_group_cost() 
    
    
cs_cost_read=pd.read_csv ('cost_cs_hanover_1size_heur.csv')
cs_cost_read=cs_cost_read.drop("Unnamed: 0",axis=1)
cs_cost_read=cs_cost_read.rename(columns={"avg_total_obj_cost_all_instances_with_pax_wait": "cs_value"})
cs_input=cs_cost_read['cs_value'].values




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
q_select_list=[]
zones_chosen=[]
for i in range(1,13):
    vehicle_ub=i
    seth_value,result_Vars_ysz, result_Vars_xsn, result_Vars_v, result_Vars_qs, obj_func_value=main_model_srd(cs_input, vehicle_ub, area_non_walk_df, all_node_order_result, final_decision_zone_list, all_decision_list_result)
    decision_nodes, selected_zones_result=post_process_data(cs_input, result_Vars_ysz, result_Vars_xsn, result_Vars_v, result_Vars_qs, all_node_order_result,area_non_walk_df, final_decision_zone_list)
    folium_map_result(depot_lat, depot_lon, decision_nodes, area_non_walk_df, depot_point, region_name, vehicle_ub, result_Vars_v)
    
    num_branches_here=result_Vars_qs.index(max(result_Vars_qs))+initial_num_branch
    q_select_list.append(num_branches_here)
    folium_map_result_zone(num_branches_here, depot_lat, depot_lon, selected_zones_result, area_non_walk_df, depot_point, region_name, vehicle_ub, result_Vars_v)
    zones_chosen.append(str(selected_zones_result))

    num_vehicle_list.append(i)
    obj_function_value_list.append(obj_func_value)
    
    evaluate_cost_single=evaluate_zones(selected_zones_result)+seth_value
    
    
    evaluation_cost.append(evaluate_cost_single)
    num_vehicle_used_list.append(result_Vars_v)
    
    list_of_tuples = list(zip(num_vehicle_list, obj_function_value_list,evaluation_cost, num_vehicle_used_list,q_select_list, zones_chosen))
    df_obj = pd.DataFrame(list_of_tuples, columns = ['num_vehicle_max', 'obj_function_value_list','evaluation_cost','num_vehicle_used','q_select_list','zones_chosen'])
   
    df_obj.to_csv("evaluation_costs_1size_orig_heur_"+str(max_zone_dem_input)+".csv")



##################################################################################################################
##################################################################################################################
##################################################################################################################


