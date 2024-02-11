#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:33:12 2022

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



##################################################################################################################
def get_intersection_points_p(depot_point,inter_dist_radius):
    #cf = '["highway"~"motorway|motorway_link"]'
    cf_search = '["highway"~"motorway|motorway_link|primary|primary_link|secondary|secondary_lisk|tertiary|residential"]'  
    G_inter=ox.graph_from_point(depot_point, dist=inter_dist_radius, dist_type='network', network_type='drive', simplify=False, custom_filter=cf_search)
    G3 = ox.simplify_graph(G_inter)
    G3 = ox.projection.project_graph(G3, to_crs=4326)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G3)
    base_map = ox.plot_graph_folium(G3)
    
    gdf_nodes_y=[]
    gdf_nodes_x=[]
    #nc = ["r" if ox.simplification._is_endpoint(G3, node, strict=False) else "y" for node in G3.nodes()]

    for i in range(len(gdf_nodes)):
        gdf_nodes_y.append(gdf_nodes['y'].iloc[i])
        gdf_nodes_x.append(gdf_nodes['x'].iloc[i])
        
    for i,j in zip(list(gdf_nodes_y),list(gdf_nodes_x)):
        folium.CircleMarker((i,j), fill_color = "r").add_to(base_map)
    #base_map.save('intersection_points_map_'+region_name+'.html')
   
    gdf_nodes_coord=[]
    for i in range(len(gdf_nodes_y)):
        gdf_nodes_coord.append((gdf_nodes_y[i],gdf_nodes_x[i]))
      
    return gdf_nodes_coord, G3

def get_intersection_points_select(depot_point, outer_bound_dist,inner_bount_dist):
    
    gdf_nodes_coord_all, graph_all=get_intersection_points_p(depot_point,outer_bound_dist)
    gdf_nodes_coord_small, graph_small=get_intersection_points_p(depot_point,inner_bount_dist*8)
    
    cur_intersection_points=[]
    for i in range(len(gdf_nodes_coord_small)):
        if gdf_nodes_coord_small[i] in gdf_nodes_coord_all:
            cur_intersection_points.append(gdf_nodes_coord_small[i])

    base_map = ox.plot_graph_folium(graph_small)
    for point in cur_intersection_points:
        folium.CircleMarker(point, fill_color = "r").add_to(base_map)
        
    #base_map.save('intersection_points_map_'+region_name+'_v1.html')

    print(len(cur_intersection_points))

    return cur_intersection_points


##################################################################################################################


def combine_shortest_path_nodes(depot_node, depot_point, area_non_walk_df, G, cur_intersection_points):
    all_nodes_id=[]
    all_nodes_id.append(depot_node)
    all_nodes_coord=[]
    all_nodes_coord.append(depot_point)
    for i in range(area_non_walk_df.shape[0]):   
        cur_point=(area_non_walk_df.iloc[i]['pad_nodes_lat'], area_non_walk_df.iloc[i]['pad_nodes_lon'])
        cur_node=ox.get_nearest_node(G, cur_point) 
        all_nodes_id.append(cur_node) 
        all_nodes_coord.append(cur_point)
        
    for i in range(len(cur_intersection_points)):
        cur_node=ox.get_nearest_node(G, cur_intersection_points[i]) 
        all_nodes_id.append(cur_node)
        all_nodes_coord.append(cur_intersection_points[i])
        
    return all_nodes_id, all_nodes_coord


def get_shortest_path_routes(G, combined_shortest_dummy_nodes, depot_node):
    shortest_path_routes=[]
    nodes, edges = ox.graph_to_gdfs(G)
    for i in range(len(combined_shortest_dummy_nodes)):
        cur_node=combined_shortest_dummy_nodes[i]
        try:
            cur_shortest_path=nx.shortest_path(G, depot_node, cur_node) 
            if len(cur_shortest_path)>0:
                route_nodes = nodes.loc[cur_shortest_path]
                shortest_path_routes.append(route_nodes.index)
        except nx.NetworkXNoPath:
            print("not reachable"+i)
            shortest_path_routes.append(0)
        
    return shortest_path_routes

##################################################################################################################

def get_shortest_path_with_nodes(combined_shortest_dummy_nodes, shortest_path_routes):
    all_nodes_id=combined_shortest_dummy_nodes

    shortest_path_with_nodes=[]
    shortest_path_node_index=[]
    for i in range(len(shortest_path_routes)):
        cur_route=shortest_path_routes[i]
        cur_route_with_nodes=[]
        cur_route_node_index=[]
        for j in range(len(cur_route)-1):
            cur_id=cur_route[j]
            if cur_id in all_nodes_id:
                cur_route_with_nodes.append(cur_id) 
                cur_route_node_index.append(all_nodes_id.index(cur_id))
        cur_route_node_index.append(i)
                
        shortest_path_with_nodes.append(cur_route_with_nodes)
        shortest_path_node_index.append(cur_route_node_index)
    
    return shortest_path_with_nodes, shortest_path_node_index, all_nodes_id


def get_shortest_path_tree(shortest_path_node_index):    
    list_lens=[]
    for i in range(len(shortest_path_node_index)):
        list_lens.append(len(shortest_path_node_index[i]))
        
    tree_nodes=[0]
    node_parent=[0]
    node_children=[[] for i in range(len(shortest_path_node_index))]
    for i in range(1, len(shortest_path_node_index)):
        if list_lens[i]==2:
            if i in set(tree_nodes):
                continue
            else:
                if not i in tree_nodes:
                    tree_nodes.append(i)
                    node_parent.append(0)
                
                if not tree_nodes[len(tree_nodes)-1] in set(node_children[0]):
                    node_children[0].append(tree_nodes[len(tree_nodes)-1])
                
        else:
            for j in range(1,list_lens[i]):
                previous_node=shortest_path_node_index[i][j-1]
                current_node=shortest_path_node_index[i][j]
                if current_node in set(tree_nodes):
                    continue
                else:
                    tree_nodes.append(current_node)
                    parent_index_in_tree=tree_nodes.index(previous_node)
                    node_parent.append(previous_node)
                    if not tree_nodes[len(tree_nodes)-1] in set(node_children[parent_index_in_tree]):
                        node_children[parent_index_in_tree].append(tree_nodes[len(tree_nodes)-1])
                    
    return tree_nodes, node_parent, node_children,list_lens

##################################################################################################################

def branch_size_of_tree(root_node_number,tree_nodes, node_children):
    
    root_node_index=tree_nodes.index(root_node_number)
    branches=node_children[root_node_index]
    num_of_branches=len(branches)
    branch_size=[]
    branch_all_nodes=[]
    for i in range(0,num_of_branches):
        cur_branch_node=branches[i]        
        more_child, children_nodes=get_all_children(cur_branch_node,[], tree_nodes, node_children)
        branch_size.append(len(children_nodes))
        branch_all_nodes.append(children_nodes)

    return branch_size, branch_all_nodes
            
            
def get_all_children(cur_branch_node, children_nodes, tree_nodes, node_children):
        
    branch_root_index=tree_nodes.index(cur_branch_node)
    #print(branch_root_index)
    branch_child=node_children[branch_root_index]

    if len(branch_child)==0:
        more_child=0
        if not cur_branch_node in set(children_nodes):
            children_nodes=children_nodes+[cur_branch_node]
        return more_child, children_nodes
    else:
        more_child=1
        children_nodes=children_nodes+branch_child
        for i in range(len(branch_child)):           
            more_on_this, children_nodes=get_all_children(branch_child[i], children_nodes, tree_nodes, node_children)
            if not cur_branch_node in set(children_nodes):
                children_nodes=children_nodes+[cur_branch_node]
        return more_child, children_nodes

##################################################################################################################
def filter_intersection_structure(area_non_walk_df, combined_shortest_dummy_coord, depot_point, tree_nodes, node_children):
    node_point_amount=area_non_walk_df.shape[0]+1
    all_combined_node_amount=len(combined_shortest_dummy_coord)
    
    print(len(combined_shortest_dummy_coord))
    
    new_combined_shortest_dummy_coord=[]
    new_combined_shortest_dummy_coord.append(depot_point)
    for i in range(1,node_point_amount):
        new_combined_shortest_dummy_coord.append((area_non_walk_df.pad_nodes_lat[i-1], area_non_walk_df.pad_nodes_lon[i-1]))
    
    new_filtered_intersection_points=[]
    for i in range(node_point_amount, all_combined_node_amount):
        
        children_index=tree_nodes.index(i)
        current_children_len=len(node_children[children_index])
        #cur_node_parent=node_parent[i]
        
        children_has_within=0
        for j in range(0, current_children_len):
            if node_children[children_index][j]<node_point_amount:
                children_has_within=1
        
        if (current_children_len==2 and children_has_within) or current_children_len>2:
        #if current_children_len>=2:
            new_combined_shortest_dummy_coord.append(combined_shortest_dummy_coord[i])
            new_filtered_intersection_points.append(combined_shortest_dummy_coord[i])
            
    print(len(new_combined_shortest_dummy_coord))
    print(len(new_filtered_intersection_points))
    
    return new_combined_shortest_dummy_coord, new_filtered_intersection_points

##################################################################################################################
##################################################################################################################






