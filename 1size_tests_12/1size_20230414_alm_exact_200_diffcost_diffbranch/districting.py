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


def get_coeff(cz_input, branch_choices, scale_ratio_cost, area_non_walk_df):
    
    
    Z_set=[i for i in range(len(cz_input))]

    #N_set=[i for i in range(area_non_walk_df.shape[0]-1)]
    N_set=[i for i in range(area_non_walk_df.shape[0])]
    initial_b_num=len(branch_choices[0])
    final_b_num=len(branch_choices[-1])
    num_choices=final_b_num-initial_b_num+1
    SN_set=[]
    c_n={}
    for i in range(0,num_choices):
        for j in range(len(N_set)):
            SN_set.append((i,j))
            c_n[j]=scale_ratio_cost*(area_non_walk_df['pad_nodes_all_pickup_amount'][j]+area_non_walk_df['pad_nodes_all_dropoff_amount'][j])

    cn_feed=c_n
    
    cF_feed=80
    
    S_set=[i for i in range(0,num_choices)]
    
    SZ_set=[]
    c_z={}
    for i in range(0,num_choices):
        for j in Z_set:
            SZ_set.append((i,j))
            c_z[i,j]=cz_input[j]

        
    cz_feed=c_z

    return cz_feed, cn_feed, cF_feed, Z_set, N_set, SN_set, S_set, SZ_set


def get_parameter(branch_choices, large_number, zone_across, zone_each_choice, area_non_walk_df):
    
    this_node=[]
    previous_node={}

    for i in range(len(branch_choices)):
        cur_choice=branch_choices[i]
        for j in range(len(cur_choice)):
            cur_branch=cur_choice[j]
            for k in range(len(cur_branch)):
                cur_node=cur_branch[k]
                this_node.append(cur_node)
                if k==0:
                    previous_node[(i,cur_node)]=large_number
                else:
                    previous_node[(i,cur_node)]=cur_branch[k-1]
    
    initial_b_num=len(branch_choices[0])
    final_b_num=len(branch_choices[-1])
    num_choices=final_b_num-initial_b_num+1
    
    u_sz={}
    for i in range(0, num_choices):
        for j in range(len(zone_across)):
            if zone_across[j] in zone_each_choice[i]:
                u_sz[i,j]=1
            else:
                u_sz[i,j]=0
    
    #N_set=[i for i in range(area_non_walk_df.shape[0]-1)]
    N_set=[i for i in range(area_non_walk_df.shape[0])]
    n_in_z_in_s={}
    for i in range(len(branch_choices)):
        for j in N_set:
            n_in_z_in_s[(i,j)]=[]
            for k in range(len(zone_across)):
                if j in zone_across[k] and zone_across[k] in zone_each_choice[i]:
                    n_in_z_in_s[(i,j)].append(k)
    
    
    
    return previous_node, u_sz, n_in_z_in_s
    

def main_model_srd(cs_input, vehicle_upper_bound, area_non_walk_df, all_node_order_result, final_decision_zone_list, all_decision_list_result):
    
    
    
    scale_ratio_cost=30
    large_number=10000
    num_nodes_in_model=area_non_walk_df.shape[0]
    cz_feed_in, cn_feed_in, cF_feed_in, Z_set_in, N_set_in, SN_set_in, S_set_in, SZ_set_in=get_coeff(cs_input, all_node_order_result, scale_ratio_cost, area_non_walk_df)
    previous_node_in, u_sz_in, n_in_z_in_s_in=get_parameter(all_node_order_result, large_number, final_decision_zone_list, all_decision_list_result, area_non_walk_df)


    model=gp.Model('service_region_design_solver_selection')
    
    
    y_sz = model.addVars(S_set_in,Z_set_in , vtype=GRB.BINARY, name='y_sz')
    x_sn = model.addVars(S_set_in, N_set_in , vtype=GRB.BINARY, name='x_sn')
    v = model.addVar(lb=0,vtype=GRB.INTEGER,name='v')
    q_s=model.addVars(S_set_in, vtype=GRB.BINARY, name='q_s')
    
    model.ModelSense=GRB.MINIMIZE
    model.setObjective(sum(cz_feed_in[i,j]*y_sz[i,j] for j in Z_set_in for i in S_set_in)+sum(cn_feed_in[i]*(1-sum(x_sn[j,i] for j in S_set_in)) for i in N_set_in) + cF_feed_in*v)
    
    first_term=sum(cz_feed_in[i,j]*y_sz[i,j] for j in Z_set_in for i in S_set_in)
    second_term=sum(cn_feed_in[i]*(1-sum(x_sn[j,i] for j in S_set_in)) for i in N_set_in)
    third_term=cF_feed_in*v
    
    sethterms = sum(cn_feed_in[i]*(1-sum(x_sn[j,i] for j in S_set_in)) for i in N_set_in) + cF_feed_in*v
    
    model.addConstr(sum(y_sz[i] for i in SZ_set_in) <= v)
    
    for i in S_set_in:
        for j in N_set_in:
            
            model.addConstr(sum(y_sz[i,k] for k in n_in_z_in_s_in[(i,j)]) >= x_sn[i,j])
            model.addConstr(sum(y_sz[i,k] for k in n_in_z_in_s_in[(i,j)]) <= x_sn[i,j])
           
    
    for j in N_set_in:
        model.addConstr( sum(x_sn[i,j] for i in S_set_in)<=1)
        
    for i in S_set_in:
        for j in N_set_in:
            if previous_node_in[i,j]<num_nodes_in_model:
                model.addConstr(x_sn[i,j]<=x_sn[i,previous_node_in[i,j]])
                
    model.addConstr(sum(q_s[i] for i in S_set_in) >=1)
    model.addConstr(sum(q_s[i] for i in S_set_in) <=1)
            
    for i in S_set_in:
        for j in N_set_in:
            model.addConstr(x_sn[i,j]<=q_s[i])
        
    
    for z in Z_set_in:
        model.addConstr(sum(u_sz_in[(s,z)] *q_s[s] for s in S_set_in) >=sum(y_sz[s,z] for s in S_set_in))
    
    
    model.addConstr(v<=vehicle_upper_bound)
    
    
    #model.addConstr(q_s[0]==0)
    #model.addConstr(q_s[6]==0)
    #model.addConstr(q_s[5]==0)
    #model.addConstr(q_s[4]==0)
    #model.addConstr(q_s[3]==0)
    #model.addConstr(q_s[1]==0)
    #model.addConstr(q_s[2]==0)
    
    model.Params.MIPGap = 0
    model.optimize()
    model.printAttr('X')
    
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)
    
    result_v=[]
    for name, val in zip(names, values):
        result_v.append(int(val))
        

    
    result_Vars_ysz=[[0 for z in Z_set_in] for s in S_set_in]
    for s in S_set_in:
        for z in Z_set_in:
            wx=int(result_v[len(Z_set_in)*s+z])
            result_Vars_ysz[s][z]=wx  
            
            
    result_Vars_xsn=[[0 for n in N_set_in] for s in S_set_in]
    before1=len(Z_set_in)*len(S_set_in)
    for s in S_set_in:
        for n in N_set_in:
            wx=int(result_v[before1+len(N_set_in)*s+n])
            result_Vars_xsn[s][n]=wx   
            
            
    result_Vars_v=0
    before2=before1+len(N_set_in)*len(S_set_in)
    wx=int(result_v[before2])
    result_Vars_v=wx

    result_Vars_qs=[0 for s in S_set_in]
    before3=before2+1
    for s in S_set_in:
        wx=int(result_v[before3+s])
        result_Vars_qs[s]=wx   

    obj = model.getObjective()
    
    seth_value=sethterms.getValue()
    
    print("first_term: "+str(first_term.getValue()))
    print("second_term: "+str(second_term.getValue()))
    print("third_term: "+str(third_term.getValue()))
             
    return seth_value, result_Vars_ysz, result_Vars_xsn, result_Vars_v, result_Vars_qs, obj.getValue()


def post_process_data(cs_input, y_result,x_result,v_result,q_result, all_node_order_result,area_non_walk_df, final_decision_zone_list):
    
    branching_choice_decision=q_result.index(1)
    
    initial_b_num=len(all_node_order_result[0])
    final_b_num=len(all_node_order_result[-1])
    num_choices=final_b_num-initial_b_num+1    
    S_set_in=[i for i in range(0,num_choices)]
    Z_set_in=[i for i in range(len(cs_input))]
    N_set_in=[i for i in range(area_non_walk_df.shape[0])] 
    
    choice=x_result[branching_choice_decision]
    choice_nodes=[]
    for i in N_set_in:
        if choice[i]==1:
            choice_nodes.append(i)

    branching_origin=all_node_order_result[branching_choice_decision]
    
    decision_node_in_branch=[[] for i in range(len(branching_origin))]
    for i in range(len(choice_nodes)):
        cur_node=choice_nodes[i]
        for j in range(len(branching_origin)):
            cur_branch=branching_origin[j]
            if cur_node in cur_branch:
                decision_node_in_branch[j].append(cur_node)
                
                
    choice_z=y_result[branching_choice_decision]
    choice_zones=[]
    for i in Z_set_in:
        if choice_z[i]==1:
            choice_zones.append(i)
    
    decision_zone_in_branch=[]
    for i in range(len(choice_zones)):
        cur_zone=choice_zones[i]
        decision_zone_in_branch.append(final_decision_zone_list[cur_zone])
    

    
    return decision_node_in_branch, decision_zone_in_branch

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
