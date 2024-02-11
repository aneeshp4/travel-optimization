#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:18:49 2022

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

def get_all_dem_files(eval_flag,num_version_dem):
    if eval_flag==0:
        all_dem_files=[]
        for i in range(0, num_version_dem):
            dem_file_name="overall_demand_instance_v%d.csv" % (i)
            dem_file_read=pd.read_csv(dem_file_name)
            all_dem_files.append(dem_file_read)
    elif eval_flag==1:
        all_dem_files=[]
        for i in range(num_version_dem, 2*num_version_dem):
            dem_file_name="overall_demand_instance_v%d.csv" % (i)
            dem_file_read=pd.read_csv(dem_file_name)
            all_dem_files.append(dem_file_read)
    
    return all_dem_files

def get_train_arr_steps():
    
    train_arr_file_name="train_arr_time_all_versions.csv"
    train_arr_steps=pd.read_csv(train_arr_file_name)
    
    return train_arr_steps



def get_pickup_devlivery_dem_arrays(area_non_walk_df, dem_file_instance, within_nodes, train_arr_at_step):

    #print(within_nodes)
    relevant_rows_index=area_non_walk_df.iloc[within_nodes]['full_index']
    pickup_demand_all=dem_file_instance['pickup_demand'][relevant_rows_index]
    sim_duration_count=len(json.loads(pickup_demand_all.iloc[0]))
    
    pax_count_total=[0]*sim_duration_count 
    demand_list=[[] for i in range(0, sim_duration_count)]
    for i in range(0, pickup_demand_all.shape[0]):
        cur_dem=json.loads(pickup_demand_all.iloc[i])
        for j in range(0, sim_duration_count):        
            pax_count_total[j]=pax_count_total[j]+cur_dem[j]
            if cur_dem[j]>0:
                for k in range(0, cur_dem[j]):
                    demand_list[j].append(i)

    #print(pickup_demand_all)
    
    train_arr_instances=[i for i in range(len(train_arr_at_step)) if train_arr_at_step[i]>0 and i<sim_duration_count]
    #print(train_arr_instances)
    dropoff_demand_all=dem_file_instance['dropoff_demand'][relevant_rows_index]
    sim_duration_count=len(json.loads(dropoff_demand_all.iloc[0]))
    
    
    pax_off_total=[0]*sim_duration_count 
    off_list=[[] for i in range(0, sim_duration_count)]
    
    #print(train_arr_instances)
    for i in range(0, dropoff_demand_all.shape[0]):
        cur_dem=json.loads(dropoff_demand_all.iloc[i])
        #print(cur_dem)
        previous_sim_time=0
        for j in range(0, sim_duration_count):            
            if j not in train_arr_instances:
                pax_off_total[j]=0
                
            else:
                #print("sum_in")
                #print(cur_dem)
                pax_off_total[j]=pax_off_total[j]+sum(cur_dem[previous_sim_time:j+1])
                if sum(cur_dem[previous_sim_time:j+1])>0:
                     for k in range(0, sum(cur_dem[previous_sim_time:j+1])):
                         off_list[j].append(i)              
                previous_sim_time=j+1
      
    #print(dropoff_demand_all)  
    #print(pax_off_total)
    #print(off_list)
        
    return pax_count_total, demand_list, pax_off_total, off_list


def get_all_dem_instance_pickup_and_dropoff(eval_flag, area_non_walk_df, num_version_dem, within_nodes):
    all_dem_files=get_all_dem_files(eval_flag, num_version_dem)
    
    pax_count_total_across_version=[]
    demand_list_across_version=[]
    pax_off_total_across_version=[]
    off_lists_across_version=[]
    for i in range(0, num_version_dem):
        
        train_arr_steps=get_train_arr_steps()
        train_arr_steps=train_arr_steps.loc[:, ~train_arr_steps.columns.str.contains('^Unnamed')]                    
        train_arr_steps=train_arr_steps.iloc[[i]]
        train_arr_at_step=train_arr_steps.values.tolist()[0]
        
        
        #print(within_nodes)
        pax_count_total, demand_list, pax_off_total, off_list=get_pickup_devlivery_dem_arrays(area_non_walk_df, all_dem_files[i],within_nodes, train_arr_at_step)
        pax_count_total_across_version.append(pax_count_total)
        demand_list_across_version.append(demand_list)
        pax_off_total_across_version.append(pax_off_total)
        off_lists_across_version.append(off_list)
    
    return pax_count_total_across_version, demand_list_across_version, pax_off_total_across_version, off_lists_across_version


##################################################################################################################
def dispatch_frequency_calc(num_points, fix_fleet_size, instances):
    
    #mini_step_size=3
    mini_step_size=1

    dispatch_vary_times=[]
    for i in range(0, instances): 
        cur_num=i % mini_step_size
        if cur_num==0:
            dispatch_vary_times.append(i)

    return dispatch_vary_times

def update_no_run(i, longest_wait_count, num_pickup_step, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, num_dropoff_step, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time):

    
    
    if num_pickup_step[i]>0:
        pickup_waiting_nodes.append(pickup_step_list[i])
        pickup_waiting.append(num_pickup_step[i])
        #pickup_wait_time=[x+1 for x in pickup_wait_time]
        pickup_wait_time.append(0)
        
    if num_dropoff_step[i]>0:
        dropoff_waiting_nodes.append(dropoff_step_list[i])
        dropoff_waiting.append(num_dropoff_step[i])
        #dropoff_wait_time=[x+1 for x in dropoff_wait_time]
        dropoff_wait_time.append(0)      

    if len(pickup_wait_time)>0 or len(dropoff_wait_time)>0:
        longest_wait_count=longest_wait_count+1

    if len(pickup_wait_time)>0:
        pickup_wait_time=[x+1 for x in pickup_wait_time]
        
    if len(dropoff_wait_time)>0:
        dropoff_wait_time=[x+1 for x in dropoff_wait_time]
    return longest_wait_count, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time

def find_next_pax(pickup_waiting, pickup_wait_time, dropoff_waiting, dropoff_wait_time, max_amount,max_amount_pickup, max_amount_dropoff):
    
    pickup_length=len(pickup_waiting)
    dropoff_length=len(dropoff_waiting)
    
    pickup_take=0
    dropoff_take=0
    taken_amount=0
    if_empty=0
    
    if pickup_length==0:
        if dropoff_length>0 and not max_amount_dropoff==0:
            if dropoff_waiting[0]<=max_amount_dropoff:
                dropoff_take=1
                taken_amount=dropoff_waiting[0]
                if_empty=1
            else:
                dropoff_take=1
                taken_amount=max_amount_dropoff
                if_empty=0               
    elif dropoff_length==0:
        if pickup_length>0 and not max_amount_pickup==0:
            if pickup_waiting[0]<=max_amount_pickup:
                pickup_take=1
                taken_amount=pickup_waiting[0]
                if_empty=1                
            else:
                pickup_take=1
                taken_amount=max_amount_pickup
                if_empty=0             
    else:      
     
        if pickup_wait_time[0]>= dropoff_wait_time[0]:
            if not max_amount_pickup==0:
                if pickup_waiting[0]<=max_amount_pickup:
                    pickup_take=1
                    taken_amount=pickup_waiting[0]
                    if_empty=1  
                    
                else:
                    pickup_take=1
                    taken_amount=max_amount_pickup
                    if_empty=0   
        else:
            if not max_amount_dropoff==0:
                if dropoff_waiting[0]<=max_amount_dropoff:
                    dropoff_take=1
                    taken_amount=dropoff_waiting[0]
                    if_empty=1
                else:
                    dropoff_take=1
                    taken_amount=max_amount_dropoff
                    if_empty=0             
    #print(pickup_waiting)
    
    return pickup_take, dropoff_take, taken_amount, if_empty


def update_with_runs(area_non_walk_df, G, depot_node, longest_distance_from_depot, within_nodes, count_not_good, opt_cap, empty_seat_ratio, cur_pickup,cur_dropoff,vehicle_avail, veh_cap, num_of_run_count, opt_run_minute, total_obj_cost,obj_cost_list, i, longest_wait_count, num_pickup_step, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, num_dropoff_step, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time, scale_up):

    veh_return_times=[]
    veh_return_amount=[]
    pickup_this_nodes=[]
    dropoff_this_nodes=[]
    more_runs_flag=1
    while more_runs_flag==1 and vehicle_avail>0:  
        print("with runs vehicle_avail:  "+ str(vehicle_avail))
        pickup_this_run=0
        dropoff_this_run=0
        total_this_run=pickup_this_run+dropoff_this_run
        
        while total_this_run<opt_cap:
            max_amount=opt_cap-total_this_run
            max_amount_pickup=veh_cap-pickup_this_run
            max_amount_dropoff=veh_cap-dropoff_this_run
            pickup_take, dropoff_take, taken_amount, if_empty=find_next_pax(pickup_waiting, pickup_wait_time, dropoff_waiting, dropoff_wait_time, max_amount,max_amount_pickup, max_amount_dropoff)
            if pickup_take==0 and dropoff_take==0:
                more_runs_flag=0
                break
            if pickup_take:
                pickup_this_run=pickup_this_run+taken_amount
                pickup_this_nodes=pickup_this_nodes+pickup_waiting_nodes[0][0:taken_amount]
                if not if_empty:
                    pickup_waiting[0]=pickup_waiting[0]-taken_amount  
                    pickup_waiting_nodes[0]=pickup_waiting_nodes[0][taken_amount:]
                else:
                    pickup_waiting=pickup_waiting[1:]
                    pickup_wait_time=pickup_wait_time[1:]  
                    pickup_waiting_nodes=pickup_waiting_nodes[1:]

            if dropoff_take:
                dropoff_this_run=dropoff_this_run+taken_amount
                dropoff_this_nodes=dropoff_this_nodes+dropoff_waiting_nodes[0][0:taken_amount]
                if not if_empty:
                    dropoff_waiting[0]=dropoff_waiting[0]-taken_amount
                    dropoff_waiting_nodes[0]=dropoff_waiting_nodes[0][taken_amount:]
                else:
                    dropoff_waiting=dropoff_waiting[1:]
                    dropoff_wait_time=dropoff_wait_time[1:]     
                    dropoff_waiting_nodes=dropoff_waiting_nodes[1:]
            total_this_run=pickup_this_run+dropoff_this_run
            
        #vehicle_amount_this_run= math.ceil(max(math.ceil(pickup_this_run/4), math.ceil(dropoff_this_run/4))*empty_seat_ratio)  
        #vehicle_amount_this_run= math.ceil(max(math.ceil(pickup_this_run/4), math.ceil(dropoff_this_run/4)))  
        vehicle_amount_this_run=1
        
        if vehicle_amount_this_run<=vehicle_avail:
            if pickup_this_run==0 and dropoff_this_run==0:
                continue
            else:
                
                pickup_dropoff_this_nodes_nx=[]
                
                #print(pickup_this_nodes)
                for j in range(0, pickup_this_run):
                    
                    cur_point=(area_non_walk_df.iloc[within_nodes[pickup_this_nodes[j]]]['pad_nodes_lat'],area_non_walk_df.iloc[within_nodes[pickup_this_nodes[j]]]['pad_nodes_lon'])
                    cur_node=ox.get_nearest_node(G, cur_point)
                    pickup_dropoff_this_nodes_nx.append(cur_node) 

                    
                for j in range(0, dropoff_this_run):
                    cur_point=(area_non_walk_df.iloc[within_nodes[dropoff_this_nodes[j]]]['pad_nodes_lat'],area_non_walk_df.iloc[within_nodes[dropoff_this_nodes[j]]]['pad_nodes_lon'])
                    cur_node=ox.get_nearest_node(G, cur_point)
                    pickup_dropoff_this_nodes_nx.append(cur_node) 
                    
                print("within time run "+ str(i))    
                point_this_run_nodes=construct_cost_point_list(depot_node, pickup_dropoff_this_nodes_nx, pickup_this_run, dropoff_this_run)
                obj_cost_list, total_obj_cost, num_of_run_count, vehicle_amount_true, vehicle_back_time,count_not_good=run_instance(longest_distance_from_depot, G, point_this_run_nodes, count_not_good, pickup_this_run, dropoff_this_run, vehicle_amount_this_run, veh_cap, num_of_run_count, total_obj_cost,obj_cost_list)                        
                opt_run_minute.append(i)
                print("testing vehicle_avail 1:  "+ str(vehicle_avail))
                print("testing vehicle_amount_true:  "+ str(vehicle_amount_true))
                vehicle_avail =vehicle_avail-vehicle_amount_true 
                print("testing vehicle_avail 2  "+ str(vehicle_avail))
                
                for veh_back_array_index in range(0, vehicle_amount_true):
                    veh_return_times.append(vehicle_back_time[veh_back_array_index] )
                    veh_return_amount.append(1)
                      
        else:
            break
     
    if len(pickup_wait_time)>0:
        pickup_wait_time=[x+1 for x in pickup_wait_time]
        
    if len(dropoff_wait_time)>0:
        dropoff_wait_time=[x+1 for x in dropoff_wait_time]
        
    if len(pickup_wait_time)==0 and len(dropoff_wait_time)==0:
        longest_wait_count=0
    
    return longest_wait_count, count_not_good, obj_cost_list, total_obj_cost, num_of_run_count, opt_run_minute, vehicle_avail, veh_return_times, veh_return_amount, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time
 
def construct_cost_point_list(depot_node, pickup_dropoff_this_nodes_nx, pickup_this_run, dropoff_this_run):
    
    print(pickup_dropoff_this_nodes_nx)
    point_this_run_nodes=[]
    point_this_run_nodes.append(depot_node)
    for i in range(0,len(pickup_dropoff_this_nodes_nx)):
        point_this_run_nodes.append(pickup_dropoff_this_nodes_nx[i])
        
    point_this_run_nodes.append(depot_node)
        
    return point_this_run_nodes

def run_instance(longest_distance_from_depot, G, point_this_run_nodes, count_not_good, cur_pickup, cur_dropoff, vehicle_amount, veh_cap, num_of_run_count, total_obj_cost,obj_cost_list):

    
    cost_kij=get_cost_matrix(G, point_this_run_nodes, vehicle_amount)
    
    print("calling optimization function")
    print(cost_kij)
    print(cur_pickup)
    print(cur_dropoff)
    print(point_this_run_nodes)
    print(len(point_this_run_nodes))
    obj_cost, vehicle_amount_true, vehicle_back_time, not_good_answer=flm_solve_distance_and_time(longest_distance_from_depot, cost_kij, cur_pickup, cur_dropoff, vehicle_amount, veh_cap)
   
    if not_good_answer==1:
        count_not_good=count_not_good+1
    if obj_cost==0:      
        time.sleep(30)   

    obj_cost_list.append(obj_cost)
    total_obj_cost=total_obj_cost+obj_cost
    num_of_run_count=num_of_run_count+1
    
    return obj_cost_list, total_obj_cost, num_of_run_count, vehicle_amount_true, vehicle_back_time, count_not_good



def get_cost_matrix(G, point_this_run_nodes, num_vehicle):  
    num_node=len(point_this_run_nodes)
    
    c_ij=np.ones(shape=(num_vehicle, num_node,num_node),dtype='float')
    for i in range(0, num_node):
        origin_node=point_this_run_nodes[i]
        for j in range(0, num_node):
            destination_node=point_this_run_nodes[j]
            if i<j:
                try:
                    length = nx.shortest_path_length(G, source=origin_node, target=destination_node, weight='length', method='dijkstra')
                    for k in range(0, num_vehicle):
                        c_ij[k][i][j]=length
                        c_ij[k][j][i]=length
                except:
                    for k in range(0, num_vehicle):
                        c_ij[k][i][j]=10
                        c_ij[k][j][i]=10
                        
            elif i==j:
                for k in range(0, num_vehicle):
                    c_ij[k][i][j]=0        
    return c_ij
    

    
##################################################################################################################



def flm_solve_distance_and_time(longest_distance_from_depot, cost_kij, num_pickup, num_delivery, num_vehicle, cap_vehicle):
    
    
    print("inside optimization function")
    
    
    num_cust_node=num_pickup+num_delivery
    num_node=num_cust_node+2
    Node=[i for i in range(0,num_node)]
    Veh=[i for i in range(0,num_vehicle)]

    
    def get_q(n_p, n_d):
        q=[1 for l in Node]
        q[0]=0
        q[num_node-1]=0
        for i in range(n_p+1, n_p+n_d+1):
            q[i]=-1
        return q
    q=get_q(num_pickup, num_delivery)
    
        
    def getSet3up():
        S3up=[]
        set3upNodes=[i for i in range(1,num_node-1)]
        for i in range(3,num_node-1):
            comb = combinations(list(set3upNodes), i)
            S3up.append(list(comb))
        S3up=list(itertools.chain(*S3up))
        return S3up    
    
    #meters/minute
    vehicle_velocity=670.56 
    
    km_multiplier=1
    t_kij=cost_kij/vehicle_velocity*km_multiplier
    
    #t_max_1=longest_distance_from_depot/vehicle_velocity*km_multiplier+1
    t_max=longest_distance_from_depot*2*(num_cust_node+1)/vehicle_velocity*km_multiplier+1
    

    M1216=num_node*num_node+1

    M1729=t_max
    M181922=cap_vehicle+1
    
    M303132=t_max
    
    #print(M1216)
    #print(M1729)
    #print(M181922)
    #print(M303132)
    
    print(t_kij)
    
    
    l=num_pickup+num_delivery+1
    #passenger time value 15.6/60
    pax_time_value=0.26
    constant_ratio=pax_time_value
    vehicle_dist_value=0.00061
    op_cost_veh=vehicle_dist_value*1
    loopSets=getSet3up()
    num_Sets=len(loopSets)
    L_set_node=[i for i in range(0,num_Sets)]    
 
    model = gp.Model('falm_multi_solver')

    x_kij = model.addVars(Veh, Node,Node, vtype=GRB.BINARY, name='x_kij')
    Q_ki = model.addVars(Veh,Node, vtype=GRB.INTEGER, name='Q_ki')
    B_ki = model.addVars(Veh, Node, vtype=GRB.CONTINUOUS, name='B_ki')
    y_ki=model.addVars(Veh, Node, vtype=GRB.BINARY, name='y_ki')
    B_bar_kil = model.addVars(Veh, Node, Node,vtype=GRB.CONTINUOUS, name='B_bar_kil')
    z_k_S=model.addVars(Veh,L_set_node, vtype=GRB.BINARY, name='z_k_S')
    model.ModelSense = GRB.MINIMIZE
    model.setObjective(op_cost_veh*sum([cost_kij[k][i][j]*x_kij[k,i,j] for k, i, j in x_kij.keys() if i!=j])+ constant_ratio*sum([(q[i]+1)/2*(B_bar_kil[k,i,l]-B_ki[k,i]) for k,i in B_ki.keys() ]) + constant_ratio*sum ([(1-q[i])/2*B_ki[k,i] for k,i in B_ki.keys()]))+constant_ratio*sum([(q[i]+1)/2*(B_ki[k,i]) for k,i in B_ki.keys() ])
    
      
    for i in range(1,num_node-1):
        model.addConstr(sum ((x_kij[k,i,j] for j in range(0,num_node) for k in range(0,num_vehicle)))>=1)
        model.addConstr(sum ((x_kij[k,i,j] for j in range(0,num_node) for k in range(0,num_vehicle)))<=1)

    for k in range (0, num_vehicle):
        model.addConstr(sum(x_kij[k,0,j] for j in range(0,num_node)) >=1)
        model.addConstr(sum(x_kij[k,0,j] for j in range(0,num_node)) <=1)
    
    for k in range (0, num_vehicle):
        model.addConstr(sum(x_kij[k,i,num_node-1] for i in range(0,num_node)) >=1)
        model.addConstr(sum(x_kij[k,i,num_node-1] for i in range(0,num_node)) <=1)    
    
    for k in range (0, num_vehicle):
        for j in range(1,num_node-1):
            model.addConstr(sum( (x_kij[k,i,j] - x_kij[k,j,i]) for i in  range(0,num_node)) >=0) 
            model.addConstr(sum( (x_kij[k,i,j] - x_kij[k,j,i]) for i in  range(0,num_node)) <=0) 
    
    for k in range (0, num_vehicle):
        model.addConstr(sum( x_kij[k,i,j] for i in range(1,num_node-1) for j in range(1,num_node-1)) <=M1216*(1- x_kij[k,0,num_node-1]) )
    
    for k in range (0, num_vehicle):
        for i in range(0,num_node):    
            for j in range(0,num_node):
                model.addConstr( x_kij[k,i,j]+x_kij[k,j,i] <=1)
         
    for k in range (0, num_vehicle):
        for s in range(0, num_Sets):
            item_set=loopSets[s] 
            set_bar=set(Node).difference(item_set)
            model.addConstr(sum(x_kij[k,i,j] for j in set_bar for i in item_set)+sum(x_kij[k,i,j] for j in item_set for i in set_bar) >=2 * z_k_S[k,s] )
            model.addConstr( z_k_S[k,s] <= sum(x_kij[k,i,j] for i in item_set for j in item_set) )
            model.addConstr( M1216*z_k_S[k,s] >= sum(x_kij[k,i,j] for i in item_set for j in item_set) )
    
    
    for k in range (0, num_vehicle):
        for i in range(0,num_node):    
            for j in range(0,num_node):    
                model.addConstr( B_ki[k,j] >= B_ki[k,i] + t_kij[k][i][j] + M1729* (x_kij[k,i,j]-1))
                
    
    for k in range (0, num_vehicle):
        for i in range(0,num_node):    
            for j in range(0,num_node):    
                model.addConstr( Q_ki[k,j] >= Q_ki[k,i] + q[j] + M181922* (x_kij[k,i,j]-1))             
                model.addConstr( Q_ki[k,j] <= Q_ki[k,i] + q[j] - M181922* (x_kij[k,i,j]-1))
    
    for k in range (0, num_vehicle):
        for i in range(0,num_node):    
            model.addConstr( Q_ki[k,i]>=0)
            #model.addConstr( Q_ki[k,i]>=q[i])
            model.addConstr( Q_ki[k,i]<= cap_vehicle)
            #model.addConstr( Q_ki[k,i]<= cap_vehicle+q[i])
                       
    for k in range (0, num_vehicle):
        for i in range(1,num_node-1):   
            model.addConstr( Q_ki[k,i]<= M181922*y_ki[k,i])
            
    for k in range (0, num_vehicle):  
       model.addConstr( Q_ki[k,0]>= -sum( q[i] * y_ki[k,i] for i in range(num_pickup+1,num_node-1))) 
       model.addConstr( Q_ki[k,0]<= -sum( q[i] * y_ki[k,i] for i in range(num_pickup+1,num_node-1)))
       model.addConstr( Q_ki[k,l]>= sum( q[i] * y_ki[k,i] for i in range(1,num_pickup+1))) 
       model.addConstr( Q_ki[k,l]<= sum( q[i] * y_ki[k,i] for i in range(1,num_pickup+1)))
       
    for k in range (0, num_vehicle):
        model.addConstr(B_ki[k,0]>=0)
        model.addConstr(B_ki[k,0]<=0)     
        for i in range(0,num_node):
            model.addConstr(x_kij[k,i,i]>=0)
            model.addConstr(x_kij[k,i,i]<=0)
            model.addConstr(x_kij[k,i,0]>=0)
            model.addConstr(x_kij[k,i,0]<=0)

            
    for k in range (0, num_vehicle):
        model.addConstr(sum( [t_kij[k][i][j]*x_kij[k,i,j] for i in range(0,num_node) for j in range(0,num_node) ])>=B_ki[k,l])
        model.addConstr(sum( [t_kij[k][i][j]*x_kij[k,i,j] for i in range(0,num_node) for j in range(0,num_node) ])<=B_ki[k,l])
        

    for k in range (0, num_vehicle):
        for i in range(1,num_node-1):
            model.addConstr( y_ki[k,i] >= sum(x_kij[k,i,j] for j in range(0,num_node)))
            model.addConstr( y_ki[k,i] <= sum(x_kij[k,i,j] for j in range(0,num_node)))
            
    for k in range (0, num_vehicle):
        for i in range(1,num_node-1):
            model.addConstr(M1729*y_ki[k,i] >=B_ki[k,i])
        
    for k in range (0, num_vehicle):
        for i in range(1,num_node-1):
           model.addConstr(B_bar_kil[k,i,l]<=B_ki[k,l] + M303132*(1-y_ki[k,i]))
           model.addConstr(B_bar_kil[k,i,l]>=B_ki[k,l] - M303132*(1-y_ki[k,i]))
           model.addConstr(B_bar_kil[k,i,l]<= M303132*y_ki[k,i])
           
    for k in range (0, num_vehicle):
        for i in range(0,num_node):
           for j in range(0,num_node-1):
               model.addConstr(B_bar_kil[k,i,j]<= 0)
               model.addConstr(B_bar_kil[k,i,j]>= 0)
         
    for k in range (0, num_vehicle):
        model.addConstr(B_bar_kil[k,0,l]>= 0)
        model.addConstr(B_bar_kil[k,0,l]<= 0)
        model.addConstr(B_bar_kil[k,l,l]>= B_ki[k,l])
        model.addConstr(B_bar_kil[k,l,l]<= B_ki[k,l])

    model.Params.MIPGap = 0
    model.optimize()       
    model.printAttr('X')     
    
    result_Vars_x_kij=[[[0 for j in Node] for i in Node] for k in Veh]
    v = model.getVars()
    for k in Veh:
        for i in Node:
            for j in Node:
                wx=v[num_node*num_node*k+num_node*i+j].getAttr("X")
                if wx>0.5:
                    result_Vars_x_kij[k][i][j]=1    
                    
    result_Vars_Q_ki=[[0 for i in Node] for k in Veh]
    before1=num_vehicle*num_node*num_node
    for k in Veh:
        for i in Node:
            wx=v[before1+num_node*k+i].getAttr("X")
            if wx>M181922:
                result_Vars_Q_ki[k][i]=0
            else:
                result_Vars_Q_ki[k][i]=wx

    result_Vars_B_ki=[[0 for i in Node] for k in Veh]
    before2=num_vehicle*num_node*num_node+num_vehicle*num_node
    for k in Veh:
        for i in Node:
            wx=v[before2+num_node*k+i].getAttr("X")
            if wx>M303132:
                result_Vars_B_ki[k][i]=0
            else:
                result_Vars_B_ki[k][i]=wx

    result_Vars_y_ki=[[0 for i in Node] for k in Veh]
    before3=num_vehicle*num_node*num_node+num_vehicle*num_node*2
    for k in Veh:
        for i in Node:
            wx=v[before3+num_node*k+i].getAttr("X")
            if wx>0.5:
                result_Vars_y_ki[k][i]=1
                
    result_Vars_B_kil=[[[0 for j in Node] for i in Node] for k in Veh]
    before4=num_vehicle*num_node*num_node+num_vehicle*num_node*3
    for k in Veh:
        for i in Node:
            for j in Node:
                wx=v[before4+num_node*num_node*k+num_node*i+j].getAttr("X")
                if wx>M303132:
                    #exit()
                    large_show=1
                    result_Vars_B_kil[k][i][j]=0
                else:
                    large_show=0
                    result_Vars_B_kil[k][i][j]=wx
                    
                
    result_Vars_z_k_S=[[0 for i in L_set_node] for k in Veh]
    before5=num_vehicle*num_node*num_node*2+num_vehicle*num_node*3
    for k in Veh:
        for i in L_set_node:
            wx=v[before5+num_Sets*k+i].getAttr("X")
            if wx>0.5:
                result_Vars_z_k_S[k][i]=1              
    
    
    #post-processing
    total_vehicle_time_list=[]
    total_vehicle_time=0
    vehicle_amount_true=num_vehicle
    for k in range(0,num_vehicle):        
        total_vehicle_time=max(max(result_Vars_B_kil[k]))
        total_vehicle_time_min=min(min(result_Vars_B_kil[k]))
        if total_vehicle_time>0:
            if total_vehicle_time<10000:
                total_vehicle_time_list.append(total_vehicle_time)
            else:
                total_vehicle_time_list.append(total_vehicle_time_min)
        else:
            vehicle_amount_true=vehicle_amount_true-1
    
   
    #print(model.ObjVal)
    #print(vehicle_amount_true)
    #print(total_vehicle_time_list)
    
    vehicle_cost=0
    for k in range(0, num_vehicle):
        for i in range(0, num_node):
             for j in range(0, num_node):
                 vehicle_cost=vehicle_cost+cost_kij[k][i][j]*result_Vars_x_kij[k][i][j]  
    vehicle_cost=op_cost_veh*vehicle_cost

    #print(vehicle_cost)    
    pax_cost=0     
    for k in range(0, num_vehicle):
        for i in range(0, num_node):
            pax_cost=pax_cost+(q[i]+1)/2*(result_Vars_B_kil[k][i][l]-result_Vars_B_ki[k][i])+(1-q[i])/2*result_Vars_B_ki[k][i]
    pax_cost=constant_ratio*pax_cost
    
    #print(pax_cost)
    #print(vehicle_cost+pax_cost)
    if pax_cost<=0 or vehicle_cost<=0 or model.ObjVal<0 or large_show==1:
        not_good_answer=1
        #exit()
    else:
        not_good_answer=0


    print("model.ObjVal: "+ str(model.ObjVal))
    print("vehicle_amount_true: "+ str(vehicle_amount_true))
    print("total_vehicle_time_list: "+ str(total_vehicle_time_list))
    print("not_good_answer: "+ str(not_good_answer))
    print("finished optimization function")

    return model.ObjVal, vehicle_amount_true, total_vehicle_time_list, not_good_answer



##################################################################################################################


def knownFSknownRmultiVersion(eval_flag,area_non_walk_df,  G, depot_node, longest_distance_from_depot, fix_fleet_size, within_nodes):
    
    print(within_nodes)
    total_time=150
    threshold_instance=10
    instances=total_time
    num_points=len(within_nodes)
    num_version_dem=20

    pax_count_total_across_version, demand_list_across_version, pax_off_total_across_version, off_lists_across_version=get_all_dem_instance_pickup_and_dropoff(eval_flag, area_non_walk_df, num_version_dem, within_nodes)
    

    total_obj_cost_versions=[]
    total_obj_cost_with_pax_wait_versions=[]
    pax_left_count_versions=[]
    pax_time_factor_never_served_versions=[]
    count_not_good_versions=[]

    for version in range(0, num_version_dem):
        print("demand version:  "+ str(version))
        
        cur_pickup_total=pax_count_total_across_version[version]
        cur_pickup_list=demand_list_across_version[version]
        cur_dropoff_total=pax_off_total_across_version[version]
        cur_dropoff_list=off_lists_across_version[version]
        
        
        num_pickup_step=cur_pickup_total     
        num_dropoff_step=cur_dropoff_total
        
        veh_cap=4
        empty_seat_ratio=1.5
        opt_cap=6
        scale_up=1
        num_vehicle_init=1
    
        pax_left_count=0
        pax_time_factor_never_served=0
        total_left=0
        
        total_obj_cost_all_instances=0
        obj_cost_list_all_instances=[]
        num_of_run_count_all_instances=[]
        opt_run_minute_all_instances=[]
        
        left_pickup=0
        left_dropoff=0
        longest_wait_count=0
        train_arrival=0 
        
        vehicle_avail=fix_fleet_size
        vehicle_back_count=np.zeros(shape=(instances),dtype='int')
        
        #extended_time=instances*3
        extended_time=instances+10
        vehicle_back_over=np.zeros(shape=(extended_time),dtype='int')
        
        pickup_waiting=[]
        pickup_wait_time=[]
        dropoff_waiting=[]
        dropoff_wait_time=[]
        vehicle_avail_list=[]
        pickup_step_list=cur_pickup_list
        dropoff_step_list=cur_dropoff_list
        pickup_waiting_nodes=[]
        dropoff_waiting_nodes=[]
        
        num_of_run_count=0
        count_not_good=0
        for i in range(0,instances):
            
            print("instance time index:  "+ str(i))
            opt_run_minute=[]
            total_obj_cost=0
            obj_cost_list=[]
            
            
            left_pickup=sum(pickup_waiting)
            cur_pickup=num_pickup_step[i]+left_pickup
            
            left_dropoff=sum(dropoff_waiting)
            cur_dropoff=num_dropoff_step[i]+left_dropoff
            
            total_pax=cur_pickup+cur_dropoff
            if num_dropoff_step[i]==0:
                train_arrival=0
            else:
                train_arrival=1
                
            vehicle_avail=vehicle_avail+vehicle_back_count[i]
            print("vehicle_avail:  "+ str(vehicle_avail))
            print("vehicle_back_count:  "+ str(vehicle_back_count))

            #dispatch_vary_times=dispatch_frequency_calc(num_points, fix_fleet_size, instances)
            #no_opt_run=total_pax<6 and longest_wait_count<8 and train_arrival==0 and (i not in dispatch_vary_times)
            #no_opt_run=total_pax<4 and longest_wait_count<8 and train_arrival==0
            no_opt_run=total_pax<4 and longest_wait_count<8
            

            if no_opt_run:
                print("pickup_waiting:  "+str(pickup_waiting))
                print("pickup_wait_time:  "+ str(pickup_wait_time))
                print("dropoff_waiting:  "+str(dropoff_waiting))
                print("dropoff_wait_time:  "+ str(dropoff_wait_time))
                print("pickup_waiting_nodes:  "+str(pickup_waiting_nodes))
                print("dropoff_waiting_nodes:  "+str(dropoff_waiting_nodes)) 
                print("sum pickup_waiting:  "+ str(sum(pickup_waiting)))
                print("sum dropoff_waiting:  "+ str(sum(dropoff_waiting)))
                pax_left_count=pax_left_count+sum(pickup_waiting)+sum(dropoff_waiting)
                print("pax_left_count no run:  "+ str(pax_left_count))   
                longest_wait_count, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time = update_no_run(i, longest_wait_count, num_pickup_step, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, num_dropoff_step, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time)

                            
                
            else:
            
                print("reason for run: total_pax "+ str(total_pax)+" longest_wait_count: "+str(longest_wait_count))
                new_instance_pickup=num_pickup_step[i]
                if new_instance_pickup>0:
                    pickup_waiting_nodes.append(pickup_step_list[i])
                    print(pickup_waiting_nodes)
                    pickup_waiting.append(new_instance_pickup)
                    pickup_wait_time.append(0)
                new_instance_dropoff=num_dropoff_step[i]
                if new_instance_dropoff>0:
                    dropoff_waiting_nodes.append(dropoff_step_list[i])
                    dropoff_waiting.append(new_instance_dropoff)
                    dropoff_wait_time.append(0)
    
                #cost_kij=setRealCost(len(node_pickup_dem),len(node_dropoff_dem), num_vehicle_init, node_pickup_lat, node_pickup_lon, prob_pickup_dem, node_dropoff_lat, node_dropoff_lon, prob_dropoff_dem, depot_lat, depot_lon, G)  
                #dropoff_dataframe=pd.DataFrame(list(zip(node_dropoff_lat,node_dropoff_lon)), columns=['dropoff_x','dropoff_y'])
                #cost_kij=getCostMatrixKnownPickups(node_dropoff_lat, node_dropoff_lon, dropoff_dataframe,pickup_all_nodes, num_pickup, num_delivery, num_vehicle_init, scale_up)   
                print("pickup_waiting:  "+str(pickup_waiting))
                print("pickup_wait_time:  "+ str(pickup_wait_time))
                print("dropoff_waiting:  "+str(dropoff_waiting))
                print("dropoff_wait_time:  "+ str(dropoff_wait_time))
                print("pickup_waiting_nodes:  "+str(pickup_waiting_nodes))
                print("dropoff_waiting_nodes:  "+str(dropoff_waiting_nodes))

                longest_wait_count, count_not_good, obj_cost_list, total_obj_cost, num_of_run_count, opt_run_minute, vehicle_avail, veh_return_times, veh_return_amount, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time = update_with_runs(area_non_walk_df, G, depot_node, longest_distance_from_depot, within_nodes, count_not_good, opt_cap, empty_seat_ratio, cur_pickup,cur_dropoff,vehicle_avail, veh_cap, num_of_run_count, opt_run_minute, total_obj_cost,obj_cost_list, i, longest_wait_count, num_pickup_step, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, num_dropoff_step, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time, scale_up)
                print("sum pickup_waiting:  "+ str(sum(pickup_waiting)))
                print("sum dropoff_waiting:  "+ str(sum(dropoff_waiting)))
                pax_left_count=pax_left_count+sum(pickup_waiting)+sum(dropoff_waiting)
                print("pax_left_count with run:  "+ str(pax_left_count))   
                
                #if instances<threshold_instance or instances>threshold_instance_high:
                if instances<threshold_instance:
                    obj_cost_list=[]
                    total_obj_cost=0
                
                if len(pickup_waiting)==0 and len(dropoff_waiting)==0:
                    longest_wait_count=0
                    pickup_waiting=[]
                    pickup_wait_time=[]
                    dropoff_waiting=[]
                    dropoff_wait_time=[] 
                else:
                    
                    if len(pickup_waiting)==0:
                        longest_wait_count=dropoff_wait_time[0]
                    elif len(dropoff_waiting)==0:
                        longest_wait_count=pickup_wait_time[0]
                    else:
                        if dropoff_wait_time[0]<=pickup_wait_time[0]:
                            longest_wait_count=pickup_wait_time[0]
                        else:
                            longest_wait_count=dropoff_wait_time[0]
    
                for veh_back_timer in range(0, len(veh_return_times)):
                    back_timer_index=math.ceil(i+veh_return_times[veh_back_timer])
                    if back_timer_index< instances-1:
                        vehicle_back_count[back_timer_index+1]=vehicle_back_count[back_timer_index+1]+1
                    elif back_timer_index<extended_time-1:
                        vehicle_back_over[back_timer_index+1]=vehicle_back_over[back_timer_index+1]+1
                
                total_obj_cost_all_instances=total_obj_cost_all_instances+total_obj_cost
                obj_cost_list_all_instances=obj_cost_list_all_instances+obj_cost_list
                num_of_run_count_all_instances.append(num_of_run_count)
                opt_run_minute_all_instances=opt_run_minute_all_instances+opt_run_minute
                #print("testing after run vehicle_avail:  "+ str(vehicle_avail))
            vehicle_avail_list.append(vehicle_avail)
            
            
        pax_left_count_all_instances=pax_left_count
        if len(pickup_waiting_nodes)>0 or len(dropoff_waiting_nodes)>0:
            #if len(pickup_wait_time)==0 and len(dropoff_wait_time)==0:
            total_left, count_not_good, total_obj_cost_all_instances, obj_cost_list_all_instances, num_of_run_count_all_instances, opt_run_minute_all_instances, pax_left_count_all_instances=after_time_service(area_non_walk_df, G, depot_node, longest_distance_from_depot, within_nodes, pax_left_count_all_instances, total_obj_cost_all_instances, obj_cost_list_all_instances, num_of_run_count_all_instances, opt_run_minute_all_instances, instances, extended_time, count_not_good, opt_cap, empty_seat_ratio, vehicle_avail, veh_cap, num_of_run_count, opt_run_minute, pax_left_count, total_obj_cost,obj_cost_list, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time, vehicle_back_over)
         
   
        pax_left_count=pax_left_count_all_instances
        pax_time_value=0.26  
        waiting_time_cost=pax_left_count*pax_time_value 
        total_obj_cost_all_instances_with_pax_wait=total_obj_cost_all_instances+waiting_time_cost
  
        total_obj_cost_versions.append(total_obj_cost_all_instances)
        total_obj_cost_with_pax_wait_versions.append(total_obj_cost_all_instances_with_pax_wait)
        pax_left_count_versions.append(pax_left_count)
        pax_time_factor_never_served_versions.append(total_left)
        count_not_good_versions.append(count_not_good)
        
    total_obj_cost_versions_avg=sum(total_obj_cost_versions)/num_version_dem
    total_obj_cost_with_pax_wait_versions_avg=sum(total_obj_cost_with_pax_wait_versions)/num_version_dem
    pax_left_count_versions_avg=sum(pax_left_count_versions)/num_version_dem
    count_not_good_versions_avg=sum(count_not_good_versions)/num_version_dem
        
    return count_not_good_versions, count_not_good_versions_avg, pax_time_factor_never_served_versions, pax_left_count_versions, total_obj_cost_versions, total_obj_cost_with_pax_wait_versions, pax_left_count_versions_avg, total_obj_cost_versions_avg, total_obj_cost_with_pax_wait_versions_avg


def update_with_runs_over(area_non_walk_df, G, longest_distance_from_depot, depot_node,within_nodes, time_instance_over, count_not_good, opt_cap, empty_seat_ratio, left_pickup, left_dropoff, vehicle_avail, veh_cap, num_of_run_count, opt_run_minute, total_obj_cost,obj_cost_list, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time):

    veh_return_times=[]
    veh_return_amount=[]
    pickup_this_nodes=[]
    dropoff_this_nodes=[]
    more_runs_flag=1
    while more_runs_flag==1 and vehicle_avail>0: 
        print("with runs over vehicle_avail:  "+ str(vehicle_avail))
        pickup_this_run=0
        dropoff_this_run=0
        total_this_run=pickup_this_run+dropoff_this_run

        while total_this_run<opt_cap:
            max_amount=opt_cap-total_this_run
            max_amount_pickup=veh_cap-pickup_this_run
            max_amount_dropoff=veh_cap-dropoff_this_run
            pickup_take, dropoff_take, taken_amount, if_empty=find_next_pax(pickup_waiting, pickup_wait_time, dropoff_waiting, dropoff_wait_time, max_amount,max_amount_pickup, max_amount_dropoff)
            print("pickup_take dropoff_take taken_amount if_empty:  "+ str(pickup_take) +" "+str(dropoff_take)+" "+str(taken_amount)+" "+str(if_empty))
            if pickup_take==0 and dropoff_take==0:
                more_runs_flag=0
                break
            if pickup_take:
                pickup_this_run=pickup_this_run+taken_amount
                pickup_this_nodes=pickup_this_nodes+pickup_waiting_nodes[0][0:taken_amount]
                if not if_empty:
                    pickup_waiting[0]=pickup_waiting[0]-taken_amount  
                    pickup_waiting_nodes[0]=pickup_waiting_nodes[0][taken_amount:]
                    print("check pickup_waiting 1:  "+ str(pickup_waiting))
                else:
                    pickup_waiting=pickup_waiting[1:]
                    pickup_wait_time=pickup_wait_time[1:]  
                    pickup_waiting_nodes=pickup_waiting_nodes[1:]
                    print("check pickup_waiting 0:  "+ str(pickup_waiting))

            if dropoff_take:
                dropoff_this_run=dropoff_this_run+taken_amount
                dropoff_this_nodes=dropoff_this_nodes+dropoff_waiting_nodes[0][0:taken_amount]
                if not if_empty:
                    dropoff_waiting[0]=dropoff_waiting[0]-taken_amount
                    dropoff_waiting_nodes[0]=dropoff_waiting_nodes[0][taken_amount:]
                    print("check dropoff_waiting 1:  "+ str(dropoff_waiting))
                else:
                    dropoff_waiting=dropoff_waiting[1:]
                    dropoff_wait_time=dropoff_wait_time[1:]     
                    dropoff_waiting_nodes=dropoff_waiting_nodes[1:]
                    print("check dropoff_waiting 0:  "+ str(dropoff_waiting))
            total_this_run=pickup_this_run+dropoff_this_run
            
        #vehicle_amount_this_run= math.ceil(max(math.ceil(pickup_this_run/4), math.ceil(dropoff_this_run/4))*empty_seat_ratio)  
        #vehicle_amount_this_run= math.ceil(max(math.ceil(pickup_this_run/4), math.ceil(dropoff_this_run/4)))  
        vehicle_amount_this_run=1
        
        if vehicle_amount_this_run<=vehicle_avail:
            if pickup_this_run==0 and dropoff_this_run==0:
                continue
            else:
                
                pickup_dropoff_this_nodes_nx=[]
                
                #print(pickup_this_nodes)
                for j in range(0, pickup_this_run):
                    
                    cur_point=(area_non_walk_df.iloc[within_nodes[pickup_this_nodes[j]]]['pad_nodes_lat'],area_non_walk_df.iloc[within_nodes[pickup_this_nodes[j]]]['pad_nodes_lon'])
                    cur_node=ox.get_nearest_node(G, cur_point)
                    pickup_dropoff_this_nodes_nx.append(cur_node) 
                    
                for j in range(0, dropoff_this_run):
                    cur_point=(area_non_walk_df.iloc[within_nodes[dropoff_this_nodes[j]]]['pad_nodes_lat'],area_non_walk_df.iloc[within_nodes[dropoff_this_nodes[j]]]['pad_nodes_lon'])
                    cur_node=ox.get_nearest_node(G, cur_point)
                    pickup_dropoff_this_nodes_nx.append(cur_node) 
                    
                print("within time run "+ str(time_instance_over)) 
                point_this_run_nodes=construct_cost_point_list(depot_node, pickup_dropoff_this_nodes_nx, pickup_this_run, dropoff_this_run)
                obj_cost_list, total_obj_cost, num_of_run_count, vehicle_amount_true, vehicle_back_time, count_not_good=run_instance(longest_distance_from_depot, G, point_this_run_nodes, count_not_good, pickup_this_run, dropoff_this_run, vehicle_amount_this_run, veh_cap, num_of_run_count, total_obj_cost,obj_cost_list)                        
                opt_run_minute.append(time_instance_over)
                vehicle_avail =vehicle_avail-vehicle_amount_true                
                for veh_back_array_index in range(0, vehicle_amount_true):
                    veh_return_times.append(vehicle_back_time[veh_back_array_index] )
                    veh_return_amount.append(1)
                      
        else:
            break
     
        
    if len(pickup_wait_time)>0:
        pickup_wait_time=[x+1 for x in pickup_wait_time]
        
    if len(dropoff_wait_time)>0:
        dropoff_wait_time=[x+1 for x in dropoff_wait_time] 
        
    if len(pickup_wait_time)==0 and len(dropoff_wait_time)==0:
        longest_wait_count=0    
    
    return count_not_good, obj_cost_list, total_obj_cost, num_of_run_count, opt_run_minute, vehicle_avail, veh_return_times, veh_return_amount, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time


def after_time_service(area_non_walk_df, G, depot_node, longest_distance_from_depot, within_nodes, pax_left_count_all_instances, total_obj_cost_all_instances, obj_cost_list_all_instances, num_of_run_count_all_instances, opt_run_minute_all_instances, instances, extended_time, count_not_good, opt_cap, empty_seat_ratio, vehicle_avail, veh_cap, num_of_run_count, opt_run_minute, pax_left_count, total_obj_cost,obj_cost_list, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time, vehicle_back_over):

    left_pickup=sum(pickup_waiting)
    left_dropoff=sum(dropoff_waiting)
    total_left=left_pickup+left_dropoff
    passenger_clear=total_left==0
    
    
    for time_instance_over in range(instances, extended_time):
        print("after time run "+ str(time_instance_over))
        if passenger_clear:
            break
        else:
            vehicle_avail=vehicle_avail+vehicle_back_over[time_instance_over]
            count_not_good, obj_cost_list, total_obj_cost, num_of_run_count, opt_run_minute, vehicle_avail, veh_return_times, veh_return_amount, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time = update_with_runs_over(area_non_walk_df, G, longest_distance_from_depot, depot_node, within_nodes, time_instance_over, count_not_good, opt_cap, empty_seat_ratio, left_pickup, left_dropoff, vehicle_avail, veh_cap, num_of_run_count, opt_run_minute, total_obj_cost,obj_cost_list, pickup_step_list, pickup_waiting_nodes, pickup_waiting, pickup_wait_time, dropoff_step_list, dropoff_waiting_nodes, dropoff_waiting, dropoff_wait_time)
            print("pickup_waiting over:  "+ str(pickup_waiting))
            print("dropoff_waiting over:  "+ str(dropoff_waiting))            
            
            left_pickup=sum(pickup_waiting)
            left_dropoff=sum(dropoff_waiting)
            total_left=left_pickup+left_dropoff
            passenger_clear=total_left==0
            
            total_obj_cost_all_instances=total_obj_cost_all_instances+total_obj_cost
            obj_cost_list_all_instances=obj_cost_list_all_instances+obj_cost_list
            num_of_run_count_all_instances.append(num_of_run_count)
            opt_run_minute_all_instances=opt_run_minute_all_instances+opt_run_minute
            print("sum pickup_waiting over:  "+ str(sum(pickup_waiting)))
            print("sum dropoff_waiting over:  "+ str(sum(dropoff_waiting)))
            pax_left_count_all_instances=pax_left_count_all_instances+sum(pickup_waiting)+sum(dropoff_waiting)
            print("pax_left_count_all_instances:  "+ str(pax_left_count_all_instances))
     
            for veh_back_timer in range(0, len(veh_return_times)):
                back_timer_index=math.ceil(time_instance_over+veh_return_times[veh_back_timer])
                if back_timer_index<extended_time-1:
                    vehicle_back_over[back_timer_index+1]=vehicle_back_over[back_timer_index+1]+1

    print("after time total_left: " +str(time_instance_over)+ " "+str(total_left))
    pax_time_factor_never_served=30  
    waiting_time_cost_factor=total_left*pax_time_factor_never_served
    pax_left_count_all_instances=pax_left_count_all_instances+waiting_time_cost_factor
        
    return total_left, count_not_good, total_obj_cost_all_instances, obj_cost_list_all_instances, num_of_run_count_all_instances, opt_run_minute_all_instances, pax_left_count_all_instances





##################################################################################################################
##################################################################################################################
##################################################################################################################

def fixRvarFS(eval_flag, area_non_walk_df, G, depot_node, longest_distance_from_depot, cur_fleet_size, cur_within):

    
    count_not_good_versions, count_not_good_versions_avg, pax_time_factor_never_served_versions, pax_left_count_versions, total_obj_cost_versions, total_obj_cost_with_pax_wait_versions, pax_left_count_versions_avg, total_obj_cost_versions_avg, total_obj_cost_with_pax_wait_versions_avg =knownFSknownRmultiVersion(eval_flag, area_non_walk_df, G, depot_node, longest_distance_from_depot, cur_fleet_size, cur_within)

    a_count_not_good_versions=[str(count_not_good_versions)]
    a_pax_left_count_versions=[str(pax_left_count_versions)]
    a_total_obj_cost_versions=[str(total_obj_cost_versions)]
    a_total_obj_cost_with_pax_wait_versions=[str(total_obj_cost_with_pax_wait_versions)]

    a_pax_time_factor_never_served_versions=[str(pax_time_factor_never_served_versions)]
    
    a_pax_left_count_versions_avg=[pax_left_count_versions_avg]
    a_total_obj_cost_versions_avg=[total_obj_cost_versions_avg]
    a_total_obj_cost_with_pax_wait_versions_avg=[total_obj_cost_with_pax_wait_versions_avg]
    a_count_not_good_versions_avg=[count_not_good_versions_avg]
                                   
    listToStr = ' '.join([str(elem) for i,elem in enumerate(cur_within)])
    fixRvarFS_df = pd.DataFrame(list(zip([listToStr], a_pax_left_count_versions_avg, a_total_obj_cost_versions_avg, a_total_obj_cost_with_pax_wait_versions_avg, a_pax_time_factor_never_served_versions, a_pax_left_count_versions, a_total_obj_cost_versions, a_total_obj_cost_with_pax_wait_versions,a_count_not_good_versions, a_count_not_good_versions_avg)), columns =['cur_within', 'avg_pax_left_count', 'avg_total_obj_cost_all_instances', 'avg_total_obj_cost_all_instances_with_pax_wait', 'a_pax_time_factor_never_served_versions', 'a_pax_left_count_versions', 'a_total_obj_cost_versions', 'a_total_obj_cost_with_pax_wait_versions','a_count_not_good_versions', 'a_count_not_good_versions_avg'])


    return fixRvarFS_df



       
            