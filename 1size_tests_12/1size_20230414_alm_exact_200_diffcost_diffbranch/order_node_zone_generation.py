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

# Suppressing warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def order_node_and_zone_generation_one_branch(branch_all_nodes_input, area_non_walk_df, depot_node, G, longest_distance_from_depot):
    """
    This function generates the node order and zone for one branch of the network.
    It removes dummy nodes, calculates distances, sorts nodes based on distance,
    and returns lists of node order and node distance order after processing.

    Parameters:
        - branch_all_nodes_input: List of branches containing nodes
        - area_non_walk_df: DataFrame containing information about nodes
        - depot_node: The starting node in the network
        - G: Graph representing the network
        - longest_distance_from_depot: Longest distance from the depot node

    Returns:
        - node_order_list_after: List of node orders after processing
        - node_distance_order_list_after: List of node distance orders after processing
    """

    def remove_dummy_nodes():
        """
        This nested function removes dummy nodes from the list of branches.
        It iterates over each branch and filters out dummy nodes,
        returning the list of branches after removing dummy nodes.

        Returns:
            - branch_all_nodes_after_deleting: List of branches after removing dummy nodes
        """

        branch_all_nodes = branch_all_nodes_input
        num_real_nodes = area_non_walk_df.shape[0] + 1 # Calculate the total number of real nodes in the area
        branch_all_nodes_after_deleting = []

        # Iterate over each branch in the input list of branches
        for i in range(len(branch_all_nodes)):
            cur_branch_old = branch_all_nodes[i]

            # Initialize an empty list to store nodes of the current branch after removing dummy nodes
            cur_branch_new = []
            for j in range(len(cur_branch_old)):
                # Get the index of the current node in the branch
                cur_node_old = cur_branch_old[j]
                # Check if the current node is not a dummy node (its index is less than the total number of real nodes)
                if cur_node_old >= num_real_nodes:
                    # If it's a dummy node, skip it
                    continue
                else:
                    # If it's a real node, add it to the list of nodes for the current branch after removing dummy nodes
                    cur_branch_new.append(cur_node_old)

            # If the current branch still has nodes after removing dummy nodes, add it to the list of branches after deleting
            if len(cur_branch_new) > 0:
                branch_all_nodes_after_deleting.append(cur_branch_new)

        return branch_all_nodes_after_deleting

    branch_all_nodes_after_deleting = remove_dummy_nodes()

    # Re-indexing after removing dummy nodes
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
        """
        This nested function calculates the shortest path lengths
        from the depot node to each node in the branches after rebranching.
        It returns a dictionary containing path lengths for each branch.

        Returns:
            - path_lengths: Dictionary containing path lengths for each branch
        """

        # Initialize a dictionary to store path lengths for each branch
        path_lengths={}

        # Iterate over each branch in the list of re-branched indexes
        for j in range(len(all_index_after_rebranching)):
            # Initialize an empty list to store path lengths for the current branch
            path_lengths[j]=[]
            # Set the current origin node as the depot node
            cur_orig_node=depot_node
            # Get the nearest points for each node in the current branch
            near_points=all_index_after_rebranching[j]

            # Iterate over each near point in the current branch
            for i in range(len(near_points)):
                # Get the coordinates of the current destination node
                cur_dest=(all_nodes_df.iloc[near_points[i]]['pad_nodes_lat'], all_nodes_df.iloc[near_points[i]]['pad_nodes_lon'])
                # Get the nearest node in the graph G to the current destination
                cur_dest_node=ox.get_nearest_node(G, cur_dest)
                # Compute the shortest path length from the current origin to the current destination node
                cur_short_path_length=nx.shortest_path_length(G, cur_orig_node, cur_dest_node, weight='length')
                # Append the shortest path length to the list of path lengths for the current branch
                path_lengths[j].append(cur_short_path_length)
        return path_lengths
                    
    path_lengths_after=get_distance_sort_list_after()

    def sort_distance_node_after():
        """
        This nested function sorts distance nodes after rebranching.
        It assigns distance values and assignment keys to respective lists
        and returns lists of distance beyond and assignment list.

        Returns:
            - distance_beyond: List of distance beyond for each node
            - assignment_list: List of assignment keys for each node
        """

        distance_beyond=[1000000 for i in range(all_nodes_df.shape[0])]
        assignment_list=[1000000 for i in range(all_nodes_df.shape[0])]
        assignment_key=0

        # Iterate over each branch in the path lengths after rebranching
        for j in range(len(path_lengths_after)):
            cur_branch=path_lengths_after[j]
            cur_branch_node=all_index_after_rebranching[j]

            # Iterate over each path length and branch node
            for i in range(len(cur_branch)):
                cur_dist=cur_branch[i]
                cur_index=cur_branch_node[i]

                # Assign distance and assignment key to respective lists
                distance_beyond[cur_index]=cur_dist
                assignment_list[cur_index]=assignment_key
            # Increment assignment key for next branch
            assignment_key+=1
        
        return distance_beyond, assignment_list

    distance_beyond_after, assignment_list_after=sort_distance_node_after()
    # 'group_index' column contains values from assignment_list_after, indicating the group index associated with each node
    # 'distance' column contains values from distance_beyond_after, representing the distance beyond each node.
    orde_node_df_after=pd.DataFrame(list(zip(assignment_list_after,distance_beyond_after)), columns=['group_index','distance'])

    def order_distance_node_after():
        """
        This nested function orders distance nodes after rebranching.
        It sorts a DataFrame by distance in ascending order,
        resets the index, renames columns, and organizes nodes
        and distances into separate lists for each group.

        Returns:
            - node_order_list: List of node orders after rebranching
            - node_distance_order_list: List of node distance orders after rebranching
        """

        # Sort the DataFrame by distance in ascending order
        ranked_node_df=orde_node_df_after.sort_values(by=['distance'], ascending=True)
        # Reset the index of the DataFrame
        ranked_node_df=ranked_node_df.reset_index()
        # Rename the index column to node_index
        ranked_node_df = ranked_node_df.rename(columns={'index': 'node_index'})
        # Initialize lists to store node order and node distance order
        node_order_list=[[] for i in range(len(path_lengths_after))]
        node_distance_order_list=[[] for i in range(len(path_lengths_after))]
        # Iterate over the rows of the DataFrame
        for i in range (ranked_node_df.shape[0]):
            # Get the group index, node index, and distance from the DataFrame
            cur_group=int(ranked_node_df.iloc[i]['group_index'])
            cur_node=int(ranked_node_df.iloc[i]['node_index'])
            cur_dist=ranked_node_df.iloc[i]['distance']

            # Check if the group index is less than a threshold value
            if cur_group<1000000:
                node_order_list[cur_group].append(cur_node)
                node_distance_order_list[cur_group].append(cur_dist)
            
        return node_order_list, node_distance_order_list

    node_order_list_after, node_distance_order_list_after=order_distance_node_after()

    return node_order_list_after, node_distance_order_list_after

def get_combine_node_order(across_branch_nodes_input, start_num_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot):
    """
    This function gets the combined node order for all decision branches.
    It iterates over each decision branch, generates node order and zone,
    and appends the node order list for each decision branch to the final list.

    Parameters:
        - across_branch_nodes_input: List of decision branches
        - start_num_branch: Starting number of branches
        - area_non_walk_df: DataFrame containing information about nodes
        - depot_node: The starting node in the network
        - G: Graph representing the network
        - longest_distance_from_depot: Longest distance from the depot node

    Returns:
        - all_node_order: Combined node order for all decision branches
    """

    # Initialize an empty list to store node orders for all decision branches
    all_node_order=[]

    # Iterate over each decision branch in the input list
    for i in range(len(across_branch_nodes_input)):
        # Get the current decision branch
        cur_decision_branch=across_branch_nodes_input[i]
        # Call the function to generate node order and zone for the current decision branch
        # Return node order list and node distance order list for the current decision branch
        node_order_list_decision, node_distance_order_list_decision=order_node_and_zone_generation_one_branch(cur_decision_branch, area_non_walk_df, depot_node, G, longest_distance_from_depot)

        # Append the node order list for the current decision branch to the list of all node orders
        all_node_order.append(node_order_list_decision)
        
    
    return all_node_order

def get_final_decision_zone_list(all_decision_list_result_input):
    """
    This function gets the final decision zone list from all decision lists.
    It iterates over all decision lists, combines unique zones, and returns
    the final decision zone list.

    Parameters:
        - all_decision_list_result_input: List containing decision lists

    Returns:
        - decision_zone_list: Final decision zone list
    """

    # Initialize the decision zone list with the zones from the first decision list
    decision_zone_list=all_decision_list_result_input[0]

    # Iterate over all other decision lists starting from the second one
    for i in range(1, len(all_decision_list_result_input)):
        # Get the current decision list
        cur_decision_list=all_decision_list_result_input[i]

        # Iterate over each zone in the current decision list
        for j in range(len(cur_decision_list)):
            # Get the current zone
            cur_zone=cur_decision_list[j]
            # Check if the current zone is not already in the decision zone list
            if not cur_zone in decision_zone_list:
                # If not, append the current zone to the decision zone list
                decision_zone_list.append(cur_zone)

    # Return the final decision zone list containing unique zones from all decision lists
    return decision_zone_list




