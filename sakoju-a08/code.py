"""
Assignment 8, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""

import re
import numpy as np
import heapq
import sys
import fileinput
import os
import copy
import math
import numpy as np
import time
import pandas as pd
import random
sys.setrecursionlimit(5000)

# def h1():
#     t ← 0 (current time)
# record that initial state literals became true at 0
# Q ← I (literals that became true at t)
# until all goals are true or Q is empty,
# Q′ ← ∅
# foreach l ∈ Q,
# foreach a that has l as a precondition,
# if all of a’s preconditions are now true,
# foreach effect e of a,
# if e is not already true,
# record that e became true at t + 1
# add it to Q′
# t ← t + 1
# Q ← Q′
# Then ∑ or max over goal

# ground all predicates with constants
def ground_predicates(raw_input):
    predictes = []
    actions = []
    lines = []
    num_actions = 0
    constants = []
    markers = []
    for i,line in enumerate(raw_input):
        this_line = [l.strip() for l in re.split(r'\s(?=[^)]*(?:\(|$))', line) if l]
        # print(this_line)
        if len(this_line) != 0 and this_line[0]:
            lines.append(this_line)
        else:
            markers.append(i)
            lines.append(str(i))
    # print(markers, len(lines))
    for i,line in enumerate(lines):
        print(line, i)
        if "predicates" in line:
            predicates = line[1:]
        elif 'actions' in line:
            num_actions = int(line[0])
            actions_strings = lines[i:]
        elif "constants" in line:
            constants = line[1:]
        
        
        
    

# standard weighted a* for regular numerical graphs/vaccuum cleaner - not for use for this assignment
def astar(graph, start, goal, heuristic):
    open_set = [(0,start)] # f(n) score and node
    came_from = {} # for reconstructing the path
    g_score = {node: float('inf') for node in graph}    # default cost for all nodes
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)
    
    while open_set:
        curr_f_score, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor, weight in graph[current]:
            temp_g_score = g_score[current] + weight
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current 
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, goal) 
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path   

#**********************************START WEIGHTED A* STRIPS PLANNER******************************************

h = None

def heuristic0(state, goal_state):
    # h0
    
    pass

def heuristic1(state, goal_state):
    # h-goal-lits
    
    pass

def heuristic2(state, goal_state):
    # h1max
    
    pass

def heuristic3(state, goal_state):
    #h1sum
    
    pass

def heuristic(state, goal_state):
    global h
    
    if h == 0:
        heuristic0(state, goal_state)
    if h == 1:
        heuristic1(state, goal_state)
    if h == 2:
        heuristic2(state, goal_state)
    if h == 3:
        heuristic3(state, goal_state)

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic
        
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

# track progression - slide 10 https://www.cs.unh.edu/~ruml/cs730/slides/lecture-15-planning.pdf
class Action:
    def __init__(self, name, preconditions, effects, cost=1):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self.cost = cost
    
    def applicable(self, state):
        return all(precondition in state for precondition in self.preconditions)

    def apply(self, state):
        new_state = set(state)
        for effect in self.effects:
            if effect.startswith("-"):
                new_state.discard(effect[1:])
            else:
                new_state.add(effect)
        return new_state

def weighted_a_star(initial_state, goal_state, actions, heuristic, weight):
    open_set = [Node(initial_state,cost=0,heuristic=heuristic(initial_state, goal_state) )]
    closed_set = set()
    
    while open_set:
        current_node = heapq.heappop(open_set)
        
        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append((current_node.action, current_node.state))
                current_node = current_node.parent 
            return path[::-1]
        closed_set.add(tuple(current_node.state))
        
        for action in actions:
            if action.applicable(current_node.state):
                new_state = action.apply(current_node.state)
                if tuple(new_state) not in closed_set:
                    new_cost = current_node.cost + action.cost
                    new_heuristic = heuristic(new_state, goal_state)
                    new_node = Node(new_state, current_node, action, new_cost, new_heuristic * weight)
                    heapq.heappush(open_set, new_node)
                    
    return None

def main():
    arr = sys.argv
    print(arr)
    h = None
    weight = None
    for arg in arr:
        if arg.isnumeric():
            weight = int(arg)
            # print(weight)
        if "code.py" in arg:
            continue
        elif "h0" in arg:
            h = 0
        elif "h-goal-lits" in arg:
            h = 1
        elif "h1" in arg:
            h = 2
        elif "h1sum" in arg:
            h = 3
        else:
            continue
    
    all_lines = sys.stdin.readlines()
    # lines = [line.strip() for line in all_lines]
    # print(all_lines)
    ground_predicates(all_lines)
    
    # tested for DIMACS cnf formats and works
    
    results = []
    res = {}
    # res = weighted_a_star(initial_state, goal_state, actions, h, weight)
    expanded = 0
    generated = 0
    if res:
        for k,v in res.items():
            results.append(v)
            
    for r in results:
        print(*r)

    print(f"{expanded} nodes expanded")
    print(f"{generated} nodes generated")
    
if __name__ == "__main__":
    main()
