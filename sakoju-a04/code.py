"""
Assignment 4, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""

import sys
import fileinput
import os
import copy
from collections import defaultdict, deque
from heapq import heappush
from collections import defaultdict, Counter
import math
import numpy as np
import time
import pandas as pd
import random
import re
sys.setrecursionlimit(5000)

count = 0
result = []
mcv_flag = False
restart = False

def dfs_backtracking(graph, color_count):
    vrtx_count = len(graph)
    colors = {e:0 for e in list(graph.keys()) }
    color_count = color_count
    global count 
    count = 0
    #check constraint at each vertex
    def is_safe(vertex, colors, color):
        for neighbor in graph[vertex]:
            if neighbor == vertex:
                return False
            if colors[neighbor] == color:
                return False
        return True

    def dfs(vertex, colors, color):
        global count, result
        #if end of graph (last vertex)
        if vertex == vrtx_count + 1:
            result.append(colors)
            return True
        
        # print(colors)
        for color in range(1, color_count + 1):
            if is_safe(vertex, colors, color):
                colors[vertex] = color
                count += 1
                if dfs(vertex + 1, colors, color):
                    return True
                #backtrack
                colors[vertex] = 0

        return False
    
    if dfs(1, colors, 1):
        return colors
    else:
        return None

def dfs_backtracking_restart(graph, color_count):
    vrtx_count = len(graph)
    colors = {e:0 for e in list(graph.keys()) }
    color_count = color_count
    global count, i, counter
    count = 0
    #check constraint at each vertex
    def is_safe(vertex, colors, color):
        for neighbor in graph[vertex]:
            if neighbor == vertex:
                return False
            if neighbor in colors and colors[neighbor] == color:
                return False
        return True

    def dfs(vertex, colors, color, result):
        global count, i
        i += 1
        #if end of graph (last vertex)
        if vertex == vrtx_count + 1:
            result.append(colors)
            return True

        for color in range(1, color_count + 1):
            if is_safe(vertex, colors, color):
                colors[vertex] = color
                count += 1
                if dfs(vertex + 1, colors, color, result):
                    return True
                #backtrack
                colors[vertex] = 0
                    
        return False
    
    if dfs(1, colors, 1, result):
        return colors
    else:
        return None

class CSP:
    def __init__(self, variables, domains, neighbors, mcv_flag, restart):
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.mcv = mcv_flag
        self.restart = restart
        self.counter = 0
        self.count = 0
        self.is_loop = False
    
    def is_safe(self, var, value, assignment):
        for neighbor in self.neighbors[var]:
            if neighbor == var:
                self.is_loop = True
                return False
            if neighbor in assignment and assignment[neighbor] == value:
                    return False
        return True
    
def recursive_backtracking(csp, assignment):
    if len(assignment) == len(csp.variables):
        return assignment
    var = select_unassigned_variables(csp, assignment)
    
    for value in order_domain_values(var, assignment, csp):
        
        if csp.is_safe(var, value, assignment):
            assignment[var] = value
            csp.counter += 1
        #reduce domain of neighboring variables
        inferences = forward_check(csp, var, value, assignment)

        if inferences is not None:
            r = recursive_backtracking(csp, assignment)
            
            if r is not None:
                return r
        
        #backtrack & restore domain of neighboring variables
        restore_domains(csp, inferences)
        del assignment[var]
    
    return None

def select_unassigned_variables(csp, assignment):
    unassigned = [v for v in csp.variables if v not in assignment]
    
    if csp.mcv and not csp.restart:
        #return first min value
        return min(unassigned, key=lambda var: len(csp.domains[var]))
    elif csp.mcv and csp.restart:
        #return random min from min list of unassigned
        min_val = min(unassigned, key=lambda var: len(csp.domains[var]))
        min_values = [v for v in unassigned if v == min_val]
        return random.choice(min_values)
    elif not csp.mcv and csp.restart:
        #sort it and return a random from sorted
        sorted_unassigned = sorted(unassigned, key=lambda var: len(csp.domains[var]))
        return random.choice(sorted_unassigned)
    else:
        # sorted_unassigned = sorted(unassigned, key=lambda var: len(csp.domains[var]))
        return unassigned[0]

def order_domain_values(var, assignment, csp):
    return csp.domains[var]

def restore_domains(csp, inferences):
    if inferences:
        for var, domain in inferences.items():
            csp.domains[var] = domain  
    
def forward_check(csp, var, value, assignment):
    inferences = {}
    for neighbor in csp.neighbors[var]:
        # csp.counter += 1
        if neighbor not in assignment:
            inferences[neighbor] = list(csp.domains[neighbor])
            csp.domains[neighbor] = [v for v in csp.domains[neighbor] 
                                     if csp.is_safe(neighbor, v, assignment)]
            
            if not csp.domains[neighbor]:
                #no possible values left, domain wipe out!
                return None
            # else:
            #     csp.counter += len(csp.domains[neighbor])
    return inferences

def solve_csp(csp, restart):
    n = len(csp.variables)
    i = 0
    res = None
    if restart:
        res = recursive_backtracking(csp, {})
        while int(np.floor(n*np.power(1.3, i))) <= csp.counter:
            # csp.counter = 0
            i += 1
            res = recursive_backtracking(csp, {})
    else:
        res = recursive_backtracking(csp, {})
    if csp.is_loop:
        csp.counter = 0
    return res

def is_safe(graph, vertex, color, assignment):
    global count
    for neighbor in graph[vertex]:
        if neighbor == vertex:
            count = 0
            return False
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True

def forward_checking(graph, vertex, color, assignment, remaining_colors):
    for neighbor in graph[vertex]:
        if neighbor not in assignment:
            if color in remaining_colors[neighbor]:
                remaining_colors[neighbor].remove(color)
                if not remaining_colors[neighbor]:
                    return False
    return True 

def backtrack_fc(graph, colors, assignment,vertex, remaining_colors):
    global count
    if vertex == len(graph) + 1:
        return True
    
    for color in colors:
        if is_safe(graph, vertex, color, assignment):
            assignment[vertex] = color
            count += 1
            temp_rem_colors = {vertex:list(remaining_colors[vertex]) for vertex in remaining_colors}
            
            if forward_checking(graph, vertex, color, assignment, remaining_colors):
                if backtrack_fc(graph, colors, assignment,vertex+1, remaining_colors):
                    return True 
            
            remaining_colors = temp_rem_colors
            del assignment[vertex]
    return False

def fc(graph, color_count):
    colors = list(range(1,color_count+1))
    assignment = {}
    remaining_colors = {vertex:list(colors) for vertex in graph.keys()}
    
    if backtrack_fc(graph, colors, assignment, 1, remaining_colors):
        return assignment 
    else:
        return None

def main():
    
    arr = sys.argv
    # print(arr)
    restart = False
    mcv_flag = False
    color_count = 0
    for arg in arr:
        restart = True if arg.startswith('-') and "restart" in arg else False
        if "code.py" in arg:
            continue
        if arg.isnumeric():
            color_count = int(arg)
        elif "dfs" in arr:
            alg = "dfs"
        elif "fc" in arg:
            alg = "fc"
        elif "mcv" in arg:
            alg = "mcv"
            mcv_flag = True
        elif arg.startswith('-') and "restart" in arg:
            restart = True
        else:
            color_count = int(arg)
        # elif arg.isdigit() :
        #     # print(arg)
        #     print(int(arg))
        #     color_count = int(arg)
    
    vrtx_count = 0
    edge_count = 0
    lines = sys.stdin.readlines()
    lines = [line for line in lines if not line.startswith("c") and len(line) > 0]
    if "p" in lines[0] and "e" in lines[0]:
        vrtx_count, edge_count = lines[0].strip().split()[-2:]
        vrtx_count = int(vrtx_count)
        edge_count = int(edge_count)
        
    graph = {e: [] for e in range(1, vrtx_count + 1) }
    
    for line in lines[1:]:
        if "e" in line:
            v1, v2 = [int(l) for l in line.strip().split()[-2:]] 
            
            graph[v1] += [v2]
            graph[v2] += [v1]
            
    # print(vrtx_count, edge_count)
    # print(graph)
    
    global count, result, i
    res = {}
    # print(alg, color_count)
    if alg == "dfs":
        
        count = 0
        i = 0
        result = []
        counter = 0
        # print(restart, color_count)
        if restart:
            # print("restart")
            res = dfs_backtracking_restart(graph, color_count)
            while np.floor(len(graph)*np.power((1.3), counter)) <= count:
                # print("restart")
                i = 0
                counter += 1
                res = dfs_backtracking_restart(graph, color_count)
        else:
            res = dfs_backtracking(graph, color_count)
            # print(res)
    elif alg == "fc":
        count = 0
        i = 0
        result = []
        counter = 0
        if restart:
            # print("restart")
            res = fc(graph, color_count)
            while np.floor(len(graph)*np.power((1.3), counter)) <= count:
                # print("restart")
                i = 0
                counter += 1
                res = fc(graph, color_count)
        else:
            res = fc(graph, color_count)
    elif alg == "mcv":
        col = [c for c in range(1,color_count+1)]
        variables = list(graph.keys())
        domains = {var:col for var in variables}
        csp = CSP(variables, domains, graph, True, restart)
        res = solve_csp(csp, restart)
        count = csp.counter
        
    results = [["s col %d"%(color_count)]]
    
    if res:
        # print(res)
        for k,v in res.items():
            results.append(["l %d %d"%(k,v)])
        c = count
        for r in results:
            print(*r)
    else:
        c = count
        print("No solution.")
    print(f"{count} branching nodes explored")

if __name__ == "__main__":
    main()