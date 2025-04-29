# CS 830 Spring 2025, Sushma Akoju

# Problem description
# Given a world description, program should return a sequence of actions, that will move the robot and vacuums the dirt.
# Vacuum robot can only move in 4 directions. All actions have same cost.
# Goal: Clean world with min cost.
# A cell can be starting (@) cell or blank (_) or \\blocked (#) or dirty (*).
# Actions : N,S,E,W or V (vacuum)
# States: start, blank, blocked, dirty and end i.e. {@,_,#,*}
# Input is the search type and a file path containing grid world data.
# Implement DFS and Uniform Cost Search
# References from AI 4th edition text book: section: 2.1, 3.2.1 
# References from AI 4th edition text book: UFS/Dijkstra: 3.4.2 & DFS: section 3.4.3
# https://www.youtube.com/watch?v=KiCBXu4P-2Y&t=17s & 

#DFS pseudo code:
#Construct adjacency matrix
#Fix A01 error: don't use node.

import argparse

from typing import *

from enum import Enum
import numpy as np

# parser = argparse.ArgumentParser()
# group = parser.add_mutually_exclusive_group(required=True)
# group.add_argument("depth-first", "depth-first", action='store_true', help = "This option is for Depth First Search")
# group.add_argument("uniform-cost", "uniform-cost", action='store_true', help="This is option for Uniform Cost Search")
# parser.add_argument("-battery", "--battery", action='store_true', help="This is option for Uniform Cost Search")

# args = parser.parse_args()

# search_type = sys.argv[0]


# read the commandline arguments for search: depth-first or uniform_cost
# search = sys.argv[1]
# file_path = sys.argv[2]
# battery = sys.argv[2]

import sys
import os
import copy
from collections import defaultdict, deque
from heapq import heappush
from collections import defaultdict

# file_path = "E:\\UNH\\spring2025\\aispring-2025\\assignments\\assignment-1\\test1.vw"


        
class Problem:
    
    def __init__(self, start_state, goal, actions):
        self.start_state = start_state
        self.goal = goal
        self.actions = actions
        self.dirt_locations = []
        self.num_dirt = 0
        self.solution = []
        self.graph = defaultdict(list)
        self.R = 0
        self.C = 0
        self.start_loc = ()
    
    def initialize(self, R, C, r0, c0, dirt_loc, num_dirt ):
        self.R = R
        self.C = C
        self.start_loc = (r0,c0)
        self.dirt_locations = dirt_loc
        self.num_dirt = num_dirt 

    def get_actions(self, state):
        return list(self.graph.get(state).keys())

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return any(x is state for x in self.goal)
        else:
            return state == self.goal
    
    def get_solution(self):
        return self.solution

class Node:
    def __init__(self, grid_world, parent, loc:tuple, action:str, problem, cost=None):
        self.state = copy.deepcopy(grid_world) #current grid world
        self.parent = parent #parent for this node
        self.action = action #action that led to this node
        self.pathcost = cost #cost until now
        self.depth = 0 #depth from this node to start
        self.problem = problem
        
        if parent:
            self.depth = parent.depth + 1
        self.x, self.y = loc
    
    def is_clean(self):
        if self.state[self.x][self.y] != "*":
            return True
        else:
            return False
    
    def remove_dirt(self, problem):
        problem.num_dirt_loc -= 1
        self.state[self.x][self.y] = " "
        problem.dirt_locations.remove((self.x, self.y))


def fill_grid(file_path):
    assert os.path.exists(file_path), "Path does not exist"
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = lines[2:]
    
    # global R, C, goal_state, grid_world, dirt_locations, num_dirt_locations, r0, c0
    # initialize row, column of @ i.e. starting location of the robot
    r0 = None
    c0 = None

    R = 0
    C = 0
    grid_world = []
    goal_state = []
    dirt_locations = []
    num_dirt_locations = 0
    R = len(data)
    for i, row in enumerate(data, 0):
        this_row = [r for r in list(row) if r != "\n"]
        # print(this_row)
        grid_world.append(this_row.copy())
        goal_state.append(this_row.copy())
        C = len(this_row)
        for j,col in enumerate(this_row, 0):
            # if row[col] != "\n":
            #     this_row.append(row[col])
            # print(this_row[j])
            if col == "@":
                #start row and column
                r0 = i
                c0 = j
                #the current location of the agent
                this_row[j] = " "
            elif col == "*":
                num_dirt_locations += 1
                dirt_locations.append((i, j))
                goal_state[i][j] = " "
    problem = Problem(grid_world, goal_state, ["N", "S", "W", "E", "V"])
    problem.R = R
    problem.C = C
    problem.num_dirt = num_dirt_locations
    problem.dirt_locations = dirt_locations
    problem.start_loc = (r0, c0)
    return problem

def expand(node, grid_world, problem):
    r = node.x
    c = node.y
    child_nodes = []
    
    
    dr = [-1,+1, 0, 0, 0] #5 actions for each possible adjacent cell
    dc = [0, 0, 0, +1, -1]
    #for this node, explore all adjacent cells, respective actions and the path costs
    # and create child nodes for all adjacent cells
    for k in range(0,5):
        row1 = r + dr[k]
        col1 = c + dc[k]
        
        if row1 >= problem.R or col1 >= problem.C:
            continue
        if row1 < 0 or col1 < 0:
            continue
        # blocked (visited wil be checked in individual algorithms)
        if grid_world[row1][col1] == "#":
            continue
        
        # blank i.e. can move next
        if grid_world[row1][col1] == "_":
            row1 = row1 + dr[k]
            col1 = col1 + dc[k]
        print(row1, col1, grid_world[row1][col1])
        
        child_node = Node(node.state, node, (row1, col1), '', problem, 0)
        child_node.parent = node
                #all actions cost equal costs
        if k == 0 and not child_node.is_clean():
            child_node.action = "V"
            child_node.remove_dirt(problem)
            child_node.pathcost  = child_node.parent.cost + 1
            child_nodes.append(child_node)
        if k == 1 and child_node.is_clean():
            child_node.action = "E"
            child_node.pathcost  = child_node.parent.cost + 1
            child_nodes.append(child_node)
        if k == 2 and child_node.is_clean():
            child_node.action = "W"
            child_node.pathcost  = child_node.parent.cost + 1
            child_nodes.append(child_node)
        if k == 3 and child_node.is_clean():
            child_node.action = "S"
            child_node.pathcost  = child_node.parent.cost + 1
            child_nodes.append(child_node)            
        if k == 4 and child_node.is_clean():
            child_node.action = "N"
            child_node.pathcost  = child_node.parent.cost + 1
            child_nodes.append(child_node)
    return child_nodes
    
    
    # def expand(self, parent):
    #     """Get all nodes reachable in one step"""
    #     return [self.child(self.state, self, action) for action in input.actions(self.state)]        
    
def heuristic(self, node, problem, hmode=0):
    dx = abs(node.x - problem.goal.x)
    dy = abs(node.y - problem.goal.y)
    if hmode == 0:
        D = 1
        print("Manhattan distance")
        return D * (dx+dy)
    elif hmode == 2:
        D = 1
        print("Euclidean distance")
        return D * math.sqrt(dx*dx +dy*dy)
    elif hmode == 3:
        D = 1
        print("Min of both heuristics: go with heuristic that gives min cost_to_go")
        return min(D * (dx+dy), D * math.sqrt(dx*dx +dy*dy))
        
 
def bfs(problem, battery):
    """Breadth first search over a graph.

    Args:
        input (_type_): _description_
        battery (_type_): _description_

    Returns:
        _type_: _description_
    """
    node = Node(problem.start)
    
    if problem.goal(node.state):
        return node
    frontier = deque([node])
    visited = set()
    while frontier:
        node = frontier.popleft()
        visited.add(node.state)
        if child in node.expand(problem):
            if child.state not in visited and child.state not in frontier:
                if problem.goal_test(node.state):
                    return node
            frontier.append(child)
    return None

def dfs(input, battery, problem):
    """Searches deepest node in the graph. 
    Frontier (LIFO) tracks which nodes need to explored, that are not yet visited."""
        visited = set()
        stack = [(Node(data))]
        while stack:
            node = stack.pop()
            if node.goal_test(node.state):
                return node
            visited.add(node.state)
            stack.extend(c for c in expand(node, node.state, problem) 
                         if c.state not in visited and c not in frontier )
        return None

def test1():
    
    global R, C, goal_state, grid_world, dirt_locations, num_dirt_locations, r0, c0
    print(grid_world)
    print("Grid world", grid_world)
    print("The start location i.e. row and column are:",r0, c0)
    print("There %d dirt locations are:"%(num_dirt_locations),dirt_locations)
    print("Goal state", goal_state)
    
    # node_len, path, generated = dfs(data, battery)
    # node_len, path, generated = unifrom_cost(data, battery)
    
def main():
    
    global R, C, goal_state, grid_world, dirt_locations, num_dirt_locations, r0, c0
    node_len = None
    path = None
    generated = None
    file_path = sys.argv[2] 
    # print(file_path)
    fill_grid(file_path)
    
    test1()

    if sys.argv[3] == "-battery":
        battery = True
    if sys.argv[1] == "depth-first":
        # print("Depth first search")
        nodes, path, generated = dfs(data, battery)
    elif sys.argv[1] == "uniform-cost":
        # print("Uniform cost search")
        nodes, path, generated = unifrom_cost(data, battery)

    print("*path", sep='\n')
    print(f"nodes expanded {node_len}")
    print(f"nodes expanded {generated}")
    
if __name__ == "__main__":
    main()

