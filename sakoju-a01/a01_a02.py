# -*- coding: utf-8 -*-
"""
Author: Sushma Akoju
"""

import sys
import os
import copy
from collections import defaultdict, deque
from heapq import heappush
from collections import defaultdict
import math
import numpy as np


class Graph:
    def __init__(self, graph_dict=None) -> None:
        self.graph_dict = graph_dict or {}

    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        self.graph_dict.setdefault(B, {})[A] = distance

    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a,{})
        if b is None:
            return links
        else:
            links.get(b)

    def nodes(self):
        nodes = set([k for k in self.graph_dict.keys()]).union(set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()]))
        return list(nodes)

class Problem:

    def __init__(self, grid_world, goal, actions, hmode=0):
        self.start_state = copy.deepcopy(grid_world)
        self.grid_world = grid_world
        self.goal = goal
        self.actions = actions
        self.dirt_locations = []
        self.num_dirt = 0
        self.solution = []
        # self.graph = graph or Graph()
        self.R = 0
        self.C = 0
        self.start_loc = tuple()
        self.actions_result = []
        self.expanded_nodes = 0
        self.generated_nodes = 0
        self.hmode = hmode

    def initialize(self, R, C, r0, c0, dirt_loc, num_dirt ):
        self.R = R
        self.C = C
        self.start_loc = (r0,c0)
        self.dirt_locations = dirt_loc
        self.num_dirt = num_dirt

    def get_actions(self, state):
        return list(self.graph.get(state).keys())

    # def goal_test(self, state):
    #     if isinstance(self.goal, list):
    #         return any(x is state for x in self.goal)
    #     else:
    #         return state == self.goal

    def get_solution(self):
        return self.solution

class Node:
    def __init__(self, grid_world, parent, loc:tuple, action:str, val, hmode, cost=0):
        self.data = copy.deepcopy(grid_world)
        self.state = (loc, self.data)
        self.parent = parent #parent for this node
        self.action = action #action that led to this node
        self.pathcost = cost #cost until now
        self.depth = 0 #depth from this node to start
        # self.problem = problem
        self.heuristic = self.get_heuristic(hmode)
        self.fn = self.pathcost + self.heuristic

        if parent:
            self.depth = parent.depth + 1
        self.x, self.y = loc
        self.loc = loc
        self.cell = val

    def is_clean(self):
        # print(self.cell, self.x, self.y)
        if self.cell != "*":
            return True
        else:
            return False
        # else:
        #     print(self.cell,problem.start_state[self.x][self.y] )
        #     return True

    def remove_dirt(self, problem):
        # print("remove dirt at:",self.x, self.y, self.data[self.x][self.y], self.cell)
        # problem.dirt_locations.remove((self.x, self.y))
        self.cell = " "
        self.data[self.x][self.y] = " "
        # problem.grid_world[self.x][self.y] = " "
        # problem.num_dirt -= 1

    def get_heuristic(self, problem, hmode=0):
        """ heuristic function:
            hmode = 0, h(n) = 0
            hmode=1: h(n) = manhattan distance i.e. distance from current node to closest dirt cell
            hmode = 2, h(n) min of l2 norm for current node to from the dirt cells. (returns distance from nearest dirt cell)
            hmode = 3, h(n) min of l2 norm and manhattan for current node to the dirt cells. (returns distance from nearest dirt cell)
        """
        if hmode == 0:
            m = float('inf')
            m = 0
            return m

        elif hmode == 1:
            m = float('inf')
            for loc in problem.dirt_locations:
                x, y = loc
                dx = abs(x - self.x )
                dy = abs(y - self.y)
                temp = dx + dy
                if temp < m:
                    m = temp
            return m

        elif hmode == 2:
            m = float('inf')
            dirt_arr = np.asarray(problem.dirt_locations)
            arr = np.asarray([[self.x,self.y]]*problem.dirt_locations)
            dxy = dirt_arr - arr
            l2 = np.linalg.norm(dxy, axis=1)
            m1, i = np.min(l2), np.argmin(l2)
            return m1
        else:
            m = float('inf')
            m = float('inf')
            dirt_arr = np.asarray(problem.dirt_locations)
            arr = np.asarray([[self.x,self.y]]*problem.dirt_locations)
            dxy = dirt_arr - arr
            l2 = np.linalg.norm(dxy, axis=1)
            m1, i = np.min(l2), np.argmin(l2)
            m = np.sum(dab, axis=1)
            m2, j = np.min(m), np.argmin(m)
            # print("l2_min, index", m1, i)
            # print("manhattan, index", m2, j)
            return min(m1, m2)

def fill_grid(file_path, hmode):
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
    problem = Problem(grid_world, goal_state, ["N", "S", "W", "E", "V"], hmode)
    problem.R = R
    problem.C = C
    problem.num_dirt = num_dirt_locations
    problem.dirt_locations = dirt_locations
    problem.start_loc = (r0, c0)
    return problem, grid_world

def expand(node, grid_world, problem):
    r = node.x
    c = node.y
    child_nodes = []
    # print(r,c)
    dr = [-1,+1, 0, 0] # actions for each possible adjacent cell
    dc = [0, 0, +1, -1]
    #for this node, explore all adjacent cells, respective actions and the path costs
    for k in range(0,4):
        row1 = r + dr[k]
        col1 = c + dc[k]

        if row1 >= problem.R or col1 >= problem.C:
            # print("less than R or C")
            continue
        if row1 < 0 or col1 < 0:
            # print("less than 0")
            continue
        # blocked (visited wil be checked in individual algorithms)
        if grid_world[row1][col1] == "#":
            # print("blocked")
            continue


        child_node = Node(node.state[1], node, (row1, col1), '', grid_world[row1][col1],problem.hmode, 0)
        child_node.parent = node
        # print(row1, col1, grid_world[row1][col1])
        #all actions cost equal costs
        # if k == 0 and
        if k == 0:
            if not child_node.is_clean():
                # print(child_node.x, child_node.y, child_node.cell)
                problem.actions_result.append("S")
                problem.actions_result.append("V")
                child_node.action = "V"
                # print("k, x,y :", k, row1, col1)
                child_node.remove_dirt(problem)
                child_node.pathcost  = child_node.parent.pathcost + 2
            else:
                child_node.action = "S"
                problem.actions_result.append("S")
                child_node.pathcost  = child_node.parent.pathcost + 1
            child_nodes.append(child_node)
        if k == 1:
            if not child_node.is_clean():
                child_node.action = "V"
                problem.actions_result.append("N")
                problem.actions_result.append("V")
                # print("k, x,y :", k, row1, col1)
                child_node.remove_dirt(problem)
                child_node.pathcost  = child_node.parent.pathcost + 2
            else:
                child_node.action = "N"
                problem.actions_result.append("N")
                child_node.pathcost  = child_node.parent.pathcost + 1
            child_nodes.append(child_node)
        if k == 2:
            if not child_node.is_clean():
                child_node.action = "V"
                problem.actions_result.append("E")
                problem.actions_result.append("V")
                # print("k, x,y :", k, row1, col1)
                child_node.remove_dirt(problem)
                child_node.pathcost  = child_node.parent.pathcost + 2
            else:
                child_node.action = "E"
                problem.actions_result.append("E")
                child_node.pathcost  = child_node.parent.pathcost + 1
            child_nodes.append(child_node)
        if k == 3 :
            if not child_node.is_clean():
                child_node.action = "V"
                problem.actions_result.append("W")
                problem.actions_result.append("V")
                # print("k, x,y :", k, row1, col1)
                child_node.remove_dirt(problem)
                child_node.pathcost  = child_node.parent.pathcost + 2
            else:
                child_node.action = "W"
                child_node.pathcost  = child_node.parent.pathcost + 1
            child_nodes.append(child_node)
        # if k == 4 and child_node.is_clean(problem):
        #     child_node.action = "N"
        #     actions.append("W")
        #     child_node.pathcost  = child_node.parent.pathcost + 1
        #     child_nodes.append(child_node)
    return child_nodes


    # def expand(self, parent):
    #     """Get all nodes reachable in one step"""
    #     return [self.child(self.state, self, action) for action in input.actions(self.state)]

def heuristic_generic(node, problem, hmode=0):
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
        return min(D * (dx+dy), D * math.sqrt(dx*dx +dy*dy))

def heuristic(self, node, problem, hmode=0):
    """ heuristic function, hmode=0: manhattan distance i.e. distance from current node to closest dirt cell
    """

    if hmode == 0:
        #h(n) = 0
        return 0
    elif hmode == 1:
        m = float('inf')
        for loc in problem.dirt_locations:
            x, y = loc
            dx = abs(x - node.x )
            dy = abs(y - node.y)
            temp = dx + dy
            if temp < m:
                m = temp
        return m

    elif hmode == 2:
        dirt_arr = np.asarray(problem.dirt_locations)
        arr = np.asarray([[node.x,node.y]]*problem.dirt_locations)
        dxy = dirt_arr - arr
        l2 = np.linalg.norm(dxy, axis=1)
        m1, i = np.min(l2), np.argmin(l2)
        m = np.sum(dab, axis=1)
        m2, j = np.min(m), np.argmin(m)
        # print("l2_min, index", m1, i)
        # print("manhattan, index", m2, j)
        return min(m1, m2)

def bfs(problem, battery):
    """Breadth first search over a graph.

    Args:
        input (_type_): _description_
        battery (_type_): _description_

    Returns:
        _type_: _description_
    """
    node = Node(problem.start_state, None, problem.start_loc, "", "@",problem.hmode, 0)

    if node.state[1] == problem.goal:
        return node
    frontier = deque([node])
    visited = []
    while frontier:
        node = frontier.popleft()

        if node.state not in visited:
            visited.append(node.state)

        child_nodes = expand(node, start_node.state[1], problem)
        for child in child_nodes :
            if child.state not in visited and child not in frontier:
                if node.state[1] == problem.goal:
                    print("Goal")
                    problem.expanded_nodes = len(visited)
                    problem.generated_nodes = len(frontier)
                    return node
                frontier.append(child)
    return None

def dfs(problem, battery):
    """Breadth first search over a graph.

    Args:
        input (_type_): _description_
        battery (_type_): _description_

    Returns:
        _type_: _description_
    """
    node = Node(problem.start_state, None, problem.start_loc, "", "@",problem.hmode, 0)

    if node.state[1] == problem.goal:
        return node
    frontier = deque([node])
    visited = []
    while frontier:
        node = frontier.pop()

        child_nodes = expand(node, start_node.state[1], problem)
        for child in child_nodes :

            if node.state not in visited:
                visited.append(node.state)

            if child.state not in visited and child not in frontier:
                if node.state[1] == problem.goal:
                    print("Goal")
                    problem.expanded_nodes = len(visited)
                    problem.generated_nodes = len(frontier)
                    return

                frontier.append(child)
    return None

def ucs(problem, battery):
    """uniform cost search over a graph.

    Args:
        input (_type_): _description_
        battery (_type_): _description_

    Returns:
        _type_: _description_
    """
    node = Node(problem.start_state, None, problem.start_loc, "", "@",problem.hmode, 0)

    if node.state[1] == problem.goal:
        return node
    frontier = deque([node])
    visited = []
    while frontier:
        frontier = deque(sorted(frontier, key=lambda x:x.pathcost))
        node = frontier.popleft()

        if node.state not in visited:
            visited.append(node.state)

        child_nodes = expand(node, start_node.state[1], problem)
        for child in child_nodes :
            if child.state not in visited and child not in frontier:
                if node.state[1] == problem.goal:
                    print("Goal")
                    problem.expanded_nodes = len(visited)
                    problem.generated_nodes = len(frontier)
                    return node
                frontier.append(child)
    return None

def ucs_heuristic(problem, battery):
    """uniform cost search over a graph.

    Args:
        input (_type_): _description_
        battery (_type_): _description_

    Returns:
        _type_: _description_
    """
    node = Node(problem.start_state, None, problem.start_loc, "", "@",problem.hmode, 0)

    if node.state[1] == problem.goal:
        return node
    frontier = deque([node])
    visited = []
    while frontier:
        frontier = deque(sorted(frontier, key=lambda x:x.fn))
        node = frontier.popleft()

        if node.state not in visited:
            visited.append(node.state)

        child_nodes = expand(node, start_node.state[1], problem)
        for child in child_nodes :
            if child.state not in visited and child not in frontier:
                if node.state[1] == problem.goal:
                    print("Goal")
                    problem.expanded_nodes = len(visited)
                    problem.generated_nodes = len(frontier)
                    return node
                frontier.append(child)
    return None

def a_star(problem, battery):
    """uniform cost search over a graph.

    Args:
        input (_type_): _description_
        battery (_type_): _description_

    Returns:
        _type_: _description_
    """
    node = Node(problem.start_state, None, problem.start_loc, "", "@",problem.hmode, 0)

    if node.state[1] == problem.goal:
        return node
    frontier = deque([node])
    visited = []
    while frontier:
        frontier = deque(sorted(frontier, key=lambda x:x.fn))
        node = frontier.popleft()

        if node.state not in visited:
            visited.append(node.state)

        child_nodes = expand(node, start_node.state[1], problem)
        for child in child_nodes :
            if child.state not in visited and child not in frontier:
                if node.state[1] == problem.goal:
                    print("Goal")
                    problem.expanded_nodes = len(visited)
                    problem.generated_nodes = len(frontier)
                    return node
                frontier.append(child)
    return None

def gbfs(problem, battery):
    """greedy best first search.
    """
    node = Node(problem.start_state, None, problem.start_loc, "", "@",problem.hmode, 0)

    if node.state[1] == problem.goal:
        return node
    frontier = deque([node])
    visited = []
    while frontier:
        frontier = deque(sorted(frontier, key=lambda x:x.heuristic))
        node = frontier.popleft()

        if node.state not in visited:
            visited.append(node.state)

        child_nodes = expand(node, start_node.state[1], problem)
        for child in child_nodes :
            if child.state not in visited and child not in frontier:
                if node.state[1] == problem.goal:
                    print("Goal")
                    problem.expanded_nodes = len(visited)
                    problem.generated_nodes = len(frontier)
                    return node
                frontier.append(child)
    return None

def main():
    
    if sys.argv[3] == "-battery":
        battery = True
    if sys.argv[1] == "depth-first":
        # print("Depth first search")
        dfs(data, battery)
    elif sys.argv[1] == "breadth-first":
        # print("Uniform cost search")
        bfs(data, battery)
    elif sys.argv[1] == "uniform-cost":
        # print("Uniform cost search")
        unifrom_cost(data, battery)

    print("*path", sep='\n')
    print(f"nodes expanded {node_len}")
    print(f"nodes expanded {generated}")
    
def test():
    
    file_paths = ["/content/test1.vw", "/content/tiny-1.vw", "/content/tiny-2.vw",
                "/content/small-1.vw", "/content/hard-1.vw", "/content/hard-2.vw"]

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        problem, grid_world = fill_grid(file_path, 0)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        dfs(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        print(len(data))
        problem, grid_world = fill_grid(file_path, 0)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        bfs(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        print(len(data))
        problem, grid_world = fill_grid(file_path, 0)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        ucs(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        print(len(data))
        problem, grid_world = fill_grid(file_path, 1)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        a_star(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        print(len(data))
        problem, grid_world = fill_grid(file_path, 2)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        a_star(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        print(len(data))
        problem, grid_world = fill_grid(file_path, 3)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        a_star(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = lines[2:]
        print(len(data))
        problem, grid_world = fill_grid(file_path, 1)
        start_node = Node(problem.start_state, None, problem.start_loc, "",problem.hmode, 0)
        print(problem.start_state, problem.dirt_locations)
        gbfs(problem, False)
        print(problem.R, problem.C, problem.expanded_nodes, problem.generated_nodes, len(problem.actions_result))


def adj_example_test_code():

    g = copy.deepcopy(problem.start_state)
    actions = {}
    rc = np.asarray([2,1])
    t = np.asarray([[2,1]]*5)
    drc = np.asarray([[-1,0], [1,0], [0,0], [0,1], [0,-1]])
    arr = t+drc
    next_layer = {}
    for i, a in enumerate(arr):
        next_layer[i]= a
    mask = np.any(arr < 0, axis=1)
    print(mask)
    arr = arr[~mask]
    print(arr)
    mask = arr[:, 0] >= 3
    print(mask)
    arr = arr[~mask]
    print(arr)
    mask = arr[:,1] >= 4

    arr = arr[~mask]
    print([(a[0], a[1]) for a in arr.tolist()])
    locs = [(a[0], a[1]) for a in arr.tolist()]
    for i, loc in enumerate(locs):
        x,y = loc
        print("i,loc", i,loc)
        if g[x][y] == "*":
            actions[loc] = "V"
        elif g[x][y] == "_":
            tmp = rc-np.asarray([x,y])
            print("tmp",tmp)
            # if tmp[]
            actions[loc] = "V"
        elif g[x][y] == "#":
            continue

    print(actions)

    # # Find rows with values below zero
    # mask = np.any(arr < 0, axis=1)

    # # Remove the rows
    # arr = arr[~mask]