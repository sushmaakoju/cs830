"""
Assignment 3, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""

#t = 1/20
# total = atan2(y, x) + 
# x ← x + (t · s · cos(total ))
# y ← y + (t · s · sin(total ))
# x ← x + (t · x)
# y ← y + (t · y)

import sys
import fileinput
import os
import copy
from collections import defaultdict, deque
from heapq import heappush
from collections import defaultdict
import math
import numpy as np
import time
import pandas as pd
import math

"""
Assignment 3, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""


import sys
import fileinput
import os
import copy
from collections import defaultdict, deque
from heapq import heappush
from collections import defaultdict
import math
import numpy as np
import time
import pandas as pd
import random


"""
Assignment 3, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""
#t = 1/20
# total = atan2(y, x) + 
# x ← x + (t · s · cos(total ))
# y ← y + (t · s · sin(total ))
# x ← x + (t · x)
# y ← y + (t · y)

import sys
import fileinput
import os
import copy
from collections import defaultdict, deque
from heapq import heappush
from collections import defaultdict
import math
import numpy as np
import time
import pandas as pd
import random

class Problem:
    def __init__(self, data, goal_bias=0.05, collision_freq=0.05):
        self.data = data #input from stdin
        self.grid_world = []
        self.goal_bias = goal_bias
        self.collision_freq = collision_freq
        self.goal_dist = 0.1 # distance of this control
        #30 different angles so a radius of 0.1
        self.goal_theta_list = np.random.uniform(0, 2 * np.pi, size=120)
        
        self.start_x = None
        self.start_y = None
        self.start_loc = tuple()
        
        self.goal_x = None
        self.goal_y = None
        self.goal = None
        self.goal_x_min = 0
        self.goal_y_min = 0
        self.goal_x_max = 0
        self.goal_y_max = 0
        self.maxX = 0
        self.maxY = 0
        self.grid_world = []
        self.obstacle_locations = []
        self.obstacles_x_loc = []
        self.obstacles_y_loc = []
        self.num_obstacles = 0
        self.time_step = 20
        self.min_distance_index = None
        self.min_control = None
        self.motion_tree = []
        self.this_tree = []
        self.iterations = 0
        self.maxiterations = 100
        self.num_nodes = 0

    def get_random_node_with_bias(self, node):
        """Generate 10 indices with goal bias i.e. probability 0.95 for each random_node in 10 controls, and 0.05 for goal
        When to call: Once random controls are generated and before distance to target location is calculated.
        """
        # prob = np.random.random(1)
        # if prob < self.goal_bias:
        #     return self.goal
        # res = np.random.choice(['R','G'],10, p=[0.95, 0.05])
        # 5 random iterations of every 100 iterations are a goal nodes
        k = 5
        # random.sample(self.iterations, 5)
        # if node.theta == None:
        #     print(self.goal.theta)
        if self.iterations%0.05 == 0:
            return self.goal
        else:
            return node

    def get_closest_node(self, this_node, trajectories, to_node):
        """
        Get closest node for this set of 10 trajectories
        Each trajectory has 20 time slices
        """
        distances = {}
        for key, node in trajectories.items():
            theta, s = key
            #check the last slice
            distance = node.get_distance(to_node,theta)
            distances[key] =distance

        #save the index of min distance from distance values to this node
        self.min_distance_index = np.argmin(list(distances.values()))
        #save the control of the respective min distance index
        self.min_control = list(trajectories.keys())[self.min_distance_index]
        nearest_node = trajectories[self.min_control]

        return [nearest_node, self.min_control]
    
    def transform_theta(self, theta):

        if theta < 0:
            theta += 2*np.pi
        
        elif theta > 2*np.pi:
            theta -= 2*np.pi
        
        return theta

    def get_trajectory(self, theta, s, x,y, this_node):
        trajectory = []
        dx =  this_node.dx
        dy = this_node.dy
        for t in range(1,self.time_step+1):
            dt = 1/self.time_step
            
            theta_total = np.arctan2(dy, dx) + theta
            theta_total = self.transform_theta(theta_total)

            dx = dx + (dt * s * np.cos(theta_total))
            dy = dy + (dt * s * np.sin(theta_total))
            x = x + (dt * dx)
            y = y + (dt * dy)
            node = Node(x, y, dx, dy)
            node.parent = this_node
            node.theta = theta
            node.s = s
            
            if not self.check_collision(node):
                trajectory.append(node)
            else:
                return []
        return trajectory
    
    def check_x_obstacle(self, x):
        for xo in self.obstacles_x_loc:
            if x > xo:
                return True
            else:
                continue
        return False

    def check_y_obstacle(self, y):
        for yo in self.obstacles_y_loc:
            if y > yo:
                return True
            else:
                continue
        return False
    
    def check_collision(self, this_node):
        x = this_node.x
        y = this_node.y

        if x <= 0 or x >= self.maxX and self.check_x_obstacle(x) \
            or y <= 0 or y >= self.maxY or self.check_y_obstacle(y): 
            #y in self.obstacles_y_loc:
            return True
        else:
            return False
    
    def steer(self, theta, s, x,y, nearest_node, random_node):
        trajectory = []
        dx =  random_node.x - nearest_node.x
        dy = random_node.dy - nearest_node.y
        for t in range(1,self.time_step+1):
            dt = 1/self.time_step
            
            theta_total = np.arctan2(dy, dx) + theta
            theta_total = self.transform_theta(theta_total)

            dx = dx + (dt * s * np.cos(theta_total))
            dy = dy + (dt * s * np.sin(theta_total))
            x = nearest_node.x + (dt * dx)
            y = nearest_node.y + (dt * dy)
            node = Node(x, y, dx, dy)

            if not self.check_collision(node):
                trajectory.append(node)
                if t > 0:
                    node.parent = trajectory[t-1]
            else:
                return []
        return trajectory

    def get_trajectories(self, this_node, random_node):
        theta_list, s_list, x, y = self.sample_random_targets()
        trajectories = {}
        for i in range(0, len(theta_list)):
            theta = theta_list[i]
            s = s_list[i]
            node = Node(x,y,this_node.dx, this_node.dy)
            node.theta = theta
            # node.s = s
            trajectories[(theta,s)] = node
        state, control = self.get_closest_node(this_node, trajectories, random_node)
        if state != None and len(control) != 0 and not self.check_collision(state):
            trajectory = self.get_trajectory(control[0], control[1], state.x, state.y, this_node)
            if len(trajectory) >0:
                self.num_nodes += 1
                state.s = trajectory[-1].s
                state.node_num = self.num_nodes - 1
                return trajectory, (state, control, self.num_nodes-1)
            else:
                return [], tuple()
        else:
            return [], tuple()
        # return trajectory, (state, control)

    def sample_random_targets(self):
        theta_list = np.random.uniform(0, 2 * np.pi, size=10)
        s_list = np.random.uniform(0, 0.5, size = 10)
        x = np.random.uniform(0, self.maxX)
        y = np.random.uniform(0, self.maxY)
        return theta_list, s_list, x, y
    
    def is_goal1(self, node):
        for i in range(0, len(self.goal_theta_list)):
            theta = self.goal_theta_list[i]
            gnode = Node(self.goal.x,self.goal.y,self.goal.dx, self.goal.dy)
            gnode.theta = theta
            gnode.s = self.goal_dist
            # node.s = s
            if gnode.get_distance(node, theta) <= self.goal_dist:
                return True
            else:
                continue

        return False
    def is_goal(self, node):
        x = node.x
        y = node.y
        if x >= self.goal_x_min and x <= self.goal_x_max and y >= self.goal_y_min and y <= self.goal_y_max:
            return True
        return False
            
    # get true state values for this trajectory
    def get_this_trajectory_paths(self, trajectory):
        pathx = []
        pathy = []
        paththeta = []
        for t in trajectory:
            pathx.append(t.x)
            pathy.append(t.y)
            paththeta.append(t.theta)
        return pathx, pathy, paththeta

    def check_hidden_obstacles(self, tree, parent_node, nearest_node):
        for this_edge in tree:
            child, parent = this_edge
            # dist1 = parent.get_distance(child, child.theta)
            # dist2 = parent_node.get_distance(nearest_node, nearest_node.theta)
            # if dist1 == dist2:
            #     return True
            #check every 1/20th i.e. 0.05 slice
            #child, parent in tree
            
            pathx_t1, pathy_t1, paththeta_t1 = \
                    self.get_this_trajectory_paths(self.get_trajectory(child.theta, child.s, child.x, child.y, parent))
            #the given parent_node and nearest_node
            pathx_t2, pathy_t2, paththeta_t2 = \
                    self.get_this_trajectory_paths(self.get_trajectory(nearest_node.theta, nearest_node.s, nearest_node.x, nearest_node.y, parent_node))
            x_flag = False
            y_flag = False
        
            for x in pathx_t1:
                if x in pathx_t2:
                    x_flag = True
            for y in pathy_t1:
                if y in pathy_t2:
                    y_flag = True
            for theta in paththeta_t1:
                if theta in paththeta_t2:
                    if x_flag and y_flag:
                        return True
        return False

#track the nodes in the tree
class Node:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.theta = None
        self.s = 0
        self.pathx = []
        self.pathy = []
        self.parent = None
        # self.edge = [] #child, parent
        self.children = []
        self.flag = False
        self.node_num = 0

    def get_distance(self, end_node, angle):
        """Get 3-dimensional distance from this_node to end_node
        """
        theta = angle
        alpha = min(abs(self.theta - theta), 2*np.pi - (abs(self.theta - theta)))
        dist = math.sqrt((self.x - end_node.x)**2 + (self.y - end_node.y)**2 + alpha**2)
        return dist

def fill_grid(data):

    # global R, C, goal_state, grid_world, dirt_locations, num_dirt_locations, r0, c0
    # initialize row, column of @ i.e. starting location of the robot
    start_x = None
    start_y = None
    maxX = 0
    maxY = 0
    grid_world = []
    goal = None
    obstacle_locations = []
    num_obstacles = 0
    maxX = int(data[0])
    maxY = int(data[1])
    x, y, goal_x, goal_y = [float(val.strip()) for val in data[-4:]]
    goal = (goal_x, goal_y)
    for i, row in enumerate(data[2:-4], 0):
        this_row = [r for r in list(row) if r != "\n"]
        grid_world.append(this_row.copy())
        
        for j,col in enumerate(this_row, 0):
            if col == "#":
                num_obstacles += 1
                obstacle_locations.append((i, j))

    problem = Problem(grid_world, [])
    problem.maxX = maxX
    problem.maxY = maxY
    problem.num_obstacles = num_obstacles
    problem.obstacle_locations = obstacle_locations
    # print(obstacle_locations)
    if num_obstacles != 0:
        problem.obstacles_x_loc = np.asarray(problem.obstacle_locations)[0,:] 
        problem.obstacles_y_loc = np.asarray(problem.obstacle_locations)[1,:]
    problem.start_x = x
    problem.start_y = y
    problem.start_loc = (x,y)
    problem.goal_x = goal_x
    problem.goal_y = goal_y
    problem.goal = Node(goal_x, goal_y, 0,0)
    problem.goal.theta = 0
    problem.goal.s = 0
    problem.goal_x_min = problem.goal_x - 0.1
    problem.goal_x_max = problem.goal_x + 0.1
    problem.goal_y_min = problem.goal_y - 0.1
    problem.goal_y_max = problem.goal_y + 0.1
    problem.grid_world = copy.deepcopy(grid_world)
    problem.iterations = 0
    return problem, grid_world

def main():
    path = []
    left = False
    
    # print(sys.stdin.name )
    for arg in sys.argv:
        if "-left" in arg:
            Left = True
        
    lines = sys.stdin.readlines()
    # print(lines)
    
    problem, grid_world = fill_grid(lines)
    # start = Node(x,y)
    # print(problem.grid_world)
    # with open("/content/space-0.sw", 'r') as f:
    #     lines = f.readlines()
    # problem, grid_world = fill_grid(lines)
    start_node = Node(problem.start_x, problem.start_y, 0,0)
    start_node.theta = 0
    start_node.parent = Node #child = parent for start_node

    # start_node.theta = 
    problem.maxiterations = 100

    tree = [(None, start_node)] #child, parent
    results = []
    node = start_node
    is_goal_reached = False

    np.random.seed(8)

    theta_list, s_list, x, y = problem.sample_random_targets()
    random_node = Node(x,y, 0,0)

    start_node.theta = np.random.uniform(0, 2 * np.pi)
    start_node.s = np.random.uniform(0, 0.5)

    # theta_list, s_list, x, y = problem.sample_random_targets()
    random_node = Node(x,y, 0,0)
    results.append((start_node.x, start_node.y, 0.0, 0.0, start_node.theta, start_node.s))
    nodes = []
    while not is_goal_reached:
        
        if problem.iterations == 0:
            node1 = start_node
            problem.iterations += 1
        else:
            node1 = Node(x,y,0,0)
            # node.theta = theta_list[0]
            # node.s = s_list[0]
            problem.iterations += 1
        random_node = problem.get_random_node_with_bias(node1)
        #check all 10 possible trajectories to get nearest node from node to random_node
        trajectory, state_control = problem.get_trajectories(node, random_node)
        if len(trajectory) != 0 and len(list(state_control)) != 0:
            nearest_node, control, numnodes = state_control
            # print(numnodes)
            #check if edge is within hidden obstacles (other nodes in the tree)
            # dist_range = node.get_distance(nearest_node, control[0])
            # if not problem.check_hidden_obstacles(tree, nearest_node, node):
                #add edge
            nodes.append(node)
            node.children.append(nearest_node)
            tree.append((nearest_node, node))
            # this_result = [nearest_node.x, nearest_node.y, node.x, node.y,]
            results.append((
                np.round(node.x,6), np.round(node.y,6), np.round(nearest_node.x, 6), 
                np.round(nearest_node.y, 6), np.round(state_control[1][0],6), np.round(state_control[1][1],6)))
            nearest_node.parent = node

            node = nearest_node
            if problem.is_goal(nearest_node):
                is_goal_reached = True
                # break
            else:
                continue
        else:
            continue
    
    print(len(results)-1)
    for res in results[0:-1]:
        print(*res)
    # print(len(results))
    
if __name__ == "__main__":
    main()