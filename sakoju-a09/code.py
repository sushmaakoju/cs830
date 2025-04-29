"""
Assignment 9, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""

import os
import sys
import random

backups = 0

class MDP:
    def __init__(self, states, actions, transitions, costs,rewards, terminals, goal_state, gamma):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.goal_state = goal_state
        self.values = {s:float('inf') for s in states}
        self.values[goal_state] =  0
        self.num_iterations = None
        self.rewards = rewards
        self.costs = costs
        self.gamma = gamma
        self.terminals = terminals
        self.policy = {s: None for s in states}

    def get_actions(self, state):
        if state in self.terminals:
            return [None]
        else:
            return self.actions

    def get_next_state(self, state, action):
        # used only for rtdp since randomized 
        # given s and a, get nextstate , get all possible next states from transitions (for whcih next state for s given is defined)
        # randomly choose  next state, prob pairs (which is what is supplied in the inputs)
        if state not in self.transitions or action not in self.transitions[state]:
            return None
        all_next_states = self.transitions[state][action]
        state_prob = [p for _,p in all_next_states]
        next_states = [s for s,_ in all_next_states]
        return random.choices(next_states, state_prob)[0] #states and corresponding weights i.e. probabilities

    def get_value(self, state, action):
        # get value prob * sum for each next_state, prob for transitions of this state, action
        if state not in self.transitions or action not in self.transitions[state]:
            return float('inf')
        return sum(prob * (self.costs[state][action] + self.values[next_state]) for next_state, prob in self.transitions[state][action])

    def q_value(self, state, action):
        # get value prob * sum for each next_state, prob for transitions of this state, action
        if state not in self.transitions or action not in self.transitions[state]:
            return float('inf')
        return self.costs[state][action] + sum(self.transitions[state][action][next_state] * self.values[next_state] for next_state in self.states)



def value_iteration(mdp, epsilon=1e-6):
    # if input threshold is not supplied for epislon
    U1 = {s:0 for s in mdp.states}
    R, T, gamma =  mdp.rewards, mdp.transitions, mdp.gamma
    while True:
        delta = 0
        U = U1.copy()
        for s in mdp.states:
            U1[s] = R[s] + gamma * max([sum(p*U[s1] for (p,s1) in transitions[s][a]) for a in mdp.get_actions()])
            delta = max(delta, abs(U1[s] - U[s]))
        #contraction for covergence :  this threshold is also the threshold supplied so it could be epsilon or threshold
        if delta <= epsilon * (1-gamma)/gamma:
            return U

def value_iteration(mdp, epsilon = 1e-6):
    # if input threshold is not supplied for epislon
    Valfn = {s:0.0 for s in mdp.states}
    #print(mdp.states)
    R, T, actions, gamma =  mdp.rewards, mdp.transitions, mdp.actions, mdp.gamma
    policy = {s: None for s in mdp.states}
    global backups
    backups = 0
    while True:
        delta = 0
        for s in mdp.states:
            if s == mdp.goal_state or s in mdp.terminals:
                continue
            v = Valfn[s]
            max_val = float('inf')
            best_a = None
            for a in actions:
                v_new = 0
                #print(mdp.transitions[s],s,mdp.goal_state, mdp.terminals)
                for s1 in mdp.transitions[s][a]:
                    #print(gamma, Valfn[s1])
                    p = mdp.transitions[s][a][s1]
                    r = mdp.rewards[s]
                    #print(Valfn)
                    v_new += p * (r + gamma * Valfn[s1])
                    backups += 1
                if v_new  > max_val:
                    best_a = a
                    max_val = v_new
            Valfn[s] = max_val
            policy[s] = best_a
            delta = max(delta, abs(v - Valfn[s]))
        if delta < epsilon:
            break
        return v, policy

def rtdp(mdp, init_state, num_iterations=1000):
    global backups
    backups = 0
    for _ in range(num_iterations):
        state = init_state
        path = [state]
        while state != mdp.goal_state:
            action = min(mdp.actions, key=lambda a:mdp.q_value(s,a))
            next_state = mdp.get_next_state(state, action)
            if next_state is None:
                break

            state = next_state
            path.append(state)
        for s in reversed(path):
            mdp.values[state] = min(mdp.get_value(state, action))
            backups += 1

def get_optimal_policy_rtdp(mdp):
    policy = {state:None for state in mdp.states}

    for s in mdp.states:
        if s != self.goal_state:
            policy[s] = min(mdp.actions, key=lambda  a: mdp.get_value(s,a))

    return policy

def extract_mdp(all_lines, gamma, threshold, num_iterations, alg):
    num_states = None
    num_actions = None
    start_state = None
    states = []
    actions = []
    transitions = {}
    costs = {}
    goal_state = 0
    terminals = []
    state_idxs = []
    rewards = {}
    all_lines = [line.strip() for line in all_lines]
    for j in range(0, len(all_lines)):
        if "#" in all_lines[j]:
            continue
        elif "number of states" in all_lines[j]:
            txt =  all_lines[j].split(':')[1].strip()
            num_states = int(txt)

        elif "start state" in all_lines[j]:
            start_state = int(all_lines[j][-1])

        elif len(all_lines[j].split(' ')) == 3:
            state_idxs.append(j)

    transitions = {s:{} for  s in range(num_states)}
    #print(num_states)
    states = [s for s in range(num_states)]
    num_actions = abs(state_idxs[0]+1 - state_idxs[1]+1)
    actions = [a for a in range(num_actions)]

    for i,j in enumerate(state_idxs):
        r, is_terminal, num_a_s = all_lines[j].split(' ')
        r = float(r)
        is_terminal = int(is_terminal)
        num_a_s = int(num_a_s)
        rewards[i] = r
        transitions [i] = {a:{} for a in range(num_a_s)}
        costs = {s:{a:r for a in range(num_a_s)} for s in states}
        #print(is_terminal, i, len(state_idxs))
        if not is_terminal:
            # or i+1 < len(state_idxs):
            #print(i, is_terminal, len(state_idxs))
            for l,k in enumerate(range(j+1, state_idxs[i+1], 1)):
                if "#" not in all_lines[k]:
                    num_a_s = 0
                    line = all_lines[k].split(' ')
                    num_successors  = int(line[0])
                    for m in range(1, (num_successors*2)+1, 2):
                        transitions[i][l][int(line[m])] = float(line[m+1])
        else:
            #print(i, is_terminal)
            goal_state = i
            terminals.append(i)
            transitions.pop(goal_state, None)
    #print(transitions)
    mdp = MDP(states, actions, transitions, costs, rewards, terminals, goal_state, gamma)
    return mdp


def main():
    args = sys.argv
    gamma = None
    alg = None
    threshold = None
    num_iterations = None 
    #print(args)
    for i,arg in enumerate(args):
        if i == 0 or "code.py" in arg:
            continue
        elif i == 1 and ("vi" in arg or "rtdp" in arg):
            alg = arg.strip()
        elif i == 2 and "." in arg:
            gamma = float(arg)
            #print(gamma)
        elif i == 3 :
            if "." in arg:
                #print(arg)
                threshold = float(arg)
            else:
                num_iterations = int(arg)
        else:
            continue

    all_lines = sys.stdin.readlines()
    #print(all_lines)	
    mdp = extract_mdp(all_lines, gamma, threshold, num_iterations, alg)
    if alg == "vi":
        values, results = value_iteration(mdp, threshold)
    elif alg == "rtdp":
        rtdp(mdp, mdp.start_state, num_iterations)
        results = get_optimal_policy(mdp)

    for r in results:
        print(r)
    print(f"{backups} backups performed.")

if __name__ == "__main__":
	main()
