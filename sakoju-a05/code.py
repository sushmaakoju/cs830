"""
Assignment 5, CS 830, Spring 2025
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
visited = []
model = []
wflag = False
is_sat = False
tries = 0
flips = 0

#experimental dll without some set_variable conditions
def dll_with_backtrack(clauses, model={}):
    global count, is_sat
    
    if any(not clause for clause in clauses):
        return False, None
    
    # check if all of the model assignments satisfied for all the clauses
    if all(any(lit > 0 and lit in model and model[lit] or lit < 0 and \
        -lit in model and not model[-lit] for lit in clause) for clause in clauses):
        return True, model
    
    if len(clauses) == 0:
        return True, model
    
    #unit propagate
    for clause in clauses:
        if len(clause) == 1:
            literal = clause[0]
            if literal > 0:
                if literal in model and not model[literal]:
                    return False, None
                # set variable acc to lit
                model[literal] = True
                return dll_with_backtrack(clauses, model)
            else:
                if -literal in model and model[-literal]:
                    return False, None
                model[-literal] = False
                return dll_with_backtrack(clauses, model)
            
    # set variable
    for clause in clauses:
        for literal in clause:
            val = abs(literal)
            if val not in model:
                model[literal] = True
                is_sat, model = dll_with_backtrack(clauses, model)
                if is_sat:
                    return is_sat, model
                
                model[val] = False
                is_sat, model = dll_with_backtrack(clauses, model)
                if is_sat:
                    return is_sat, model
                
                # the failed assignment, remove and backtrack
                del model[val]
                return False, None
    return False, None
    
    # #set variable
    # _clauses = [c for c in clauses if v not in c]
    # this_cnf = [c.difference({-v}) for c in _clauses]
    # # model.append(int(v))
    # # print(int(v))
    # is_sat, model = dll(_clauses, {**model, **{v:int(v)}} )
    # if is_sat:
    #     return is_sat, model

    # #set variable v = false
    # _clauses = [c for c in clauses if int(-v) not in c]
    # _clauses = [c.difference({v}) for c in _clauses]
    # # model.append(int(-v))
    # # print(int(-v))
    # is_sat, model = dll(_clauses, {**model, **{v: int(-v)}})
    
    # if is_sat:
    #     return is_sat, model
    
    # return False, model


def set_variable(clauses, literal):
    global count
    
    _clauses = []
    for clause in clauses:
        # remove clauses where literal appears as literal i.e. val = True
        if literal in clause:
            continue
        # print( "-literal",-literal in clause)
        if -literal in clause:
            # remove literal from clause where literal appears as -literal i.e. val = False
            _clause = tuple([l for l in clause if l != -literal])
            # print("clause",_clause)
            # count += 1
            _clauses.append(_clause)
        else:
            _clauses.append(clause)
            # count += 1
    return _clauses

def get_pure_literals(clauses):
    literals = set()
    for clause in clauses:
        for literal in clause:
            literals.add(literal)
    return literals

def vanilla_dll(clauses, model={}):
    """Davis-Logemann-Loveland Algorithm
    UnitPropagate and SetVariable

    Args:
        clauses (_type_): _description_
        model (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    global count, is_sat
    # print(clauses)

    if not clauses:
        return True, model
    
    if any(not clause for clause in clauses):
        return False, model

    # check if all of the model assignments satisfied for all the clauses - is NOT recommended to unquote
    # if all(any( [lit > 0 and lit in model and model[lit], \
    #     lit < 0 and -lit in model and not model[-lit]] for lit in clause) for clause in clauses):
    #     return True, model
    
    # unit propagation
    for clause in clauses:
        if len(clause) == 1:
            l = clause[0]
            value = abs(l)
            if value not in model:
                model[value] = l > 0
                count += 1
                return vanilla_dll(set_variable(clauses, l), model)
    
    # Pure literal elimination, for current_clauses
    # for literal in get_pure_literals(clauses):
    #     value = abs(literal)
    #     if -literal not in get_pure_literals(clauses):
    #         model[value] = literal > 0
    #         return vanilla_dll(set_variable(clauses, literal), model)
    
    if clauses:
        value = abs(clauses[0][0])
        # dll(SetVariable(Phi with v =true)) = SAT
        model[value] = True
        count += 1
        is_sat, model = vanilla_dll(set_variable(clauses, value), model.copy())
        if is_sat:
            return is_sat, model

        # return DLL(SetVariable(Phi with v =false))
        model[value] = False
        count += 1
        return vanilla_dll(set_variable(clauses, -value), model.copy())

    return True, model

def is_clause_sat(clause, model):
    
    for lit in clause:
        if lit > 0:
            if model[lit]:
                return True
        else:
            if not model[-lit]:
                return True
    return False

def count_unsat(clauses, model):
    # count number unsatisfied clauses for this model
    count = 0
    for clause in clauses:
        if not is_clause_sat(clause, model):
            count += 1
    return count

def walksat(clauses, variables,  p=0.5, maxflips=10000, maxtries=100):
    global count, is_sat, tries, flips
    # model = [1]
    
    model = {}
    for _ in range(maxtries):
        tries += 1
        #set random assignment
        model = {var: random.choice([True, False]) for var in variables }
        for _ in range(maxflips):
            #count this flip
            flips += 1
            unsat = [ clause for clause in clauses if not is_clause_sat(clause, model)]
            if not unsat:
                return True, model, tries, flips
            
            flip_this_clause = random.choice(unsat)
            
            if random.random() < p:
                flip_this_lit = random.choice(flip_this_clause)
                flip_this_var = abs(flip_this_lit)
                # print(flip_this_var, model)
                model[flip_this_var] = not model[flip_this_var]
            else:
                # greedy - get best var to flip
                flip_best_var = None
                min_break = float('inf')
                
                # search flip_this_clause
                for lit_to_flip in flip_this_clause:
                    flip_this_var = abs(lit_to_flip)
                    temp = model.copy()
                    temp[flip_this_var] = not temp[flip_this_var]
                    num_break = count_unsat(clauses, temp)
                    
                    if num_break < min_break:
                        flip_best_var = flip_this_var
                model[flip_this_var] = not model[flip_this_var]
    return False, model, tries, flips

def main():
    global count, result, wflag, is_sat, tries, flips
    arr = sys.argv
    # print(arr)
    wflag = False
    
    for arg in arr:
        restart = True if arg.startswith('-') and "restart" in arg else False
        if "code.py" in arg:
            continue
        elif "-w" in arr:
            wflag = True
        else:
            continue
    
    var_count = 0
    clause_count = 0
    lines = sys.stdin.readlines()
    lines = [line for line in lines if not line.startswith("c")]
    if "p" in lines[0] and "cnf" in lines[0]:
        var_count, clause_count = lines[0].strip().split()[-2:]
        var_count = int(var_count)
        clause_count = int(clause_count)
        
    cnf = []
    literals = []
    i = 0
    for line in lines[1:]:
        clauses = [cl for cl in [cl for cl in line.strip().split(" 0") if cl]]
        # print(line,clauses)
        for cl in clauses:
            # print(cl)
            clause = tuple([int(l.strip()) for l in cl.split(" ") if l.strip()])
            cnf.append(clause)
            i += 1
            literals.extend([abs(val) for val in clause])
    literals = list(set(literals))

    # print(var_count, clause_count)
    # print(len(cnf)==clause_count,cnf,literals,len(set(literals)) == var_count)
    assert len(cnf)==clause_count, "Clauses generated are not equal to the number of clauses expected"
    assert len(set(literals)) == var_count, "Number of literals do not match the required number."
        
    # cnf = CNF(clauses, literals, {})
    res = {}
    if wflag:
        count = 0
        result = []
        is_sat, res, tries, flips = walksat(cnf, literals)
    else:
        count = 0
        result = []
        # print(cnf)
        # is_sat, res = dll_with_backtrack(clauses,{}) #restricted dpll but is slower
        is_sat, res = vanilla_dll(cnf)
        # print(is_sat, res)
        
    results = [["s cnf %d %d %d"%(is_sat, var_count, clause_count )]]
    
    if is_sat:
        if res:
            # print(res)
            for k,val in res.items():
                if not val:
                    l = int(-k)
                    results.append(["v %d"%(l)])
                else:
                    results.append(["v %d"%(k)])
            if wflag:
                results.append(["%d tries."%(tries)])
                results.append(["%d total flips."%(flips)])
            else:
                results.append(["%d branching nodes explored."%(count)])
    else:
        if wflag:
            results.append(["%d tries."%(tries)])
            results.append(["%d total flips."%(flips)])
        else:
            results.append(["%d branching nodes explored."%(count)])
    for r in results:
        print(*r)
            

if __name__ == "__main__":
    main()