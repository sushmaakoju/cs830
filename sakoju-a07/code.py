"""
Assignment 7, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""

import re
import numpy as np
import os
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

class ClausalForm:
    """Clausal form expressions
    """
    
    def __init__(self, raw_clauses):
        self.raw_clauses = raw_clauses
        self.tokens = [] # filled by extract
        self.variables = {} # filled by extract
        self.clauses = []
        self.kb_set = None
        self.predicate_set = None
        self.mod_varibales = {}
        self.constants = {} # fill constants
        self.functions = {} # the ones inside parenthesis with variables/constants within another pair of parenthesis
        self.predicates ={} # the first in each clause's each one of the sublists
        self.this_predicate = None
    
    def extract_nested_list(self):
        cls = []
        variables = {}
        clause_lengths = {}
        predicate_set = {}
        kb_set = {} #with predicate keys and filtered set elements (for union of resolvable clauses)
        # the actual set of support : kb_set
        for clause in self.raw_clauses:
            cl = []
            depth = 0
            for literal in clause.strip().split('|'):
                lit, _ = self.get_tokens(literal)
                depth += len(re.findall(r"\(", literal))
                predicate_set[lit[0]] = set(lit[1:])
                kb_set[lit[0]] = set()
                result = []
                stack = [result]
                current = result
                i = 0
                while i < len(lit):            
                    if lit[i] == '(':
                        new_list = []
                        current.append(new_list)
                        stack.append(new_list)
                        current = new_list
                        i += 1
                    elif lit[i] == ')':
                        if stack:
                            current = stack.pop()
                        i += 1
                    elif lit[i] != ',':
                        j = i
                        while j < len(lit) and (self.isconstant(lit[j]) or self.isvariable(lit[j]) or self.isliteral(lit[j])):
                            j+=1
                        if len(lit[i:j]) == 1:
                            if self.isvariable(lit[i]):
                                if lit[i] not in variables:
                                    variables[lit[i]] = 1
                                    r = lit[i]+str(1)
                                    current.append(r)
                                    if lit[0] in kb_set:
                                        kb_set[lit[0]].add(r)
                                    else:
                                        kb_set[lit[0]] = set([r])
                                else:
                                    variables[lit[i]] += 1
                                    r = lit[i]+str(variables[lit[i]])
                                    current.append(r)
                                    if lit[0] in kb_set:
                                        kb_set[lit[0]].add(r)
                                    else:
                                        kb_set[lit[0]] = set([r])
                            else:
                                current.append(lit[i])
                                                                   
                                if lit[0] in kb_set:
                                    kb_set[lit[0]].add(lit[i])
                                else:
                                    kb_set[lit[0]] = set([lit[i]])
                        else:
                            current.append(lit[i:j])
                                                                                               
                            if lit[0] in kb_set:
                                for v in lit[i:j]:
                                    kb_set[lit[0]].add(v)
                            else:
                                kb_set[lit[0]] = set([lit[i:j]])
                        i = j
                    else:
                        i += 1
                #for smaller clauses
                cl.append(result)
            cls.append(cl)
            if (len(cl), depth) not in clause_lengths:
                clause_lengths[(len(cl), depth)] = [cls.index(cl)]
            else:
                clause_lengths[(len(cl), depth)].append(cls.index(cl))
            
        return cls, variables, clause_lengths, predicate_set, kb_set
    
    def process_input(self):
        cls, vars, lengths,predicate_set, kb_set = self.extract_nested_list()
        self.clauses = cls
        self.variables = vars
        self.mod_varibales = self.get_variables()
        self.clauses_lengths = lengths
        self.kb_set = kb_set
        self.predicate_set = predicate_set
        self.constants = self.get_constants()
        self.functions = self.get_functions() # the ones inside parenthesis with variables/constants within another pair of parenthesis
        self.predicates = self.get_predicates() # the first in each clause's each one of the sublists
    
    def isvariable(self,s):
        return isinstance(s, str) and s.islower() and (s.isalnum() or s.isalpha())

    def isconstant(self,s):
        return s and isinstance(s, str) and (s.isupper() or  s[0].isupper()) and s.isalpha()

    def isliteral(self,s):
        if s.startswith('-'):
            return (s.isupper() or (len(s) > 1 and  s[1].isupper())) and \
                (s.isalpha() or (len(s) > 1 and (s[1:].isalpha() or s[1:].isalnum()) ))
        else:
            return (s.isupper() or s[0].isupper()) and (s.isalpha() or s.isalnum())

    def get_predicate(self,s):
        if s.startswith('-'):
            return re.findall(r'^\-[A-Z][a-z]*[0-9]*\(', s)[0][0:-1]
        else:
            return re.findall(r'^[A-Z][a-z]*[0-9]*\(', s)[0][0:-1]
        
    def get_constants(self):
        constants = {}
        for i,clause in enumerate(self.clauses):
            for j, lit in enumerate(clause):
                for k, token in enumerate(lit):
                    if isinstance(token, list):
                        if self.isconstant(token[0]):
                            if token[0] not in constants:
                                constants[token[0]] = []
                            constants[ token[0]].append([i,j,k])
        return constants
    
    def get_variables(self):
        variables = {}
        for i,clause in enumerate(self.clauses):
            for j, lit in enumerate(clause):
                for k, token in enumerate(lit):
                    if isinstance(token, list):
                        if self.isvariable(token[0]):
                            if token[0] not in variables:
                                variables[token[0]] = []
                            variables[ token[0]].append([i,j,k])
        return variables
    
    def get_functions(self):
        functions = {}
        for i,clause in enumerate(self.clauses):
            for j, lit in enumerate(clause):
                for k, token in enumerate(lit):
                    if isinstance(token, str):
                        if self.isvariable(token) and token not in self.predicates:
                            if token not in functions:
                                functions[token[0]] = []
                            functions[ token[0]].append([i,j,k])
        return functions
    
    def get_predicates(self):
        predicates = []
        for clause in self.clauses:
            for lit in clause:
                if lit and lit[0] not in predicates and self.isliteral(lit[0]):
                    predicates.append(lit[0])
        return predicates
    
    def get_tokens(self, s):
        token_pattern = r"([-\*\w]+|[(,*)+\-*/])"
        parenthesis_pattern = r"(|[()+\-*/])"
        all_tokens = re.findall(token_pattern, s)
        parenthesis = [p.strip() for p in re.findall(parenthesis_pattern, s) if p]
        return all_tokens, parenthesis

def negate(query):
    neg_query = ""
    if isinstance(query, str) and len(query)>2:
        if query[0] == '-':
            neg_query= query[1:]
        else:
            neg_query = '-'+query
    return neg_query

# a working version of unify
def unify(exp1, exp2, sub={}):
    if sub is None:
        return None
    
    if exp1 == exp2:
        return sub

    if isinstance(exp1, list) and isinstance(exp2, list):
        if len(exp1) != len(exp2):
            return None
        return unify(exp1[1:], exp2[1:], unify(exp1[0], exp2[0], sub))
    if isinstance(exp1, str):
        if isinstance(exp2, str):
            return unify_var(exp1, exp2, sub)
    if isinstance(exp2, str):
        if isinstance(exp1, str):
            return unify_var(exp2, exp1, sub)
    return None

def unify_var(var, exp, sub):
    if var in sub:
        return unify(sub[var], exp, sub)
    if exp in sub:
        return unify(var, sub[exp], sub)
    if occur_check(var, exp):
        return None
    sub[var] = exp
    return sub

# as required by the grad extensions
def occur_check(var, exp):
    if isinstance(exp, str):
        return var == exp
    if isinstance(exp, list):
        return any(occur_check(var, e) for e in exp)
    return False

def run_unify(sorted_clauses_list):
    # sorted min to max (start with (0,1), then (2,3)...so on)
    l = len(sorted_clauses_list)
    n,m = l-1, l-1
    sub_memory = {}
    sub_list = []
    for i, j in zip(range(0,n,1), range(1,m,1)):
        sub = unify(sorted_clauses_list[i], sorted_clauses_list[j])
        if sub:
            sub_memory[(i,j)] = [sub, [sorted_clauses_list[i], sorted_clauses_list[j]]]
            sub_list.append(sub)
        else:
            sub_list.append([])
    return sub_memory, sub_list

def substitute(exp, sub):
    if isinstance(exp, str):
        return sub.get(exp, exp)
    elif isinstance(exp, list):
        return[substitute(e, sub) for e in exp]
    else:
        return exp

history = {}
new_clauses = []
resolvents = []
clauses_idx = []
def resolve(clause1, clause2):
    global clause_idx
    for lit1 in clause1:
        for lit2 in clause2:
            # check for negation to derive bottom
            if lit1[0] == '-' and len(lit1) > 1 and lit1[1:] == lit2:
                sub = unify(lit1[1:], lit2)
                if sub is not None:
                    new_clause = [lit for lit in clause1 if lit != lit1] + [lit for lit in clause2 if lit != lit2]
                    clauses_idx.append({"idx":None, "clause1":clause1, "clause2":clause2, "new_clause": new_clause})
                    return [substitute(lit, sub) for lit in new_clause]
            # check other way
            elif lit2[0] == '-' and len(lit2) > 1 and lit2[1:] == lit1:
                sub = unify(lit1, lit2[1:])
                if sub is not None:
                    new_clause = [lit for lit in clause1 if lit != lit1] + [lit for lit in clause2 if lit != lit2]
                    clauses_idx.append({"idx":None, "clause1":clause1, "clause2":clause2, "new_clause": new_clause})
                    return [substitute(lit, sub) for lit in new_clause]
    return None

# for history "set" (not included in this submission)
def get_resolving_clauses(kb_set, predicate_set): # kb_set is a dictionary with predicates as keys
    resolving_set = set()
    for predicate in predicate_set:
        if predicate in kb_set:
            resolving_set = resolving_set.union(kb_set[predicate])
    return resolving_set

def get_clauses(predicate, clauses):
    pred_clauses = []
    for cl in clauses:
        for lit in cl:
            if predicate == lit:
                pred_clauses.append(cl)
    return pred_clauses

def search_predicate(clauses, predicates):
    resolving_clauses = set()
    for predicate in predicates:
        clauses_p = get_clauses(predicate, clauses)
        resolving_clauses = resolving_clauses.union(clauses_p)
    return resolving_clauses

def is_subset_literal(lit1, clauses):
    for cl in clauses:
        for lit in cl:
            for token in lit:
                if token in lit:
                    return True
    
def is_subset(new_clauses, clauses):
    subset_flag = False
    counter = 0
    n = len(new_clauses)
    for c1 in new_clauses:
        m = len(c1)
        counter2 = 0
        for lit1 in c1:
            if is_subset_literal(lit1, clauses):
                counter2 += 1
        if counter2 == m:
            counter += 1
    if counter >= 1:
        return True
    return False

def is_same_clause(c1, c2):
    counter = 0
    for lit in c1:
        for lit2 in c2:
            if not all(map(lit.__contains__, lit2)):
                return False
    return True
                

def get_remaining_clauses(new_clauses, clauses):
    diff = []
    for c1 in new_clauses:
        for c2 in clauses:
            if is_same_clause(c1, c2):
                continue
            else:
                diff.append(c2)
    return c2           

def resolution(clauses, query, cnf):
    """Sorted clauses, query, cnf instance

    Args:
        clauses (_type_): _description_
        query (_type_): _description_
        cnf (_type_): _description_
    """
    
    while True:
        global history, new_clauses, resolvents
        if len(clauses) > 3000:
            return False
        for i,c1 in enumerate(clauses):

            resolvin_cls = search_predicate(clauses, cnf.predicates)
            for j,c2 in enumerate(resolvin_cls):
                
                if c1 == c2:
                    continue
            is_in_memory1 = False
            is_in_memory2 = False
            if c2 in history:
                is_in_memory1 = True
                if c1 in history:
                    history[c2].remove(c1)
                    continue
            if c1 in history:
                is_in_memory2 = True
                if c2 in history:
                    history[c1].remove(c2)
                    continue
            if is_in_memory2:
                history[c1].append([c2])
            else:
                history[c1] = [c2]
            resolvents = resolve(c1, c2)
            if resolvents == False:
                clauses_idx.append({"idx":i, "clause1":c1, "clause2":c2})
                return True # derived empty - contradiction found. conclusion holds.
            new_clauses = set(new_clauses).union(resolvents)
            
        #check if any new inferences
        if is_subset(new_clauses, clauses):
            return False 
        new_clauses = get_remaining_clauses(new_clauses, clauses)
        for c1 in new_clauses:
            clauses.append(c1)


def main():
    global result
    arr = sys.argv
    # print(arr)

    for arg in arr:
        if "code.py" in arg:
            continue
        else:
            continue
    
    all_lines = sys.stdin.readlines()
    lines = [line.strip() for line in all_lines][:-2]
    # add negated query to the support - add to clause lists
    lines.append(all_lines[-1])
    lines.append(negate(all_lines[-1])) # add negation
    
    cnf = ClausalForm(lines)
    cnf.process_input()
    query = cnf.clauses[-1]
    # print(cnf.clauses)
    # print(cnf.kb_set)
    # print(cnf.clauses_lengths.items())

    sorted_list = []
    # sorted by depth (max number of opening parenthesis in each clause)
    # creating the set of support sorted by min to max length clauses (by depth)
    for k, indices in dict(sorted(cnf.clauses_lengths.items(), key=lambda key: key[1], reverse=True)).items():
        if len(indices) > 1:
            for id in indices:
                sorted_list.append(cnf.clauses[id])
        else:
            sorted_list.append(cnf.clauses[indices[0]])
    # print(sorted_list)
    
    res = {}
    # checking with dp memoization but resolves true when it should fail or possibly run forever on edge case: recursion clause
    # sub_memory, sub_list = run_unify(sorted_list)
    # print(sub_list)
    if resolution(sorted_list, query, cnf):
        global history
        #process clauses_idx
        #results = reconstruct(string_clauses, clauses_idx)

    results = []
    
    if res:
        for k,v in res.items():
            results.append(v)
            
    for r in results:
        print(*r)
            

if __name__ == "__main__":
    main()