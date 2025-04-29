"""
Assignment 6, CS 830, Spring 2025
Author: Sushma Anand Akoju
"""
#### Note about assignment 6:
# The grammar for CNF is custom and is unlike regular CNFs.

# #### Assignment 6's questions and Syllabus:

# Syllabus says "prohibits the use of **outside code**, requires that all work be the student's
# own". Assignment 6 provides an exception to this policy that allows the use of parser
# generators like ANTLR or Bison, which generate code files considered **outside code**,
# contradicting the syllabus policy's prohibition on external code. 
# It also requires submitting these generated files and the parser grammar file "as is," 
# further conflicting with the policy. Additionally, the provided parsers are unreliable 
# due to line-ending sensitivity on Windows versus Linux, creating technical challenges
# that hinder compliance with the academic integrity policy.
# However ANTLR and Bison do not have any boiler-plate grammars written by software developers,
# that work between windows and linux environments at the same time.

# <a href="https://github.com/antlr/antlr4">ANTLR4 version's original </a> which provides grammar files from <a href="https://github.com/antlr/grammars-v4">
#  The proof that the developer-provide-default-example does not run on default ANTLR4 parser on linux <a href="https://github.com/antlr/grammars-v4/blob/master/fol/fol.g4">fol/fol.g4</a> as they have different line endings not honored on Linux but work on Windows. They fail to recognize certain patterns of words as names/variables/predicates/functions.

# Due to this problem from ANTLR4 parsers, I implemented my own parser for assignment 7
# which is available at: <a href="https://github.com/sushmaakoju/cs830/blob/master/sakoju-a07/code.py"> my own parser (recursive descent) </a>

from antlr4 import *
from cnf.clausal_formParser import clausal_formParser
from cnf.clausal_formLexer import clausal_formLexer

#output
#A(F(B), F(B))
#-A(F(B), F(B))

def parse(lines):
    parsed_input = []
    

    for line in lines:
        lexer = clausal_formLexer(InputStream(line))
        stream = CommonTokenStream(lexer)
        parser = clausal_formParser(stream)
        tree = parser.condition()
        # print(tree.toStringTree(recog=parser))
    return parsed_input

def unify(x,y, substitution=None):
    if substitution is False:
        return False
    
    if x == y:
        return substitution

    # not a list
    if isinstance(x, str):
        return unify_variable(x, y, substitution)
    
    if isinstance(y, str):
        return unify_variable(y, x, substitution)
    
    if not isinstance(x, list) or not isinstance(y, list) or not len(x) == len(y):
        return False
    if not x and not y:
        return substitution
    
    return unify(x[1:], y[1:], unify(x[0], y[0], substitution))      
    
def unify_variable(variable, x, substitution):
    if variable in substitution:
            return unify(substitution[variable], x, substitution)
    if x in substitution:
        return unify(variable, substitution[x], substitution)
    substitution[variable] = x
    return substitution

def main():
    global result
    arr = sys.argv
    # print(arr)

    
    for arg in arr:
        if "code.py" in arg:
            continue
        else:
            continue
    
    lines = sys.stdin.readlines()
    lines = [line for line in lines]
    parsed_input = parse(lines)


    res = {}
    res = unify(parsed_input)
        
    results = []
    
    if res:
        for k,v in res.items():
            results.append(v)
            
    for r in results:
        print(*r)
            

if __name__ == "__main__":
    main()