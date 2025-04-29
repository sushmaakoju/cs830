##### Note about assignment 6:
The grammar for CNF is custom and is unlike regular CNFs.

##### Assignment 6's questions and Syllabus:

Syllabus says "prohibits the use of **outside code**, requires that all work be the student's
own". Assignment 6 provides an exception to this policy that allows the use of parser 
generators like ANTLR or Bison, which generate code files considered **outside code**, 
contradicting the syllabus policy's prohibition on external code. It also requires 
submitting these generated files and the parser grammar file "as is," further conflicting 
with the policy. Additionally, the provided parsers are unreliable due to line-ending 
sensitivity on Windows versus Linux, creating technical challenges that hinder compliance 
with the academic integrity policy. However ANTLR and Bison do not have any boiler-plate 
grammars written by software developers, that work between windows and linux environments 
at the same time.

<a href="https://github.com/antlr/antlr4">ANTLR4 version's original </a> 
which provides grammar files from <a href="https://github.com/antlr/grammars-v4"> 
and the developer's provided example does not run on default ANTLR4 parser on linux 
<a href="https://github.com/antlr/grammars-v4/blob/master/fol/fol.g4">fol/fol.g4</a> 
as they have different line endings not honored on Linux. 
They fail to recognize certain patterns of words in predicate names.

Due to this problem from ANTLR4 parsers, 
I implemented my own parser for assignment 7 which is available at: 
<a href="https://github.com/sushmaakoju/cs830/blob/master/sakoju-a07/code.py"> my own parser (recursive descent) </a>

/*
* FOL rewritten for Antlr4 by Kamil Kapałka
*
*/

// $antlr-format alignTrailingComments true, columnLimit 150, minEmptyLines 1, maxEmptyLinesToKeep 1, reflowComments false, useTab false
// $antlr-format allowShortRulesOnASingleLine false, allowShortBlocksOnASingleLine true, alignSemicolons hanging, alignColons hanging

grammar cnf;

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

condition
    : formula (ENDLINE formula)* ENDLINE* EOF
    ;

formula
    : formula bin_connective formula
    | NOT formula bin_connective formula
    | NOT formula
    | pred_constant LPAREN term (separator term)* RPAREN
    ;

term
    : ind_constant
    | variable
    | func_constant LPAREN term (separator term)* RPAREN
    ;

bin_connective
    : CONJ
    ;

variable
    : VARIABLE_LOWER
    ;

//predicate constant - np. _isProfesor(?x)   
pred_constant
    : CAPS_PRED_FUNC
    ;

//individual constant - used in single predicates
ind_constant
    : CONSTANT_CHAR
    ;

//used to create functions, np. .presidentOf(?America) = #Trump
func_constant
    : CAPS_PRED_FUNC
    ;

LPAREN
    : '('
    ;

RPAREN
    : ')'
    ;

separator
    : ','
    ;

NOT
    : '-'
    ;

//CHARACTER
  //  : ('0' .. '9' | 'a' .. 'z' | 'A' .. 'Z')
   // ;

CONSTANT_CHAR
    : [A-Z][a-z]*
    ;

CAPS_PRED_FUNC
    : [A-Z][a-z]*
    ;

VARIABLE_LOWER
    : [a-z][0-9]*
    ;

CONJ
    : '|'
    ;


ENDLINE
    : ('\r' | '\n')+
    ;

WHITESPACE
    : (' ' | '\t')+ -> skip
    ;