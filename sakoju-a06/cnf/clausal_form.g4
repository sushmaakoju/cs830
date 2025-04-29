/*
* FOL rewritten for Antlr4 by Kamil Kapałka
*
*/

// $antlr-format alignTrailingComments true, columnLimit 150, minEmptyLines 1, maxEmptyLinesToKeep 1, reflowComments false, useTab false
// $antlr-format allowShortRulesOnASingleLine false, allowShortBlocksOnASingleLine true, alignSemicolons hanging, alignColons hanging

grammar clausal_form;

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

condition
    : clause (ENDLINE clause)* ENDLINE* EOF
    ;

clause
    : literal bin_connective clause 
    | literal
    ;

literal
    : predicate 
    | NOT predicate
    ;

predicate
    : capitalizedname(termlist)
    ;

termlist
    : term (separator termlist)*
    | term
    ;

term 
    : capitalizedconstantname
    | capitalizedfunctionname LPAREN termlist RPAREN
    | lowercasevariablename
    ;

bin_connective
    : CONJ
    ;

lowercasevariablename
    : VARIABLE_LOWER
    ;
 
capitalizedconstantname
    : CAPS_PRED_FUNC
    ;

capitalizedfunctionname
    : CAPS_PRED_FUNC
    ;

capitalizedname
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
    : MINUS
    ;

fragment MINUS
    :'-' 
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