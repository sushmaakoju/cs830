# Generated from ./cnf/clausal_form.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .clausal_formParser import clausal_formParser
else:
    from clausal_formParser import clausal_formParser

# This class defines a complete listener for a parse tree produced by clausal_formParser.
class clausal_formListener(ParseTreeListener):

    # Enter a parse tree produced by clausal_formParser#condition.
    def enterCondition(self, ctx:clausal_formParser.ConditionContext):
        pass

    # Exit a parse tree produced by clausal_formParser#condition.
    def exitCondition(self, ctx:clausal_formParser.ConditionContext):
        pass


    # Enter a parse tree produced by clausal_formParser#clause.
    def enterClause(self, ctx:clausal_formParser.ClauseContext):
        pass

    # Exit a parse tree produced by clausal_formParser#clause.
    def exitClause(self, ctx:clausal_formParser.ClauseContext):
        pass


    # Enter a parse tree produced by clausal_formParser#literal.
    def enterLiteral(self, ctx:clausal_formParser.LiteralContext):
        pass

    # Exit a parse tree produced by clausal_formParser#literal.
    def exitLiteral(self, ctx:clausal_formParser.LiteralContext):
        pass


    # Enter a parse tree produced by clausal_formParser#predicate.
    def enterPredicate(self, ctx:clausal_formParser.PredicateContext):
        pass

    # Exit a parse tree produced by clausal_formParser#predicate.
    def exitPredicate(self, ctx:clausal_formParser.PredicateContext):
        pass


    # Enter a parse tree produced by clausal_formParser#termlist.
    def enterTermlist(self, ctx:clausal_formParser.TermlistContext):
        pass

    # Exit a parse tree produced by clausal_formParser#termlist.
    def exitTermlist(self, ctx:clausal_formParser.TermlistContext):
        pass


    # Enter a parse tree produced by clausal_formParser#term.
    def enterTerm(self, ctx:clausal_formParser.TermContext):
        pass

    # Exit a parse tree produced by clausal_formParser#term.
    def exitTerm(self, ctx:clausal_formParser.TermContext):
        pass


    # Enter a parse tree produced by clausal_formParser#bin_connective.
    def enterBin_connective(self, ctx:clausal_formParser.Bin_connectiveContext):
        pass

    # Exit a parse tree produced by clausal_formParser#bin_connective.
    def exitBin_connective(self, ctx:clausal_formParser.Bin_connectiveContext):
        pass


    # Enter a parse tree produced by clausal_formParser#lowercasevariablename.
    def enterLowercasevariablename(self, ctx:clausal_formParser.LowercasevariablenameContext):
        pass

    # Exit a parse tree produced by clausal_formParser#lowercasevariablename.
    def exitLowercasevariablename(self, ctx:clausal_formParser.LowercasevariablenameContext):
        pass


    # Enter a parse tree produced by clausal_formParser#capitalizedconstantname.
    def enterCapitalizedconstantname(self, ctx:clausal_formParser.CapitalizedconstantnameContext):
        pass

    # Exit a parse tree produced by clausal_formParser#capitalizedconstantname.
    def exitCapitalizedconstantname(self, ctx:clausal_formParser.CapitalizedconstantnameContext):
        pass


    # Enter a parse tree produced by clausal_formParser#capitalizedfunctionname.
    def enterCapitalizedfunctionname(self, ctx:clausal_formParser.CapitalizedfunctionnameContext):
        pass

    # Exit a parse tree produced by clausal_formParser#capitalizedfunctionname.
    def exitCapitalizedfunctionname(self, ctx:clausal_formParser.CapitalizedfunctionnameContext):
        pass


    # Enter a parse tree produced by clausal_formParser#capitalizedname.
    def enterCapitalizedname(self, ctx:clausal_formParser.CapitalizednameContext):
        pass

    # Exit a parse tree produced by clausal_formParser#capitalizedname.
    def exitCapitalizedname(self, ctx:clausal_formParser.CapitalizednameContext):
        pass


    # Enter a parse tree produced by clausal_formParser#separator.
    def enterSeparator(self, ctx:clausal_formParser.SeparatorContext):
        pass

    # Exit a parse tree produced by clausal_formParser#separator.
    def exitSeparator(self, ctx:clausal_formParser.SeparatorContext):
        pass



del clausal_formParser