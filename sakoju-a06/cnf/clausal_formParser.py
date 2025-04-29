# Generated from ./cnf/clausal_form.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,9,89,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,
        2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,1,0,1,0,1,0,5,0,28,8,
        0,10,0,12,0,31,9,0,1,0,5,0,34,8,0,10,0,12,0,37,9,0,1,0,1,0,1,1,1,
        1,1,1,1,1,1,1,3,1,46,8,1,1,2,1,2,1,2,3,2,51,8,2,1,3,1,3,1,3,1,4,
        1,4,1,4,1,4,5,4,60,8,4,10,4,12,4,63,9,4,1,4,3,4,66,8,4,1,5,1,5,1,
        5,1,5,1,5,1,5,1,5,3,5,75,8,5,1,6,1,6,1,7,1,7,1,8,1,8,1,9,1,9,1,10,
        1,10,1,11,1,11,1,11,0,0,12,0,2,4,6,8,10,12,14,16,18,20,22,0,0,84,
        0,24,1,0,0,0,2,45,1,0,0,0,4,50,1,0,0,0,6,52,1,0,0,0,8,65,1,0,0,0,
        10,74,1,0,0,0,12,76,1,0,0,0,14,78,1,0,0,0,16,80,1,0,0,0,18,82,1,
        0,0,0,20,84,1,0,0,0,22,86,1,0,0,0,24,29,3,2,1,0,25,26,5,8,0,0,26,
        28,3,2,1,0,27,25,1,0,0,0,28,31,1,0,0,0,29,27,1,0,0,0,29,30,1,0,0,
        0,30,35,1,0,0,0,31,29,1,0,0,0,32,34,5,8,0,0,33,32,1,0,0,0,34,37,
        1,0,0,0,35,33,1,0,0,0,35,36,1,0,0,0,36,38,1,0,0,0,37,35,1,0,0,0,
        38,39,5,0,0,1,39,1,1,0,0,0,40,41,3,4,2,0,41,42,3,12,6,0,42,43,3,
        2,1,0,43,46,1,0,0,0,44,46,3,4,2,0,45,40,1,0,0,0,45,44,1,0,0,0,46,
        3,1,0,0,0,47,51,3,6,3,0,48,49,5,4,0,0,49,51,3,6,3,0,50,47,1,0,0,
        0,50,48,1,0,0,0,51,5,1,0,0,0,52,53,3,20,10,0,53,54,3,8,4,0,54,7,
        1,0,0,0,55,61,3,10,5,0,56,57,3,22,11,0,57,58,3,8,4,0,58,60,1,0,0,
        0,59,56,1,0,0,0,60,63,1,0,0,0,61,59,1,0,0,0,61,62,1,0,0,0,62,66,
        1,0,0,0,63,61,1,0,0,0,64,66,3,10,5,0,65,55,1,0,0,0,65,64,1,0,0,0,
        66,9,1,0,0,0,67,75,3,16,8,0,68,69,3,18,9,0,69,70,5,2,0,0,70,71,3,
        8,4,0,71,72,5,3,0,0,72,75,1,0,0,0,73,75,3,14,7,0,74,67,1,0,0,0,74,
        68,1,0,0,0,74,73,1,0,0,0,75,11,1,0,0,0,76,77,5,7,0,0,77,13,1,0,0,
        0,78,79,5,6,0,0,79,15,1,0,0,0,80,81,5,5,0,0,81,17,1,0,0,0,82,83,
        5,5,0,0,83,19,1,0,0,0,84,85,5,5,0,0,85,21,1,0,0,0,86,87,5,1,0,0,
        87,23,1,0,0,0,7,29,35,45,50,61,65,74
    ]

class clausal_formParser ( Parser ):

    grammarFileName = "clausal_form.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "','", "'('", "')'", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'|'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "LPAREN", "RPAREN", "NOT", 
                      "CAPS_PRED_FUNC", "VARIABLE_LOWER", "CONJ", "ENDLINE", 
                      "WHITESPACE" ]

    RULE_condition = 0
    RULE_clause = 1
    RULE_literal = 2
    RULE_predicate = 3
    RULE_termlist = 4
    RULE_term = 5
    RULE_bin_connective = 6
    RULE_lowercasevariablename = 7
    RULE_capitalizedconstantname = 8
    RULE_capitalizedfunctionname = 9
    RULE_capitalizedname = 10
    RULE_separator = 11

    ruleNames =  [ "condition", "clause", "literal", "predicate", "termlist", 
                   "term", "bin_connective", "lowercasevariablename", "capitalizedconstantname", 
                   "capitalizedfunctionname", "capitalizedname", "separator" ]

    EOF = Token.EOF
    T__0=1
    LPAREN=2
    RPAREN=3
    NOT=4
    CAPS_PRED_FUNC=5
    VARIABLE_LOWER=6
    CONJ=7
    ENDLINE=8
    WHITESPACE=9

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ConditionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def clause(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(clausal_formParser.ClauseContext)
            else:
                return self.getTypedRuleContext(clausal_formParser.ClauseContext,i)


        def EOF(self):
            return self.getToken(clausal_formParser.EOF, 0)

        def ENDLINE(self, i:int=None):
            if i is None:
                return self.getTokens(clausal_formParser.ENDLINE)
            else:
                return self.getToken(clausal_formParser.ENDLINE, i)

        def getRuleIndex(self):
            return clausal_formParser.RULE_condition

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCondition" ):
                listener.enterCondition(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCondition" ):
                listener.exitCondition(self)




    def condition(self):

        localctx = clausal_formParser.ConditionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_condition)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            self.clause()
            self.state = 29
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 25
                    self.match(clausal_formParser.ENDLINE)
                    self.state = 26
                    self.clause() 
                self.state = 31
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

            self.state = 35
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==8:
                self.state = 32
                self.match(clausal_formParser.ENDLINE)
                self.state = 37
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 38
            self.match(clausal_formParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def literal(self):
            return self.getTypedRuleContext(clausal_formParser.LiteralContext,0)


        def bin_connective(self):
            return self.getTypedRuleContext(clausal_formParser.Bin_connectiveContext,0)


        def clause(self):
            return self.getTypedRuleContext(clausal_formParser.ClauseContext,0)


        def getRuleIndex(self):
            return clausal_formParser.RULE_clause

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterClause" ):
                listener.enterClause(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitClause" ):
                listener.exitClause(self)




    def clause(self):

        localctx = clausal_formParser.ClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_clause)
        try:
            self.state = 45
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 40
                self.literal()
                self.state = 41
                self.bin_connective()
                self.state = 42
                self.clause()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 44
                self.literal()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LiteralContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def predicate(self):
            return self.getTypedRuleContext(clausal_formParser.PredicateContext,0)


        def NOT(self):
            return self.getToken(clausal_formParser.NOT, 0)

        def getRuleIndex(self):
            return clausal_formParser.RULE_literal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLiteral" ):
                listener.enterLiteral(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLiteral" ):
                listener.exitLiteral(self)




    def literal(self):

        localctx = clausal_formParser.LiteralContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_literal)
        try:
            self.state = 50
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [5]:
                self.enterOuterAlt(localctx, 1)
                self.state = 47
                self.predicate()
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 2)
                self.state = 48
                self.match(clausal_formParser.NOT)
                self.state = 49
                self.predicate()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PredicateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def capitalizedname(self):
            return self.getTypedRuleContext(clausal_formParser.CapitalizednameContext,0)


        def termlist(self):
            return self.getTypedRuleContext(clausal_formParser.TermlistContext,0)


        def getRuleIndex(self):
            return clausal_formParser.RULE_predicate

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPredicate" ):
                listener.enterPredicate(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPredicate" ):
                listener.exitPredicate(self)




    def predicate(self):

        localctx = clausal_formParser.PredicateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_predicate)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 52
            self.capitalizedname()

            self.state = 53
            self.termlist()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TermlistContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def term(self):
            return self.getTypedRuleContext(clausal_formParser.TermContext,0)


        def separator(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(clausal_formParser.SeparatorContext)
            else:
                return self.getTypedRuleContext(clausal_formParser.SeparatorContext,i)


        def termlist(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(clausal_formParser.TermlistContext)
            else:
                return self.getTypedRuleContext(clausal_formParser.TermlistContext,i)


        def getRuleIndex(self):
            return clausal_formParser.RULE_termlist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTermlist" ):
                listener.enterTermlist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTermlist" ):
                listener.exitTermlist(self)




    def termlist(self):

        localctx = clausal_formParser.TermlistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_termlist)
        try:
            self.state = 65
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 55
                self.term()
                self.state = 61
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,4,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 56
                        self.separator()
                        self.state = 57
                        self.termlist() 
                    self.state = 63
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,4,self._ctx)

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 64
                self.term()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TermContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def capitalizedconstantname(self):
            return self.getTypedRuleContext(clausal_formParser.CapitalizedconstantnameContext,0)


        def capitalizedfunctionname(self):
            return self.getTypedRuleContext(clausal_formParser.CapitalizedfunctionnameContext,0)


        def LPAREN(self):
            return self.getToken(clausal_formParser.LPAREN, 0)

        def termlist(self):
            return self.getTypedRuleContext(clausal_formParser.TermlistContext,0)


        def RPAREN(self):
            return self.getToken(clausal_formParser.RPAREN, 0)

        def lowercasevariablename(self):
            return self.getTypedRuleContext(clausal_formParser.LowercasevariablenameContext,0)


        def getRuleIndex(self):
            return clausal_formParser.RULE_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTerm" ):
                listener.enterTerm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTerm" ):
                listener.exitTerm(self)




    def term(self):

        localctx = clausal_formParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_term)
        try:
            self.state = 74
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 67
                self.capitalizedconstantname()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 68
                self.capitalizedfunctionname()
                self.state = 69
                self.match(clausal_formParser.LPAREN)
                self.state = 70
                self.termlist()
                self.state = 71
                self.match(clausal_formParser.RPAREN)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 73
                self.lowercasevariablename()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bin_connectiveContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CONJ(self):
            return self.getToken(clausal_formParser.CONJ, 0)

        def getRuleIndex(self):
            return clausal_formParser.RULE_bin_connective

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBin_connective" ):
                listener.enterBin_connective(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBin_connective" ):
                listener.exitBin_connective(self)




    def bin_connective(self):

        localctx = clausal_formParser.Bin_connectiveContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_bin_connective)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 76
            self.match(clausal_formParser.CONJ)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LowercasevariablenameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def VARIABLE_LOWER(self):
            return self.getToken(clausal_formParser.VARIABLE_LOWER, 0)

        def getRuleIndex(self):
            return clausal_formParser.RULE_lowercasevariablename

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLowercasevariablename" ):
                listener.enterLowercasevariablename(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLowercasevariablename" ):
                listener.exitLowercasevariablename(self)




    def lowercasevariablename(self):

        localctx = clausal_formParser.LowercasevariablenameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_lowercasevariablename)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 78
            self.match(clausal_formParser.VARIABLE_LOWER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CapitalizedconstantnameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CAPS_PRED_FUNC(self):
            return self.getToken(clausal_formParser.CAPS_PRED_FUNC, 0)

        def getRuleIndex(self):
            return clausal_formParser.RULE_capitalizedconstantname

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCapitalizedconstantname" ):
                listener.enterCapitalizedconstantname(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCapitalizedconstantname" ):
                listener.exitCapitalizedconstantname(self)




    def capitalizedconstantname(self):

        localctx = clausal_formParser.CapitalizedconstantnameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_capitalizedconstantname)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 80
            self.match(clausal_formParser.CAPS_PRED_FUNC)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CapitalizedfunctionnameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CAPS_PRED_FUNC(self):
            return self.getToken(clausal_formParser.CAPS_PRED_FUNC, 0)

        def getRuleIndex(self):
            return clausal_formParser.RULE_capitalizedfunctionname

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCapitalizedfunctionname" ):
                listener.enterCapitalizedfunctionname(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCapitalizedfunctionname" ):
                listener.exitCapitalizedfunctionname(self)




    def capitalizedfunctionname(self):

        localctx = clausal_formParser.CapitalizedfunctionnameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_capitalizedfunctionname)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 82
            self.match(clausal_formParser.CAPS_PRED_FUNC)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CapitalizednameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CAPS_PRED_FUNC(self):
            return self.getToken(clausal_formParser.CAPS_PRED_FUNC, 0)

        def getRuleIndex(self):
            return clausal_formParser.RULE_capitalizedname

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCapitalizedname" ):
                listener.enterCapitalizedname(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCapitalizedname" ):
                listener.exitCapitalizedname(self)




    def capitalizedname(self):

        localctx = clausal_formParser.CapitalizednameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_capitalizedname)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self.match(clausal_formParser.CAPS_PRED_FUNC)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SeparatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return clausal_formParser.RULE_separator

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSeparator" ):
                listener.enterSeparator(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSeparator" ):
                listener.exitSeparator(self)




    def separator(self):

        localctx = clausal_formParser.SeparatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_separator)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 86
            self.match(clausal_formParser.T__0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





