container = {}
my_stack = []
container_function = {}

class myNode:
    def __init__(self):
        print("")
    def evaluate(self):
        return 0



class String_nod(myNode):
    def __init__(self, v):
        self.value = str(v)[1:len(str(v))-1]

    def evaluate(self):
        return self.value



class Var_node(myNode):
    def __init__(self, v):
        self.v = v

    def evaluate(self):
        x=container[self.v]
        return x



class Assingment(myNode):
    def __init__(self, v1, v2):
            self.v1 = v1
            self.v2 = v2

    def evaluate(self):
        if (isinstance(self.v1, Var_node)):
            container[self.v1.v] = self.v2.evaluate()
        elif (isinstance(self.v1, List_At_I)):
            (self.v1.v1.evaluate())[self.v1.v2.evaluate()] = self.v2.evaluate()
            container[self.v1.v1] = self.v1.v1.evaluate()
        else:
            print("Semantic Error")
            exit()




class Else_statement(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        if (isinstance(self.v1, If_statement) and isinstance(self.v2, Blocks)):
            if (self.v1.v1.evaluate()):
                self.v1.evaluate()
            else:
                self.v2.evaluate()
        else:
            print("Semantic Error")
            exit()

class If_statement(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        if(isinstance(self.v2, Blocks)):
            if (isinstance(self.v1.evaluate(), bool)):
                if (self.v1.evaluate()):
                    self.v2.evaluate()
            else:
                print("Semantic Error")
                exit()
        else:
            print("Semantic Error")
            exit()



class PrintmyNode(myNode):
    def __init__(self, v):
        self.v = v

    def evaluate(self):
        print(self.v.evaluate())


class Blocks(myNode):
    def __init__(self,sl):
        self.list = sl

    def evaluate(self):
        if (self.list == None):
            pass
        else:
            for s in self.list:
                evaluated_node = s
                evaluated_node.evaluate()


class While_loop(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        try:
            while(self.v1.evaluate()):
                x =self.v2
                x.evaluate()
        except:
            print("Semantic Error")
            exit()

class BopmyNode(myNode):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self):
        try:
            if (self.op == '+'):
                return self.v1.evaluate() + self.v2.evaluate()
            elif (self.op == '-'):
                return self.v1.evaluate() - self.v2.evaluate()
            elif (self.op == '*'):
                return self.v1.evaluate() * self.v2.evaluate()
            elif (self.op == '/'):
                return self.v1.evaluate() / self.v2.evaluate()
            elif (self.op == 'mod'):
                return self.v1.evaluate() % self.v2.evaluate()
            elif (self.op == '**'):
                return self.v1.evaluate() ** self.v2.evaluate()
            elif (self.op == 'div'):
                return self.v1.evaluate() // self.v2.evaluate()
        except:
            print("Semantic Error")
            exit()


class Num_nod(myNode):
    def __init__(self, v):

        if( 'E' in v ):
            self.value = float(v)
        elif( '.' in v ):
            self.value = float(v)
        elif( 'e' in v):
            self.value = float(v)
        else:
            self.value = int(v)

    def evaluate(self):
        return self.value


class Operation(myNode):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self):
        try:
            if (self.op == '>'):
                return self.v1.evaluate() > self.v2.evaluate()
            elif (self.op == '>='):
                return self.v1.evaluate() >= self.v2.evaluate()
            elif (self.op == '<'):
                return self.v1.evaluate() < self.v2.evaluate()
            elif (self.op == '<='):
                return self.v1.evaluate() <= self.v2.evaluate()
            elif (self.op == '=='):
                return self.v1.evaluate() == self.v2.evaluate()
            elif (self.op == '<>'):
                return self.v1.evaluate() != self.v2.evaluate()
        except:
            print("Semantic Error")
            exit()



class BooleanOp(myNode):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self):
        if (isinstance(self.v1.evaluate(), bool) and isinstance(self.v2.evaluate(), bool)):
            if (self.op == 'andalso'):
                return self.v1.evaluate() and self.v2.evaluate()
            elif (self.op == 'orelse'):
                return self.v1.evaluate() or self.v2.evaluate()
        else:
            print("Semantic Error")
            exit()

class In_statemnt(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        try:
            return self.v1.evaluate() in self.v2.evaluate()
        except:
            print("Semantic Error")
            exit()

class Not_Statement(myNode):
    def __init__(self, v1):
        self.v1 = v1

    def evaluate(self):
        try:
            return not self.v1.evaluate()
        except:
            print("Semantic Error")
            exit()


class ListmyNode(myNode):
    def __init__(self, v1):
        self.v1 = v1

    def evaluate(self):
        if (self.v1 == None):
            return []
        else:
            return [self.v1.evaluate()]



class ListmyNode2(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        return self.v1.evaluate() + [self.v2.evaluate()]



class TuplemyNode(myNode):
    def __init__(self, v1):
        self.v1 = v1

    def evaluate(self):
        if (self.v1 == None):
            return ()
        else:
            return tuple(self.v1.evaluate())



class neg_nod(myNode):
    def __init__(self, v1):
        self.v1 = v1

    def evaluate(self):
        if (isinstance(self.v1.evaluate(), float) or isinstance(self.v1.evaluate(), int)):
            return -1 * self.v1.evaluate()
        else:
            print("Semantic Error")
            exit()


class Concatenation(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        try:
            x =[self.v1.evaluate()] + self.v2.evaluate()
            return x
        except:
            print("Semantic Error")
            exit()


class List_At_I(myNode):
    def __init__(self, v1, v2):
            self.v1 = v1
            self.v2 = v2

    def evaluate(self):

        if ((isinstance(self.v1.evaluate(), list) or isinstance(self.v1.evaluate(), str))):
            if(isinstance(self.v2.evaluate(), int)):
                return self.v1.evaluate()[self.v2.evaluate()]
            else:
                print("Semantic Error")
                exit()
        else:
            print("Semantic Error")
            exit()


class Tuple_At_I(myNode):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2

    def evaluate(self):
        if (isinstance(self.v2.evaluate(), tuple) and isinstance(self.v1.evaluate(), int) and self.v1.evaluate() >= 1):
            x =self.v2.evaluate()[self.v1.evaluate() - 1]
            return x
        else:
            print("Semantic Error")
            exit()



class Bool_nod(myNode):
    def __init__(self, v):
        if (v == 'False'):
            self.v = False
        elif (v == 'True'):
            self.v = True
        else:
            print("Semantic Error")
            exit()

    def evaluate(self):
        return self.v

class ProgramNode(myNode):
    def __init__(self, f, b):
        self.functions = f
        self.block = b

    def evaluate(self):
        my_stack.append(container_function)
        for i in self.functions:
            lll=isinstance(i, FuncNode)
            if (lll):
                i.evaluate()
            else:
                print("Semantic Error")

        if (isinstance(self.block, Blocks)):
            self.block.evaluate()
        else:
            print("Semantic Error")



class FuncNode(myNode):
    def __init__(self, name, params, block, output):
        self.name = name
        self.params = params
        self.block = block
        self.output = output

    def evaluate(self):
        global my_stack
        xxxxx=isinstance(self.name, Var_node)
        if (xxxxx):
            my_stack[0][self.name.v] = self
        else:
            print("Semantic Error")

    def go(self, args):
        global my_stack
        global container
        pp =len(args)
        pp2=len(self.params)
        if (pp != pp2):
            print("Semantic Error")

        for i in range(0,len(args)):
            ggg=isinstance(self.params[i], Var_node)
            if (ggg):
                my_stack[-2+1][self.params[i].v] = args[i].evaluate()
            else:
                print("Semantic Error")

        saved_dict = container
        container = my_stack[-1]
        ppp=isinstance(self.block, Blocks)
        if (ppp):
            self.block.evaluate()
            anser = self.output.evaluate()
            container = saved_dict
        else:
            ans = self.output.evaluate()
            print("Semantic Error")
            container = saved_dict

        return anser


class FuncCaNode(myNode):
    def __init__(self, n, a):
        self.n = n
        self.a = a

    def evaluate(self):
        global my_stack
        n_d = {}
        my_stack.append(n_d)
        if (isinstance(self.n, Var_node)):
            functionNode = my_stack[0][self.n.v]
        else:
            print("Semantic Error")
        ans = functionNode.go(self.a)
        my_stack.pop()
        return ans



reserved = {
    'else'  : 'ELSE',
    'in'    : 'IN',
    'if'    : 'IF',
    'while' : 'WHILE',
    'print' : 'PRINT',
    'mod'   : 'MOD',
    'div'   : 'DIV',
    'not': 'NOT',
    'andalso' : 'ANDALSO',
    'orelse'  : 'ORELSE',
    'fun'   :   'FUN'
}

tokens = (
    'NUMBER','STRING','BOOLEAN',
    'PLUS','MINUS','TIMES','DIVIDE',
    'COMMA',
    'COMMENT',
    'IN','DIV','MOD', 'POWER','CONCAT', 'NOT',
    'ANDALSO', 'ORELSE',
    'EQUAL', 'NOTEQUAL','GREATER', 'GREATEREQUAL','LESS', 'LESSEQUAL',
    'SEMICOLON',
    'VARIABLE', 'ASSIGN',
    'LCURLY', 'RCURLY',
    'IF', 'ELSE', 'WHILE', 'PRINT',
    'LPAREN', 'RPAREN','LBRACKET', 'RBRACKET',
    'FUN',
    )

# Tokens
t_CONCAT = r'::'
t_LESS = r'<'
t_LESSEQUAL = r'<='
t_EQUAL = r'=='
t_ASSIGN  = r'='
t_LPAREN  = r'\('
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_COMMENT = r'\#'
t_RPAREN  = r'\)'
t_PLUS    = r'\+'
t_SEMICOLON = r';'
t_LCURLY = r'{'
t_RCURLY = r'}'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_POWER = r'\*\*'
t_NOTEQUAL = r'<>'
t_GREATER = r'>'
t_GREATEREQUAL = r'>='

def t_NUMBER(t):
    r'-?\d*(\d\.|\.\d|\d)\d*[Ee](-|\+)?\d+|\d*(\d\.|\.\d)\d*|\d+'
    try:
        t.value = Num_nod(t.value)
    except ValueError:
        print("Syntax Error")
        exit()
        t.value = 0
    return t

def t_STRING(t):
    r'\'((\\\')*(\\\")*(\\\\)*[^\'\\]*)*\'|\"((\\\')*(\\\")*(\\\\)*[^\\\"]*)*\"'
    try:
        t.value = String_nod(t.value)
    except:
        print("Syntax Error")
        exit()
    return t

def t_BOOLEAN(t):
    'False|True'
    t.value = Bool_nod(t.value)
    return t

def t_VARIABLE(t):
    r'[A-Za-z][A-Za-z0-9_]*'
    try:
        if t.value in reserved:
            t.type = reserved[t.value]
        else:
            t.value = Var_node(t.value)
    except:
        print("Syntax Error")
        exit()
    return t

t_ignore = " \t"

def t_error(t):
    print("Syntax Error")
    exit()

import ply.lex as lex
lex.lex(debug=0)

precedence = (
    ('left', 'ORELSE'),
    ('left', 'ANDALSO'),
    ('left', 'NOT'),
    ('left', 'GREATER', 'GREATEREQUAL', 'LESS', 'LESSEQUAL', 'EQUAL', 'NOTEQUAL'),
    ('right', 'CONCAT'),
    ('left', 'IN'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE', 'MOD', 'DIV'),
    ('left', 'UMINUS'),
    ('right', 'POWER'),
    ('left', 'RBRACKET', 'LBRACKET'),
    ('left', 'COMMENT'),
    ('left', 'LPAREN', 'RPAREN')
    )

def p_pro(t):
    'program : functions block'
    t[0] = ProgramNode(t[1], t[2])

def p_pro1(t):
    'program : block'
    t[0] = t[1]

def p_funcs(t):
    'functions : functions function'
    t[0] = t[1] + [t[2]]

def p_funcs1(t):
    'functions : function'
    t[0] = [t[1]]

def p_func(t):
    'function : FUN VARIABLE LPAREN params RPAREN ASSIGN block expression SEMICOLON'
    t[0] = FuncNode(t[2], t[4], t[7], t[8])

def p_pms(t):
    'params : params COMMA VARIABLE'
    t[0] = t[1] + [t[3]]

def p_pms1(t):
    'params : VARIABLE'
    t[0] = [t[1]]

def p_expr_f_call(t):
    'expression : function_call'
    t[0] = t[1]

def p_f_call(t):
    'function_call : VARIABLE LPAREN args RPAREN'
    t[0] = FuncCaNode(t[1], t[3])

def p_args(t):
    'args : args COMMA expression'
    t[0] = t[1] + [t[3]]

def p_args1(t):
    'args : expression'
    t[0] = [t[1]]

def p_block(t):
    'block : LCURLY list_s RCURLY'
    t[0] = Blocks(t[2])

def p_block2(t):
    'block : LCURLY RCURLY'
    t[0] = Blocks(None)

def p_list_s(t):
    'list_s : list_s statement'
    t[0] = t[1]+[t[2]]

def p_list_s_num(t):
    'list_s : statement'
    t[0] = [t[1]]

def p_wholeBlock(t):
    'statement : block'
    t[0] = t[1]

def p_print_statement(t) :
    'statement : PRINT LPAREN expression RPAREN SEMICOLON'
    t[0] = PrintmyNode(t[3])

def p_while(t):
    'while : WHILE LPAREN expression RPAREN block'
    t[0] = While_loop(t[3], t[5])

def p_while_s(t):
    'statement : while'
    t[0] = t[1]

def p_if_s(t):
    'if_s : IF LPAREN expression RPAREN block'
    t[0] = If_statement(t[3], t[5])

def p_if_small(t):
    'statement : if_s'
    t[0] = t[1]

def p_if_and_else(t):
    'else_statement : if_s ELSE block'
    t[0] = Else_statement(t[1], t[3])

def p_else(t):
    'statement : else_statement'
    t[0] = t[1]

def p_statement_expression_semicolon(t):
    'statement : expression SEMICOLON'
    t[0] = t[1]

def p_assignment(t):
    'statement : expression ASSIGN expression SEMICOLON'
    t[0] = Assingment(t[1], t[3])

def p_expr_l_expr_r(t):
    'expression : LPAREN expression RPAREN'
    t[0] = t[2]

def p_binaryop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression
                  | expression MOD expression
                  | expression DIV expression'''
    t[0] = BopmyNode(t[2], t[1], t[3])

def p_operator(t):
    '''expression : expression GREATER expression
                  | expression GREATEREQUAL expression
                  | expression LESS expression
                  | expression LESSEQUAL expression
                  | expression EQUAL expression
                  | expression NOTEQUAL expression'''
    t[0] = Operation(t[2], t[1], t[3])

def p_expr_uminus(t):
    'expression : MINUS expression %prec UMINUS'
    t[0] = neg_nod(t[2])

def p_expr_booleanop(t):
    '''expression : expression ANDALSO expression
                  | expression ORELSE expression'''
    t[0] = BooleanOp(t[2], t[1], t[3])

def p_concatenation(t):
    'expression : expression CONCAT expression'
    t[0] = Concatenation(t[1], t[3])

def p_i_List(t):
    'expression : expression LBRACKET expression RBRACKET'
    t[0] = List_At_I(t[1], t[3])

def p_i_Tuple(t):
    'expression : COMMENT expression expression'
    t[0] = Tuple_At_I(t[2], t[3])

def p_i_Tuple2(t):
    'expression : COMMENT LPAREN expression RPAREN expression'
    t[0] = Tuple_At_I(t[3], t[5])

def p_expr_not(t):
    'expression : NOT expression'
    t[0] = Not_Statement( t[2])

def p_expr_in(t):
    'expression : expression IN expression'
    t[0] = In_statemnt( t[1], t[3])

def p_expr_fac(t):
    '''expression : factor'''
    t[0] = t[1]

def p_expr_l_l_r(t):
    'expression : LBRACKET list RBRACKET'
    t[0]= t[2]

def p_expr_assignment(t):
    'expression : VARIABLE'
    t[0] = t[1]

def p_expr_bool(t):
    'expression : BOOLEAN'
    t[0] = t[1]

def p_emptyList(t):
    'expression : LBRACKET RBRACKET'
    t[0] = ListmyNode(None)

def p_number(t):
    '''factor : NUMBER'''
    t[0] = t[1]

def p_string(t):
    'factor : STRING'
    t[0] = t[1]

def p_list(t):
    'list : expression'
    t[0] = ListmyNode(t[1])

def p_list_2(t):
    'list : list COMMA expression'
    t[0] = ListmyNode2(t[1], t[3])

def p_none_tuple(t):
    'expression : LPAREN RPAREN'
    t[0] = TuplemyNode(None)

def p_list_tuple(t):
    'expression : LPAREN list RPAREN'
    t[0] = TuplemyNode(t[2])

def p_error(t):
    print("Syntax Error")
    exit()

import ply.yacc as yacc
parser = yacc.yacc(debug = 0)

import sys

try:
    with open(sys.argv[1], 'r') as myf:
        output = parser.parse(myf.read().replace('\n', ''))
        output.evaluate()
except:
    pass
