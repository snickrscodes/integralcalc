import re
import shunting_yard as sy

OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'neg': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'cbrt': 1,
        'exp': 1,
        'ln': 1,
        'log': 1,
        'abs': 1,
        'sign': 1,
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'csc': 1,
        'sec': 1,
        'cot': 1,
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acsc': 1,
        'asec': 1,
        'acot': 1,
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'csch': 1,
        'sech': 1,
        'coth': 1,
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acsch': 1,
        'asech': 1,
        'acoth': 1
}
BIN_OPS = ['+', '-', '*', '/', '^', 'add', 'sub', 'mul', 'div', 'pow', 'rac']
UN_OPS = list(filter(lambda x: OPERATORS[x] == 1, OPERATORS.keys()))
OPS = BIN_OPS + UN_OPS

def prio(op: str):
    if op in ['add', 'sub', '+', '-']:
        return 1
    elif op in ['mul', 'div', '*', '/']:
        return 2
    elif op in ['pow', 'rac', '^']:
        return 3
    elif op in UN_OPS:
        return 4
    return 0
 
def tokenize(expression):
    tokens = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isdigit():
            num, spaces = find_integer(expression, index)
            tokens.append(num)
            index += spaces
        elif char.isalpha():
            cur_op, length = find_any_operator(expression, index)
            if cur_op is not None:
                tokens.append(cur_op)
                index += length
            else:
                # Append alphabets as is (could be part of a function or variable name)
                tokens.append(char)
                index += 1
        elif char in '()+-*/^':
            tokens.append(char)
            index += 1
        else:
            index += 1
    return tokens

def postfix_to_prefix(tokens):
    stack = []
    for token in tokens:
        if token in BIN_OPS:
            operand2 = stack.pop()
            operand1 = stack.pop()
            new_expr = [token] + operand1 + operand2
            stack.append(new_expr)
        elif token in UN_OPS:
            operand = stack.pop()
            new_expr = [token] + operand
            stack.append(new_expr)
        else:
            stack.append([token])
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    return stack[0]

# def to_prefix(infix: list[str]):
#     infix = infix[::-1]
#     for i in range(len(infix)):
#         if infix[i] == '(':
#             infix[i] = ')'
#         elif infix[i] == ')':
#             infix[i] = '('
#     postfix = to_postfix(infix)
#     return postfix[::-1]

def find_main_operator(expr):
        min_p = 4
        index = -1
        bracket_count = 0
        for i, c in enumerate(expr):
            if c == '(':
                bracket_count += 1
            elif c == ')':
                bracket_count -= 1
            elif bracket_count == 0 and c in BIN_OPS:
                p = prio(c)
                if p <= min_p:
                    min_p = p
                    index = i
        return index

def find_any_operator(expression, start_index, max_length=5):
    last_op, last_index = None, 0
    for index in range(start_index, min(len(expression), start_index+max_length+1)):
        segment = expression[start_index:index]
        if segment in OPS:
            last_op, last_index = segment, index-start_index
    return last_op, last_index

def find_unary_operator(expression, start_index, max_length=5):
    last_op, last_index = None, 0
    for index in range(start_index, min(len(expression), start_index+max_length+1)):
        segment = expression[start_index:index]
        if segment in UN_OPS:
            last_op, last_index = segment, index-start_index
    return last_op, last_index

def find_integer(expression, start_index):
    num, spaces = '', 0
    for index in range(start_index, len(expression)):
        char = expression[index]
        if char.isdigit() or char == '.' or (index == start_index and char == '-'):
            num += char
            spaces += 1
        else:
            break
    return num, spaces

def find_first(expr: str, chars: str):
    match = re.search(f"[{re.escape(chars)}]", expr)
    if match:
        return match.start()
    return len(expr)

def find_stop_index(expression: str, chars='+-*/^', start_index=0):
    open_parentheses = 0
    i = start_index
    while i < len(expression):
        char = expression[i]
        if char == '(':
            open_parentheses += 1
        elif char == ')':
            open_parentheses -= 1
        if open_parentheses == 0:
            op, length = find_unary_operator(expression, i)
            if op is not None:
                i += length-1
            else:
                return i
        i += 1
    return len(expression) # if all fails

def match_parentheses(expression: str, start_index=0):
    open_parentheses = 0
    for i in range(start_index, len(expression)):
        char = expression[i]
        if char == '(':
            open_parentheses += 1
        elif char == ')':
            open_parentheses -= 1
        if open_parentheses == 0:
            return i # only stop when we have closed all parentheses
    return len(expression) # if all fails

def add_parentheses(expr):
    if len(expr) > 0 and expr[0] == '(' and match_parentheses(expr) == len(expr)-1: # if the whole expression is in parentheses
        return add_parentheses(expr[1:-1]) # then get rid of it: ((x^2+1)) -> (x^2+1)
    main_op_index = find_main_operator(expr)
    if main_op_index == -1:
        return expr
    left_expr = add_parentheses(expr[:main_op_index])
    right_expr = add_parentheses(expr[main_op_index + 1:])
    main_op = expr[main_op_index]
    return f'({left_expr}{main_op}{right_expr})'

def process_unary_operators(expression):
    result = ''
    i = 0
    while i < len(expression):
        cur_op, length = find_unary_operator(expression, i)
        if cur_op is not None:
            result += cur_op
            i += length
            # first case: an integer immediately follows like 'sinh12' -> 'sinh(12)'
            num, spaces = find_integer(expression, i)
            if num != '':
                result += f'({num})'
                i += spaces-1
            elif expression[i].isalnum():
                next_op, next_len = find_unary_operator(expression, i)
                if next_op is None:
                    result += f'({expression[i]})' # case 2: 'sinhx' -> 'sinh(x)'
                else:
                    stop_index = find_stop_index(expression[i:], '+-*/^0123456789abcdefghijklmnopqrstuvwxyz')
                    stop_cond = stop_index+i+1
                    new_expr = process_unary_operators(expression[i:stop_cond])
                    result += f'({new_expr})'
                    i += stop_index
                
            elif expression[i] == '(': # case 3: ops in parentheses 'sinh(expression)' remains unchanged
                stop_index = match_parentheses(expression[i:]) # only look from current position forward
                if stop_index > 0: 
                    result += process_unary_operators(expression[i:i+1+stop_index]) # nested operators
                    i += stop_index
            else:
                # case 4: if the next statement immediately after is a unary operator 'sinhtanx' -> 'sinh(tan(x))
                next_op, next_len = find_unary_operator(expression, i)
                if next_op is not None:
                    stop_index = find_stop_index(expression[i:], '+-*/^0123456789abcdefghijklmnopqrstuvwxyz')
                    stop_cond = stop_index+i+1
                    new_expr = process_unary_operators(expression[i:stop_cond])
                    result += f'({new_expr})'
                    i += stop_index
                else:
                    result += expression[i]
        else:
            result += expression[i]
        i += 1
    return result

def find_unary_term(expression, start_index, max_length=5):
    op, length = find_unary_operator(expression, start_index, max_length)
    if op is None: return None, 0, None, 0
    # because unary operators have been previously formatted
    # we can assume that all of their arguments will have proper closed parentheses
    arg_index = match_parentheses(expression, start_index+length)
    return (expression[start_index:arg_index+1], arg_index-start_index, op, length)

def insert_implicit_multiplication(input: str) -> str:
    table = []
    k = 0
    expression = ''
    while k < len(input):
        term, length, op, op_length = find_unary_term(input, k)
        if term is not None:
            expression += 'F'
            table.append((op+'('+insert_implicit_multiplication(term[op_length+1:-1])+')', 'F')) # replace unary expressions with a placeholder character
            k += length
        else:
            expression += input[k]
        k += 1
    # use regex expressions to handle easy algebraic terms
    expression = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expression) # number and variable
    expression = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expression) # variable and number
    expression = re.sub(r'([a-zA-Z\d])(\()', r'\1*\2', expression) # variable and parentheses
    expression = re.sub(r'(\))([a-zA-Z\d])', r'\1*\2', expression) # parentheses and number
    expression = re.sub(r'(\))(\()', r'\1*\2', expression) # adjacent parentheses
    expression = re.sub(r'([a-zA-Z])(?=[a-zA-Z])', r'\1*', expression) # adjacent variables
    expression = add_parentheses(expression) # add parentheses around binary operators
    # use the lookup table to substitute unary terms back in
    result = ''
    count = 0
    for i in range(len(expression)):
        if expression[i] == 'F':
            result += table[count][0]
            count += 1
        else:
            result += expression[i]
    return result

def convert_powers(expression):
    result = re.sub(r'-(\w+)\^([^\)]+)', r'-(\1^\2)', expression)
    result = re.sub(r'-\(([^)]+)\)\^([^\)]+)', r'-((\1)^\2)', result)
    return result

def convert_unary_minus(expression):
    i = 0
    result = ''
    def is_unary_function(expression, index):
        op, _ = find_unary_operator(expression, index)
        return op is not None
    while i < len(expression):
        if expression[i] == '-':
            if i == 0 or expression[i-1] in BIN_OPS or is_unary_function(expression, max(0, i-5)):
                result += 'neg'
                i += 1
                if expression[i] == '(':
                    end_index = match_parentheses(expression, i)
                    result += '(' + convert_unary_minus(expression[i+1:end_index]) + ')'
                    i = end_index
                elif expression[i].isalpha() or expression[i] == '-':
                    result += convert_unary_minus(expression[i:])
                    break
                else:
                    result += expression[i]
            else:
                result += '-'
        else:
            result += expression[i]
        i += 1
    return result

def show_interp(expression: str):
    parsed = expression.replace(' ', '')
    parsed = convert_unary_minus(parsed)
    parsed = process_unary_operators(parsed)
    parsed = insert_implicit_multiplication(parsed)
    return input(f'original input: {expression}\nparsed input: {parsed}\nis this reformatted version of the input correct? ')

def insert_rac(expression: str):
    result = re.sub(r'\(([^()]+)\)\^\(1/\(([^()]+)\)\)', r'(\1)rac(\2)', expression)
    result = re.sub(r'\(([^()]+)\)\^\(1/([^()]+)\)', r'(\1)rac\2', result) # expression1 has parentheses but expression2 does not: (expression1)^(1/expression2)
    result = re.sub(r'([^()\s]+)\^\(1/\(([^()]+)\)\)', r'\1rac(\2)', result) # expression1 has no parentheses but expression2 has parentheses: expression1^(1/(expression2))
    result = re.sub(r'([^()\s]+)\^\(1/([^()]+)\)', r'\1rac\2', result) # neither expression1 nor expression2 has parentheses: expression1^(1/expression2)
    return result

def replace_binaries(expression: str):
    result = expression.replace('+', 'add')
    result = result.replace('-', 'sub')
    result = result.replace('*', 'mul')
    result = result.replace('/', 'div')
    result = result.replace('^', 'pow')
    return result

def replace_pow(match):
    base = match.group(1)
    exponent = match.group(2)
    return f'pow{exponent}({base})'

def process_expression(expr: str) -> str: # converts to pow_n format x^2 -> pow2(x)
    expr = re.sub(r'\(([^()]+)\)\^([2345])', replace_pow, expr)
    expr = re.sub(r'([^\s()]+)\^([2345])', replace_pow, expr)
    return expr

def find_power_term(input: str, start_index):
    # the input expression will have parentheses around all ops
    _, length = find_unary_operator(input, start_index)
    index = match_parentheses(input, start_index+length)
    if index + 3 <= len(input) and input[index+1:index+3] in ['^2', '^3', '^4', '^5']:
        return input[start_index:index+3], index+2-start_index, input[index+2]
    return None, 0, None
        
def convert_pow_n(input: str) -> str:
    table = []
    k = 0
    expression = ''
    while k < len(input):
        term, length, power = find_power_term(input, k)
        if term is not None:
            expression += 'F'
            table.append(('(pow'+str(power)+'('+convert_pow_n(term[:-2])+'))', 'F')) # replace unary expressions with a placeholder character
            k += length
        else:
            expression += input[k]
        k += 1
    expression = process_expression(expression)
    # use the lookup table to substitute terms back in
    result = ''
    count = 0
    for i in range(len(expression)):
        if expression[i] == 'F':
            result += table[count][0]
            count += 1
        else:
            result += expression[i]
    return result

def replace_exponential(expression):
    result = re.sub(r"e\^(\(.+\)|[^()])", lambda match: "exp("+replace_exponential(match.group(1))+")", expression)
    return result

def swap_parentheses(s):
    s = re.sub(r'\(', '\x00', s)
    s = re.sub(r'\)', '(', s)
    s = re.sub('\x00', ')', s)
    return s

def transform_tokens(tokens):
    transformed = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            transformed.append('INT+')
            for digit in tokens[i]:
                transformed.append(digit)
        elif token == 'neg' and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            transformed.append('INT-')
            for digit in tokens[i+1]:
                transformed.append(digit)
            i += 1
        else:
            transformed.append(token)
        i += 1
    return transformed

def format(input: str):
    result = input.replace(' ', '') # first remove any spaces
    result = convert_powers(result) # convert power expressions to parse in unary function
    result = convert_unary_minus(result) # convert unary minus to 'neg'
    result = process_unary_operators(result) # add parentheses around all unary operators
    result = insert_implicit_multiplication(result) # then add parentheses around all binary operators and add implicit multiplication
    result = convert_pow_n(result)
    result = replace_exponential(result)
    result = add_parentheses(result) # get rid of unnecessary parentheses
    return result

def parse_expr(input: str):
    result = input.replace(' ', '') # first remove any spaces
    result = convert_powers(result) # convert power expressions to parse in unary function
    result = convert_unary_minus(result) # convert unary minus to 'neg'
    result = process_unary_operators(result) # add parentheses around all unary operators
    result = insert_implicit_multiplication(result) # then add parentheses around all binary operators and add implicit multiplication
    result = convert_pow_n(result)
    result = replace_exponential(result)
    result = add_parentheses(result) # get rid of unnecessary parentheses
    # result = insert_rac(result)
    # result = replace_binaries(result)
    # result = tokenize(result) # convert this to tokens
    result = sy.shunting_yard(result)
    result = result.split(' ')
    result = list(map(lambda x: x.replace('+', 'add'), result))
    result = list(map(lambda x: x.replace('-', 'sub'), result))
    result = list(map(lambda x: x.replace('*', 'mul'), result))
    result = list(map(lambda x: x.replace('/', 'div'), result))
    result = list(map(lambda x: x.replace('^', 'pow'), result))
    result = postfix_to_prefix(result)
    result = transform_tokens(result)
    return result

# test cases
# expressions = [
#     "log(sin(x^2)+cos(y^2))",
#     "sqrt(x^2 + y^2) + exp(x) * log(y)",
#     "a * (b + c * (d - e)) / f",
#     "(a + b) * (c - d) / (e + f)",
#     "x^3 + y * sin(z) - log(x * y)",
#     "tan(sin(x) + cos(y)) * exp(z)",
#     "(a + b) * (c / d) + sqrt(e - f)",
#     "exp(log(x + y) * z)",
#     "x^(y + z) * (a + b)",
#     "csc(x) + sec(y) - tan(z)",
#     "sqrt(a^2 + b^2) / (x + y) * (log(z) + e^x)",
#     "sin(a + b) * (cos(x) + tan(y))",
#     "log(sqrt(x) * (y + z)) - exp(x - y)",
#     "a^(b + (c * d))",
#     "2 * (x + y) / (z - a) + sin(b) * cos(c)"
#     "e^x",
#     "e^(x^2)",
#     "e^(e^x)",
#     "x+5",
#     "-tanh(x-2)",
#     "-x+3",
#     "exp(-x^2)+sqrt(-y)",
#     "sinh12x",
#     "asinhx+2x",
#     "atanhsinx",
#     "cosh(x)",
#     "sinhxcoshx",
#     "tanh(sinhx+3x)",
#     "tanh(sinhxcoshx)",
#     "sincostanx",
#     "coscotsec(x^2)+2x",
#     "tanh(sin(x)+3x)",
#     "csc(sinx+x^2)",
#     "tancos(x)",
#     "a(b+c)d",
#     "x(x+1)(y+2)",
#     "a(b(c+d)e)f",
#     "x(y(z))",
#     "x(y+z)w",
#     "x(y+z)(a+b)",
#     "a(x(y+z))b",
#     "x(y+z)^2",
#     "x(y+z)w(z+a)",
#     "a(x^2+y^2)b(c+d)",
#     "2x",
#     "x2",
#     "2x+3x^2"
#     "sin(x)cos(y)",
#     "tan(sin(x))cos(x)",
#     "cosh(sinh(x))x",
#     "sqrt(x)(y+1)",
#     "log(sin(x)+cos(x))^2",
#     "exp(x)(y+z)",
#     "csc(x)sec(y)",
#     "a(tan(x)+b)^2",
#     "e^(x+y)(z+1)",
#     "sin(x+y)cos(z)"
# ]

# for x in expressions:
#     print(f'{x} -> {parse_expr(x)}')

# tokens = list(map(lambda x: parse_expr(x), expressions))