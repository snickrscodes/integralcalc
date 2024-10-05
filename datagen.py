import sympy as sp
import numpy as np

VAR = sp.Symbol('x')
INT_FUNC = {sp.erf, sp.erfc, sp.erfi, sp.erfinv, sp.erfcinv, sp.expint, sp.Ei, sp.li, sp.Li, sp.Si, sp.Ci, sp.Shi, sp.Chi, sp.fresnelc, sp.fresnels}
OPS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',
        sp.sign: 'sign',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        sp.cot: 'cot',
        sp.sec: 'sec',
        sp.csc: 'csc',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
        sp.acot: 'acot',
        sp.asec: 'asec',
        sp.acsc: 'acsc',
        # Hyperbolic Functions
        sp.sinh: 'sinh',
        sp.cosh: 'cosh',
        sp.tanh: 'tanh',
        sp.coth: 'coth',
        sp.sech: 'sech',
        sp.csch: 'csch',
        # Hyperbolic Inverses
        sp.asinh: 'asinh',
        sp.acosh: 'acosh',
        sp.atanh: 'atanh',
        sp.acoth: 'acoth',
        sp.asech: 'asech',
        sp.acsch: 'acsch',
        # Derivative
        sp.Derivative: 'derivative',
}

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
        'abs': 1,
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'asinh': 1,
        'acosh': 1,
        'atanh': 1
}

params = dict(
    num_constants=4, 
    num_ints=5,
    max_ops=5
)

class DataGenerator(object):
    def __init__(self, params: dict):
        self.num_constants = params['num_constants']
        self.num_ints = params['num_ints']
        self.max_ops = params['max_ops']
        self.operators = sorted(list(OPERATORS.keys()))
        self.constants = [f'a{x}' for x in range(self.num_constants)]
        self.integers = [str(x) for x in range(self.num_ints)]
        self.words = ['[START]', '[END]', 'pi', 'e', 'x', '.', '-'] + self.constants + self.integers + self.operators
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.n_words = len(self.words)
        self.leaves = np.array(['x']+self.integers)
        self.unary_ops = np.array([x for x in self.operators if OPERATORS[x] == 1])
        self.binary_ops = np.array([x for x in self.operators if OPERATORS[x] == 2])
        self.nl = np.size(self.leaves)
        self.p1 = np.size(self.unary_ops)
        self.p2 = np.size(self.binary_ops)
        self.ubi_dist = self.generate_ubi_dist(self.max_ops)
        self.cache = {} # this will get converted to a dataset later on
    
    def prefix_to_infix(self, tokens: list[str]):
        stack = []
        for i in range(len(tokens)-1, -1, -1):
            if tokens[i] in self.operators:
                if OPERATORS[tokens[i]] == 1:
                    match tokens[i]:
                        case 'neg':
                            stack.append(f'(-{stack.pop()})')
                        case 'inv':
                            stack.append(f'(1/{stack.pop()})')
                        case 'pow2':
                            stack.append(f'({stack.pop()}**2)')
                        case 'pow3':
                            stack.append(f'({stack.pop()}**3)')
                        case 'pow4':
                            stack.append(f'({stack.pop()}**4)')
                        case 'pow5':
                            stack.append(f'({stack.pop()}**5)')
                        case _:
                            stack.append(f'{tokens[i]}({stack.pop()})')
                else:
                    match tokens[i]:
                        case 'add':
                            stack.append(f'({stack.pop()}+{stack.pop()})')
                        case 'sub':
                            stack.append(f'({stack.pop()}-{stack.pop()})')
                        case 'mul':
                            stack.append(f'({stack.pop()}*{stack.pop()})')
                        case 'div':
                            stack.append(f'({stack.pop()}/{stack.pop()})')
                        case 'pow':
                            stack.append(f'({stack.pop()}**{stack.pop()})')
                        case 'rac':
                            stack.append(f'({stack.pop()}**(1/{stack.pop()}))')
            else:
                stack.append(tokens[i])
        return stack[-1]
    
    def generate_binary_dist(self, max_ops):
        # the maximum number of nodes in a tree with n operators is 2n+1
        """
        D[e][n] = the number of different binary trees with n nodes and e empty nodes
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # that means as e increases by 1 n decreases by 1 too
        D = []
        D.append([0]*(2*max_ops+1))
        catalans = [1]
        for i in range(1, 2*max_ops):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))
        D.append(catalans)
        for e in range(2, max_ops+1): # want a tree with n ops
            D.append([D[e - 1][n + 1] - D[e - 2][n + 1] for n in range(2*max_ops-e+1)])
        return D

    def generate_ubi_dist(self, max_ops):
        # enumerate possible trees
        D = []
        D.append([0] + [1]*(2*self.max_ops+1))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                # the first term combines the previous number of structures with terminal nodes (leaves)
                # the second term combines the previous number of structures with unary operators
                # the third term does the same thing for binary operators, but e+1 because 2 (one more) child 
                s.append(self.nl * s[e - 1] + self.p1 * D[n - 1][e] + self.p2 * D[n - 1][e + 1])
            D.append(s)
        # transpose the table so e = empty nodes and n = num ops
        D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
        return D

    def sample_next_pos_ubi(self, nb_empty, nb_ops):
        probs = []
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1])
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = np.random.choice(2 * nb_empty, p=probs)
        type = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, type

    def generate_function(self, n_ops):
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1
        def generate_leaf(max_int=self.num_ints-1):
            leaf_type = np.random.choice(np.arange(4))
            if leaf_type == 0:
                return ['x']
            elif leaf_type == 1:
                return [self.constants[np.random.randint(self.num_constants)]]
            elif leaf_type == 2:
                c = np.random.randint(1, max_int + 1)
                return [str(c)] if (np.random.randint(2) == 0) else ['neg', str(c)]
            else:
                return [self.constants[np.random.randint(self.num_constants)]]
        # create tree
        for nb_ops in range(n_ops, 0, -1):

            # next operator, type and position
            skipped, type = self.sample_next_pos_ubi(nb_empty, nb_ops)
            if type == 1:
                op = np.random.choice(self.unary_ops)
            else:
                op = np.random.choice(self.binary_ops)

            nb_empty += OPERATORS[op] - 1 - skipped  # created empty nodes - skipped future leaves
            l_leaves += skipped                           # update number of left leaves
            t_leaves += OPERATORS[op] - 1
            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(OPERATORS[op])] + stack[pos + 1:]
        leaves = [generate_leaf() for _ in range(t_leaves)]
        if not any(len(leaf) == 1 and leaf[0] == 'x' for leaf in leaves):
            leaves[-1] = ['x']
        np.random.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        return stack
    
    def generate_expression(self, n_ops=4):
        return sp.parse_expr(self.prefix_to_infix(self.generate_function(n_ops)), evaluate=True)

    def invalid_expr(self, expr) -> bool:
        return expr.has(sp.Integral) or expr.has(sp.I) or expr.has(sp.oo) or expr.has(sp.Piecewise) or isinstance(expr, sp.integrals.risch.NonElementaryIntegral) or any(op.func in INT_FUNC for op in sp.preorder_traversal(expr))

    def generate_minibatch(self, n_ops=4):
        # generates 6 examples, 2 forward mode, 2 backward mode, 2 ibp mode
        # returns 6 derivative and integral pairs if all works well...
        # first generate expressions
        expr1 = self.generate_expression(n_ops)
        expr2 = self.generate_expression(n_ops)
        # take derivatives for backward mode
        d1 = sp.simplify(sp.diff(expr1, VAR), seconds=1)
        d2 = sp.simplify(sp.diff(expr2, VAR), seconds=1)
        self.cache.__setitem__(d1, expr1)
        self.cache.__setitem__(d2, expr2)
        # integrate them for forward mode
        int1 = sp.simplify(sp.integrate(expr1, VAR, risch=True), seconds=1)
        int2 = sp.simplify(sp.integrate(expr2, VAR, risch=True), seconds=1)
        if not self.invalid_expr(int1):
            self.cache.__setitem__(expr1, int1)
        if not self.invalid_expr(int2):
            self.cache.__setitem__(expr2, int2)
        # integration by parts mode
        # the paper implemented this by checking if F*g or G*f was already in training data
        h1 = sp.simplify(expr1 * d2, seconds=1)
        h2 = sp.simplify(expr2 * d1, seconds=1)
        H1 = self.cache.get(h1)
        H2 = self.cache.get(h2)
        if H1 is not None:
            self.cache.__setitem__(h1, sp.simplify(expr1*expr2 - H1, seconds=1))
        if H2 is not None:
            self.cache.__setitem__(h2, sp.simplify(expr1*expr2 - H2, seconds=1))

gen = DataGenerator(params)
# data generator will take too long
for i in range(100):
    gen.generate_minibatch(6)
    print(True)