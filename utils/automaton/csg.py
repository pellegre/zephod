from utils.language.grammar import *
from utils.language.formula import *

from functools import reduce


class GrammarLeaf:
    def __init__(self, language, initial, grammar):
        if len(language.symbols) > 1:
            language.info()
            raise RuntimeError("invalid language for leaf")

        self.language = language
        self.initial = initial

        self.grammar = grammar

    @staticmethod
    def build(language, initial, grammar):
        if len(language.expression) == 1:
            expr = language.expression[0]
            if isinstance(expr, Symbol):
                return LoneLeaf(language=language, initial=initial, grammar=grammar)

            else:
                return SingleLeaf(language=language, initial=initial, grammar=grammar)

        elif len(language.expression) == 2:
            return TupleLeaf(language=language, initial=initial, grammar=grammar)

        elif len(language.expression) == 3:
            return TripletLeaf(language=language, initial=initial, grammar=grammar)


class LoneLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.language.expression) == 1

        expr = self.language.expression[0]
        assert isinstance(expr, Symbol)
        self.non_terminals = {expr: self.initial}


class SingleLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 1

        expr = self.language.expression[0]

        assert isinstance(expr, Pow)

        non_terminal = self.grammar.get_non_terminal()
        self.non_terminals = {expr: non_terminal}

        self.grammar.add(self.initial, self.initial + non_terminal)
        self.grammar.add(self.initial, non_terminal)


class TupleLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 2

        expr_left = self.language.expression[0]
        expr_right = self.language.expression[1]

        assert isinstance(expr_left, Pow) and isinstance(expr_right, Pow)

        non_terminal_left = self.grammar.get_non_terminal()
        non_terminal_right = self.grammar.get_non_terminal()

        self.non_terminals = {expr_left: non_terminal_left, expr_right: non_terminal_right}

        self.grammar.add(self.initial, non_terminal_left + self.initial + non_terminal_right)
        self.grammar.add(self.initial, non_terminal_left + non_terminal_right)


class TripletLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 3

        expr_a, ta = self.language.expression[0], self.grammar.get_non_terminal()
        expr_b, tb = self.language.expression[1], self.grammar.get_non_terminal()
        expr_c, tc = self.language.expression[2], self.grammar.get_non_terminal()

        self.non_terminals = {expr_a: ta, expr_b: tb, expr_c: tc}

        self.grammar.add(self.initial, self.initial + ta + tb + tc)
        self.grammar.add(self.initial, ta + tb + tc)

        self.grammar.add(tc + ta, ta + tc)
        self.grammar.add(tc + tb, tb + tc)
        self.grammar.add(tb + ta, ta + tb)


class GrammarTree:
    ONE = "one"

    @staticmethod
    def generate_rule_swapper(buffer, non_terminal_blocks, grammar):
        non_terminal_arrangement = reduce(lambda l, xx: l.append(xx) or l if xx not in l else l, buffer, [])

        for i in range(0, len(non_terminal_blocks)):
            symbol = non_terminal_blocks[i]
            for j in filter(lambda m: non_terminal_arrangement[m] == symbol,
                            range(0, len(non_terminal_arrangement))):

                if j > i:
                    for k in range(i, j):
                        neighbor = non_terminal_arrangement[k]

                        grammar.add(neighbor + symbol, symbol + neighbor)

                    after_reordering = list()
                    for k in range(0, i):
                        after_reordering.append(non_terminal_arrangement[k])

                    after_reordering.append(symbol)

                    for k in filter(lambda m: non_terminal_arrangement[m] != symbol,
                                    range(i, len(non_terminal_arrangement))):
                        after_reordering.append(non_terminal_arrangement[k])

                    non_terminal_arrangement = after_reordering

        return non_terminal_arrangement

    def __init__(self, language, initial="S"):
        self.language = LanguageFormula.normalize(language)
        self.tree = networkx.DiGraph()

        self.initial = initial
        self.non_terminal_counter = ord(initial)

        self.grammar = Grammar()

        self.tree.add_node(self.initial, language=self.language)

        self._build_tree(self.initial)

        leafs = networkx.get_node_attributes(self.tree, name="leaf")
        self.non_terminals, self.expression_non_terminals = {}, {}

        for node in leafs:
            for expr in leafs[node].non_terminals:
                self.non_terminals[expr] = leafs[node].non_terminals[expr]
                self.expression_non_terminals[leafs[node].non_terminals[expr]] = expr

        self.non_terminal_blocks = [self.non_terminals[e] for e in self.language.expression if e in self.non_terminals]

        self._swap_symbols()

        self._add_terminal_rules()

        genesis_block = self.non_terminal_blocks[0]
        expr = self.language.expression[0]

        if isinstance(expr, Symbol):
            terminal = str(expr)
            self.grammar.rules[self.initial] = [terminal + each for each in self.grammar.rules[self.initial]]
            self.grammar.add(terminal + genesis_block, terminal + str(self.language.expression[1].base))
        else:
            terminal = str(expr.base)

            grammar = Grammar()
            for each in self.grammar.rules:
                if genesis_block not in each:
                    for right in self.grammar.rules[each]:
                        if genesis_block in right:
                            if right[0] == genesis_block:
                                grammar.add(each, right.replace(genesis_block, terminal))
                            else:
                                grammar.add(each, right.replace(each + genesis_block, terminal + each))
                        else:
                            grammar.add(each, right)

            self.grammar = grammar

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _add_terminal_rules(self):
        non_terminal_arrangement = self.non_terminal_blocks

        block = 0
        for i in range(0, len(non_terminal_arrangement) - 1):
            symbol, next_symbol = non_terminal_arrangement[i], non_terminal_arrangement[i + 1]

            expr, next_expr = self.expression_non_terminals[symbol], self.expression_non_terminals[next_symbol]

            while self.language.expression[block] != expr and block < len(self.language.expression):
                block += 1

            next_block_expr = self.language.expression[block + 1]

            self.grammar.add(str(expr.base) + symbol, 2*str(expr.base))

            if isinstance(next_block_expr, Symbol):
                self.grammar.add(str(expr.base) + next_symbol,
                                 str(expr.base) + str(next_block_expr) + str(next_expr.base))
            else:
                self.grammar.add(str(expr.base) + next_symbol, str(expr.base) + str(next_expr.base))

        # omega
        last_symbol = non_terminal_arrangement[-1]
        last_expr = self.language.expression[-1]

        if isinstance(last_expr, Pow):
            self.grammar.add(str(last_expr.base) + last_symbol, 2 * str(last_expr.base))

    def _swap_symbols(self):
        for each in self.generate_non_terminal():
            self.generate_rule_swapper(each, self.non_terminal_blocks, self.grammar)

    def _is_root(self, node):
        return self.initial == node

    def _build_tree(self, node):
        language = networkx.get_node_attributes(self.tree, "language")

        blocks = []
        stack, block = set(), 0
        for each in language[node].expression:

            if isinstance(each, Pow):
                if len(language[node].expression_partition) > 1:
                    part = language[node].symbols_partition[each.exp]
                    expr = [e for e in language[node].expression if isinstance(e, Pow) and e.exp in part]
                    cond = [c for c in language[node].conditions if c.free_symbols.issubset(part)]

                else:
                    part = each.exp
                    expr = [e for e in language[node].expression if isinstance(e, Pow) and e.exp == part]
                    cond = [each.exp >= 0]

                if part not in stack:
                    stack.add(part)

                    non_terminal = self.grammar.get_non_terminal()

                    blocks.append(non_terminal)

                    lang = LanguageFormula(expression=expr, conditions=cond)

                    if len(lang.symbols) == 1:
                        self.tree.add_node(non_terminal, language=lang,
                                           leaf=GrammarLeaf.build(lang, non_terminal, self.grammar))

                    else:
                        self.tree.add_node(non_terminal, language=lang)
                        self._build_tree(non_terminal)

                    self.tree.add_edge(node, non_terminal, block=block)

                    block += 1

        if node == self.initial:
            self.grammar.add(node, ''.join(blocks))
        else:
            self.grammar.add(node, node + ''.join(blocks))
            self.grammar.add(node, ''.join(blocks))

    def _print_tree(self, node, space=""):
        blocks = networkx.get_edge_attributes(self.tree, "block")
        edges = sorted({e: blocks[e] for e in blocks if e[0] == node}, key=lambda e: blocks[e])

        attr = networkx.get_node_attributes(self.tree, name="language")

        print(space, node, "->", attr[node].expression)

        space += "   "
        if len(edges):
            for each in edges:
                self._print_tree(each[1], space)
        else:
            attr = networkx.get_node_attributes(self.tree, name="leaf")
            non_terminals = attr[node].non_terminals

            for each in non_terminals:
                print(space, non_terminals[each], "->", each)

    def _run_non_terminal_rules(self, node, buffer):
        blocks = networkx.get_edge_attributes(self.tree, "block")

        edges = sorted({e: blocks[e] for e in blocks if e[0] == node}, key=lambda e: blocks[e])

        if node in self.grammar.rules:
            recursive_rules = [r for r in self.grammar.rules[node] if node in r]
            terminal_rules = [r for r in self.grammar.rules[node] if node not in r]

            for rule in recursive_rules:
                if set([c for c in rule]).issubset(self.grammar.non_terminal):
                    buffer.run_rule(node, rule, times=3)

            for rule in terminal_rules:
                if set([c for c in rule]).issubset(self.grammar.non_terminal):
                    buffer.run_rule_until(node, rule)

            if len(edges):
                for each in edges:
                    self._run_non_terminal_rules(each[1], buffer)

    def info(self):
        print("\n[+] language")
        self.language.info()

        self._print_tree(self.initial)

        print("\n[+] grammar")
        print(self.grammar)

        print("\n[+] non terminal blocks", self.non_terminal_blocks)

    def generate_non_terminal(self):
        buffers = list()

        for initial_rule in self.grammar.rules[self.initial]:
            buffer = self.grammar.get_string()
            buffer.run_rule(self.initial, initial_rule)

            self._run_non_terminal_rules(self.initial, buffer)

            run_rules = True
            while run_rules:
                run_rules = False

                for each in self.grammar.rules:
                    if each in buffer.current and len(each) == 2 and \
                            set([c for c in each]).issubset(self.grammar.non_terminal):

                        for right in self.grammar.rules[each]:
                            if set([c for c in right]).issubset(self.grammar.non_terminal):
                                buffer.run_rule_until(each, right)
                                run_rules = True

            buffers.append(buffer.current)

        return buffers

    def generate_with_terminals(self):
        buffers = list()

        for each in self.generate_non_terminal():
            run_rules = True

            buffer = GrammarString(grammar=self.grammar, current=each)

            while run_rules:
                run_rules = False
                for left in self.grammar.rules:
                    if left in each:
                        for right in self.grammar.rules[left]:
                            each = buffer.run_rule(left, right)
                        run_rules = True

            buffers.append(each)

        return buffers

    def plot(self):
        attr = networkx.get_node_attributes(self.tree, name="language")
        expr = {s: s + " " + str(attr[s].expression) for s in attr}

        networkx.draw_networkx(self.tree, labels=expr)
        plt.show()
