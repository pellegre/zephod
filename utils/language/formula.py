from sympy import *
from sympy.logic.boolalg import *
from functools import reduce

# --------------------------------------------------------------------
#
# languages definition using formula
#
# --------------------------------------------------------------------


class ExponentSpace:
    MAX_LENGTH = 100

    def __init__(self, sym, conditions, length):
        self.symbols = sym
        self.conditions = conditions
        self.length = length

        self.free_symbols = {set(e.free_symbols).pop(): e for e in self.conditions if len(e.free_symbols) == 1}

    def info(self):
        print("[+] symbols", self.symbols)
        print("[+] conditions", self.conditions)
        print("[+] free_symbols", self.free_symbols)

    def _get_partial_sum(self, current):
        if isinstance(self.length, Add):
            partial = Number(0)

            for each in self.length.args:
                if isinstance(each, Number):
                    partial = partial + each
                elif set(each.free_symbols).pop() in current:
                    partial = partial + each

            return partial.subs(current)

        else:
            partial = Number(0)

            if next(iter(self.length.free_symbols)) in current:
                partial = partial + self.length

            return partial.subs(current)

    def _generate_space(self, length, current, stack):
        rest = set(self.symbols.difference(set(current.keys())))
        safety_counter = 0

        if len(rest):
            symbol = rest.pop()

            value = 0
            while self._get_partial_sum(current) <= length:
                current[symbol] = value

                conditions = [c.subs(current) for c in self.conditions]

                if not any([isinstance(c, Boolean) and isinstance(c, BooleanFalse) for c in conditions]):
                    self._generate_space(length, current.copy(), stack)

                value += 1
                safety_counter += 1

                if safety_counter % 1500 == 0:
                    print("[+] _generate_space is not meeting conditions...", current)

        else:
            if self._get_partial_sum(current) <= length:
                stack.append(current)

    def _generate_minimal_stack(self, current, stack):
        def get_total_sum(v):
            return reduce(lambda ss, rr: ss + rr, v.values())

        rest = set(self.symbols.difference(set(current.keys())))

        if len(rest):
            symbol = rest.pop()

            value = 0
            while not len(stack) or get_total_sum(current) <= get_total_sum(next(iter(stack))):
                current[symbol] = value

                conditions = [c.subs(current) for c in self.conditions]

                if not any([isinstance(c, Boolean) and isinstance(c, BooleanFalse) for c in conditions]):
                    self._generate_minimal_stack(current.copy(), stack)

                if self._get_partial_sum(current) > ExponentSpace.MAX_LENGTH:
                    break

                value += 1
        else:
            if not len(stack) or get_total_sum(current) <= get_total_sum(next(iter(stack))):
                stack.append(current)

    def get_space(self, length):
        stack = list()
        self._generate_space(length, dict(), stack)

        return stack

    def get_minimal(self, constraint=None):
        minimal_free = dict()

        for each in self.free_symbols:
            value = 0

            if constraint and each in constraint:
                while not self.free_symbols[each].subs({each: value}) or value < constraint[each]:
                    value += 1
            else:
                while not self.free_symbols[each].subs({each: value}):
                    value += 1

            minimal_free[each] = value

        stack = list()
        self._generate_minimal_stack(minimal_free, stack)

        return stack


class Language:
    def __init__(self):
        pass

    def enumerate_strings(self, length=5):
        raise RuntimeError("enumerate_strings not implemented")

    def check_grammar(self, grammar, length=5):
        generated_strings = set(self.enumerate_strings(length))
        max_length = len(max(generated_strings, key=lambda w: len(w)))

        print("[+] max string length :", max_length)

        grammar_strings = set(filter(lambda w: len(w) <= length, grammar.enumerate(length=length + 5)))

        missing_strings = generated_strings.difference(grammar_strings)
        invalid_strings = grammar_strings.difference(generated_strings)

        if not len(missing_strings) and not len(invalid_strings):
            return True
        else:
            if len(missing_strings):
                print("\n\n[+] MISSING strings :")
                for s in missing_strings:
                    print(" -", s, len(s))

            if len(invalid_strings):
                print("\n\n[+] INVALID strings :")
                for s in invalid_strings:
                    print(" -", s, len(s))
                    grammar.print_stack(s)

            return False


class LanguageFormula(Language):
    def __init__(self, expression, conditions):
        self.expression, self.conditions = expression, conditions

        self.symbols, self.symbols_partition = set(), dict()
        for c in self.conditions:
            for symbol in c.free_symbols:
                self.symbols.add(symbol)

            if symbol not in self.symbols_partition:
                self.symbols_partition[symbol] = {symbol}

            for other in c.free_symbols:
                self.symbols_partition[symbol].add(other)

                if other not in self.symbols_partition:
                    self.symbols_partition[other] = {other}

                self.symbols_partition[other].add(symbol)

        for symbol in self.symbols:
            for each in set(self.symbols_partition[symbol]):
                self.symbols_partition[symbol].update(self.symbols_partition[each])

            for each in set(self.symbols_partition[symbol]):
                self.symbols_partition[each].update(self.symbols_partition[symbol])

        self.symbols_partition = {s: frozenset(self.symbols_partition[s]) for s in self.symbols_partition}

        self.conditions_partition = {}
        for condition in self.conditions:
            symbol = next(iter(condition.free_symbols))
            symbol_set = self.symbols_partition[symbol]

            if symbol_set not in self.conditions_partition:
                self.conditions_partition[symbol_set] = list()

            self.conditions_partition[symbol_set].append(condition)

        self.expression_partition = {}
        self.lone_symbols = set()
        for expr in self.expression:
            if isinstance(expr, Pow):
                symbol = next(iter(expr.exp.free_symbols))
                symbol_set = self.symbols_partition[symbol]

                if symbol_set not in self.expression_partition:
                    self.expression_partition[symbol_set] = list()

                self.expression_partition[symbol_set].append(expr)
            else:
                self.lone_symbols.add(expr)

        self.group_length, self.total_length = {}, Number(0)
        if len(self.lone_symbols) > 1:
            self.total_length += reduce(lambda w, v: len(str(w)) + len(str(v)), self.lone_symbols)

        elif len(self.lone_symbols) == 1:
            self.total_length += len(str(next(iter(self.lone_symbols))))

        for part in self.expression_partition:
            length = Number(0)
            for each in self.expression_partition[part]:
                length = length + each.exp * len(str(each.base))
                self.total_length = self.total_length + each.exp * len(str(each.base))

            self.group_length[part] = length

        self.partitions = {self.symbols_partition[s] for s in self.symbols_partition}

    @staticmethod
    def normalize(language):
        expression, conditions = language.expression, language.conditions
        normalized_expression = list()

        for expr in expression:
            if isinstance(expr, Symbol):
                normalized_expression.append(expr)

            elif isinstance(expr, Pow):
                base = expr.base

                zeroed = {s: 0 for s in expr.exp.free_symbols}
                minimal = expr.exp.subs(zeroed)

                if minimal > 0:
                    normalized_expression.append(symbols(str(base) * minimal))

                rest = expr.exp - minimal

                LanguageFormula._push_expression(normalized_expression, base, rest)

        return LanguageFormula(expression=normalized_expression, conditions=conditions)

    @staticmethod
    def _push_expression(expression, base, exponent):
        if isinstance(exponent, Symbol):
            expression.append(base ** exponent)

        elif isinstance(exponent, Mul):
            assert len(exponent.free_symbols) == 1

            index = next(iter(exponent.free_symbols))
            assert isinstance(index, Symbol)

            multiplier = exponent.as_coefficient(index)

            symbol = symbols(str(base) * multiplier)
            LanguageFormula._push_expression(expression, symbol, index)

        elif isinstance(exponent, Add):
            for each in exponent.args:
                LanguageFormula._push_expression(expression, base, each)

        else:
            raise RuntimeError("unrecognized expression in exponent " +
                               str(exponent) + "@" + str(type(exponent)) + " - base " + str(base))

    def info(self):
        print("[+] expression", [str(e) + " (" + str(i) + ")" for i, e in enumerate(self.expression)])
        print("[+] partitions", self.partitions)
        print("[+] symbols", self.symbols)
        print("[+] symbol partition", self.symbols_partition)
        print("[+] condition partition", self.conditions_partition)
        print("[+] expression partition", self.expression_partition)
        print("[+] minimal index", self._get_minimum_indices())

        if len(self.lone_symbols):
            print("[+] lone symbols", self.lone_symbols)

        print("[+] total length expression", self.total_length)

    def enumerate_strings(self, length=5):
        space = ExponentSpace(sym=self.symbols, conditions=self.conditions, length=self.total_length)
        cross = space.get_space(length)

        return sorted(list(set([self._generate_string(each) for each in cross])), key=lambda s: len(s))

    def get_index_space(self, length=5):
        space = ExponentSpace(sym=self.symbols, conditions=self.conditions, length=self.total_length)
        return space.get_space(length)

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.symbols, conditions=self.conditions,
                              length=self.total_length)

        return space.get_minimal(constraints)

    def _generate_string(self, values):
        generated = str()
        for expr in self.expression:
            if isinstance(expr, Pow):
                local = str(expr.base)
                count = expr.exp.subs(values)
                generated += count * local
            else:
                generated += str(expr)

        return generated

    def __add__(self, other):
        languages = [self, other]
        return LanguageUnion(languages=languages)


class LanguageUnion(Language):
    def __init__(self, languages):
        self.languages = languages

    def info(self):
        for i, each in enumerate(self.languages):
            print("[*] language " + str(i))
            each.info()
            print()

    def enumerate_strings(self, length=5):
        enumerated = set()

        for each in self.languages:
            enumerated.update(each.enumerate_strings(length=length))

        return sorted(list(enumerated), key=lambda w: len(w))

    def __add__(self, other):
        if not isinstance(other, Language):
            raise RuntimeError("can't add language to " + str(type(other)))

        if not isinstance(other, LanguageUnion):
            return LanguageUnion(languages=self.languages + [other])
        else:
            return LanguageUnion(languages=self.languages + other.languages)
