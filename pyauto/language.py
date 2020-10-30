from pyauto.grammar import *

from utils.builder import *

from sympy import *

import itertools
import pandas
import sympy
from functools import reduce

# --------------------------------------------------------------------
#
# automata based rules for languages
#
# --------------------------------------------------------------------


class Constraint:
    def __init__(self, transition, initial, final, closure):
        self.transition, self.initial, self.final = transition, initial, final
        self.closure = closure
        self.fda = None

    def states(self):
        return sorted(self.fda.transition.states)

    def compile(self):
        self.fda = FiniteAutomata(self.transition, self.initial, self.final)
        return self

    def __call__(self, state, character):
        assert self.fda

        buffers = list(self.fda.transition(Buffer(data=character, initial=State(state))))

        non_error_states = list(map(lambda b: not b.error, buffers))

        if sum(non_error_states):
            assert sum(non_error_states) == 1

            index = non_error_states.index(True)
            return buffers[index].state()

        return ErrorState()


class ContainedRule(Constraint):
    def __init__(self, pattern, closure):
        self.pattern = pattern
        self.read_none = "read_none_0"

        states = {i + 1: "read_" + s + "_" + str(i + 1) for i, s in enumerate(self.pattern)}
        states[0] = self.read_none
        states[len(self.pattern)] = "read_" + self.pattern + "_" + str(len(self.pattern))

        transition = FADelta()
        for i, s in enumerate(self.pattern):
            transition.add(states[i], states[i + 1], s)

            for a in filter(lambda b: b != s, closure):
                transition.add(states[i], self.read_none, a)

        final = states[len(self.pattern)]
        for a in closure:
            transition.add(final, final, a)

        super().__init__(transition=transition, initial=states[0], final={final}, closure=closure)


class NotContainedRule(Constraint):
    def __init__(self, pattern, closure):
        self.pattern = pattern
        self.read_none = "read_none_0"

        states = {i + 1: "read_" + s + "_" + str(i + 1) for i, s in enumerate(self.pattern)}
        states[0] = self.read_none
        del states[len(self.pattern)]

        transition = FADelta()
        for i, s in enumerate(self.pattern):
            if i < len(self.pattern) - 1:
                transition.add(states[i], states[i + 1], s)

            for a in filter(lambda b: b != s, closure):
                transition.add(states[i], self.read_none, a)

        final = {states[i] for i in range(len(states))}
        super().__init__(transition=transition, initial=states[0], final=final, closure=closure)


class ParityRule(Constraint):
    def __init__(self, pattern, closure, is_even=True):
        self.pattern = pattern
        self.read_even = "read_even_" + self.pattern + "_0"
        self.read_odd = "read_odd_" + self.pattern + "_0"
        if is_even:
            self.read_any = "got_even_" + self.pattern + "_0"
        else:
            self.read_any = "got_odd_" + self.pattern + "_0"

        states = {i + 1: "read_even_" + s + "_" + str(i + 1) for i, s in enumerate(self.pattern)}
        states.update({i + 1 + len(self.pattern):
                           "read_odd_" + s + "_" + str(i + 1) for i, s in enumerate(self.pattern)})

        states[0] = self.read_even
        states[len(self.pattern)] = self.read_odd
        states[2 * len(self.pattern)] = self.read_even

        transition = FADelta()
        for i, s in enumerate(self.pattern):
            transition.add(states[i], states[i + 1], s)

            for a in filter(lambda b: b != s, closure):
                transition.add(states[i], self.read_even, a)

        for i, s in enumerate(self.pattern):
            transition.add(states[i + len(self.pattern)], states[i + len(self.pattern) + 1], s)

            for a in filter(lambda b: b != s, closure):
                transition.add(states[i + len(self.pattern)], self.read_odd, a)

        if is_even:
            final = {self.read_even}
        else:
            final = {self.read_odd}

        super().__init__(transition=transition, initial=states[0], final=final, closure=closure)


class EvenRule(ParityRule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, is_even=True)


class OddRule(ParityRule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, is_even=False)


class OrderRule(Constraint):
    def __init__(self, before, after, closure):
        self.state_before = "read_in_" + "".join([s for s in before]) + "_0"
        self.state_after = "read_inr_" + "".join([s for s in after]) + "_0"

        final = {self.state_before, self.state_after}

        transition = FADelta()
        for s in before:
            transition.add(self.state_before, self.state_before, s)

        for s in after:
            transition.add(self.state_after, self.state_after, s)

        for s in after.difference(before):
            transition.add(self.state_before, self.state_after, s)

        super().__init__(transition=transition, initial=self.state_before, final=final, closure=closure)


# --------------------------------------------------------------------
#
# rules declaration
#
# --------------------------------------------------------------------

class RuleDefinition:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self, closure):
        return self._build(**self.kwargs, closure=closure)

    def _build(self, **kwargs):
        raise RuntimeError("_build not implemented")


class Even(RuleDefinition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build(self, **kwargs):
        return EvenRule(**kwargs)


class Odd(RuleDefinition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build(self, **kwargs):
        return OddRule(**kwargs)


class Contained(RuleDefinition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build(self, **kwargs):
        return ContainedRule(**kwargs)


class NotContained(RuleDefinition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build(self, **kwargs):
        return NotContainedRule(**kwargs)


class Order(RuleDefinition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build(self, **kwargs):
        return OrderRule(**kwargs)


# --------------------------------------------------------------------
#
# regular languages definition
#
# --------------------------------------------------------------------


class RegularLanguage:
    def __init__(self, alphabet, definition):
        self.alphabet, self.definition = list(alphabet), definition
        self._sanity()

        closure = definition["closure"]
        self.rules = [rule.build(closure).compile() for rule in definition["rules"]]

        states = [sorted(each.fda.transition.states) for each in self.rules]

        states_map, initial, final = {}, None, set()
        for i, each in enumerate(list(itertools.product(*states))):
            state = "z" + str(i)
            states_map[each] = state

            all_initial = all(map(lambda ini, s: s == ini,
                                  [r.fda.initial for r in self.rules], [e for e in each]))

            all_final = all(map(lambda fin, s: s in fin,
                                [r.fda.final for r in self.rules], [e for e in each]))

            if all_initial:
                assert not initial
                initial = state

            if all_final:
                final.add(state)

        columns = ["states"] + ["P_" + str(i) for i in range(len(self.rules))] + \
                  ["type"] + [a for a in self.alphabet]
        self.frame = pandas.DataFrame(columns=columns)

        for each in states_map:
            state = states_map[each]

            next_states = []
            for a in self.alphabet:
                next_tuple = tuple(self.rules[i](s, a) if a in self.rules[i].fda.transition.alphabet else s
                                   for i, s in enumerate(each))
                if next_tuple in states_map:
                    next_states.append(states_map[next_tuple])
                else:
                    next_states.append("err")

            state_type = FiniteAutomata.NodeType.NONE
            if state == initial and state in final:
                state_type = FiniteAutomata.NodeType.INITIAL + "/" + FiniteAutomata.NodeType.FINAL
            elif state == initial:
                state_type = FiniteAutomata.NodeType.INITIAL
            elif state in final:
                state_type = FiniteAutomata.NodeType.FINAL

            row = [state] + [c for c in each] + [state_type] + next_states
            self.frame = pandas.concat([self.frame, pandas.DataFrame([row], columns=columns)])

        self.fda = FiniteAutomataBuilder. \
            get_finite_automata_from_frame(self.frame[["states", "type"] + self.alphabet]).strip_redundant()

    def _sanity(self):
        assert "rules" in self.definition
        assert "closure" in self.definition

        assert len(self.definition["closure"]) == len(self.alphabet)


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

        if len(rest):
            symbol = rest.pop()

            value = 0
            while self._get_partial_sum(current) <= length:
                current[symbol] = value

                conditions = [c.subs(current) for c in self.conditions]

                if not any([isinstance(c, Boolean) and isinstance(c, BooleanFalse) for c in conditions]):
                    self._generate_space(length, current.copy(), stack)

                value += 1

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
                    print(" -", s)

            if len(invalid_strings):
                print("\n\n[+] INVALID strings :")
                for s in invalid_strings:
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
        if len(self.lone_symbols):
            self.total_length += reduce(lambda w, v: len(str(w)) + len(str(v)), self.lone_symbols)

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
            expression.append(base**exponent)

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

# --------------------------------------------------------------------
#
# context-free languages definition
#
# --------------------------------------------------------------------


class ContextFreeLanguage:
    def __init__(self, expression, conditions):
        self.expression, self.conditions = expression, conditions
        self._sanity()

        exponent_relations = {}

        for each in self.conditions:
            lhs, rhs = each.lhs, each.rhs

            left_symbols = []
            if isinstance(lhs, sympy.core.symbol.Symbol):
                left_symbols.append(lhs)
            elif isinstance(each, sympy.core.relational.Relational):
                for ll in lhs.args:
                    left_symbols.append(ll)

            right_symbols = []
            if isinstance(rhs, sympy.core.symbol.Symbol):
                right_symbols.append(rhs)
            elif isinstance(each, sympy.core.relational.Relational):
                for rr in rhs.args:
                    right_symbols.append(rr)

            for ll in left_symbols:
                if ll not in exponent_relations:
                    exponent_relations[ll] = {ll}

                for rr in right_symbols:
                    if rr not in exponent_relations:
                        exponent_relations[rr] = {rr}

                    exponent_relations[ll].update(left_symbols + right_symbols)
                    exponent_relations[rr].update(left_symbols + right_symbols)

        self.relations = {frozenset(e): list() for e in exponent_relations.values()}
        self.constraints = {frozenset(e): 1 for e in exponent_relations.values()}

        for i, expr in enumerate(self.expression):
            self.relations[self._get_group_from_expression(expr)].append(i)

        for expr in self.conditions:
            self.constraints[self._get_group_from_expression(expr)] += 1

        if not all(map(lambda r, c: len(self.relations[r]) <= self.constraints[c],
                       self.relations, self.constraints)):
            raise RuntimeError("not context free")

    def check(self, data):
        exponent = {}
        for expr in self.expression:
            if expr.base not in exponent:
                exponent[expr.base] = expr.exp
            else:
                exponent[expr.base] = exponent[expr.base] + expr.exp

        count = {expr.base: data.count(str(expr.base)) for expr in self.expression}
        sol = solve([Eq(exponent[e], count[e]) for e in exponent])

        if len(sol):
            return all(map(lambda c: c.subs(sol), self.conditions))

        return False

    def _get_group_from_expression(self, expr):
        if isinstance(expr, sympy.core.power.Pow):
            expression = expr.exp
        elif isinstance(expr, sympy.core.relational.Relational):
            expression = expr

        groups = [each for each in self.relations if expression.free_symbols.issubset(each)]
        assert len(groups) == 1

        return groups[0]

    def _sanity(self):
        for each in self.expression:
            assert isinstance(each, sympy.core.power.Pow)

        for each in self.conditions:
            assert isinstance(each, sympy.core.relational.Relational)

    def generate_grammar(self):
        grammar = OpenGrammar()
        relations = list(self.relations.items())
        return grammar

