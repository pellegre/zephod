from pyauto.finite_automata import *
from utils.builder import *

import itertools

import pandas

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

        buffers = list(self.fda.transition(Buffer(data=character, initial=state)))
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
# languages definition
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

            elif all_final:
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
            if state == initial:
                state_type = FiniteAutomata.NodeType.INITIAL
            elif state == initial and state in final:
                state_type = FiniteAutomata.NodeType.INITIAL + "/" + FiniteAutomata.NodeType.FINAL
            elif state in final:
                state_type = FiniteAutomata.NodeType.FINAL

            row = [state] + [c for c in each] + [state_type] + next_states
            self.frame = pandas.concat([self.frame, pandas.DataFrame([row], columns=columns)])

        self.fda = FiniteAutomataBuilder.\
            get_finite_automata_from_frame(self.frame[["states", "type"] + self.alphabet]).strip_redundant()

    def _sanity(self):
        assert "rules" in self.definition
        assert "closure" in self.definition

        assert len(self.definition["closure"]) == len(self.alphabet)
