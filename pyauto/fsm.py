from graphviz import Digraph
import re


class State:
    def __init__(self, name):
        self.name = name
        groups = re.search("([a-zA-Z]+)([0-9]+)", self.name).groups()
        self.prefix = groups[0]
        self.number = groups[1]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Error(State):
    def __init__(self):
        super().__init__(name="__fsm_error_state_" + str(hex(id(self))))


class Transition:
    EPSILON = "$"

    def __init__(self):
        self.states = set()
        self.alphabet = {Transition.EPSILON}
        self.delta = {}

    def add(self, ei, ef, symbols):
        assertion = [isinstance(s, str) for s in symbols]
        assert sum(assertion) == len(assertion)

        state_ei = State(ei)
        state_ef = State(ef)

        self.states.add(state_ei)
        self.states.add(state_ef)
        self.alphabet.update(symbols)

        if state_ei not in self.delta:
            self.delta[state_ei] = {}

        for s in symbols:
            if s not in self.delta[state_ei]:
                self.delta[state_ei][s] = set()

            self.delta[state_ei][s].add(state_ef)

    def __call__(self, state, string):
        assert state in self.states
        state_e = state
        if isinstance(state_e, str):
            state_e = State(state_e)

        next_states = set()
        if len(string):
            symbol = string[0]
            assert symbol in self.alphabet

            if symbol in self.delta[state_e]:
                next_states.update({(e, string[1:]) for e in self.delta[state_e][symbol]})

        if Transition.EPSILON in self.delta[state_e]:
            next_states.update({(e, string[:]) for e in self.delta[state_e][Transition.EPSILON]})

        return next_states

    def __str__(self):
        string = "states = " + str(self.states) + " ; alphabet = " + str(self.alphabet) + "\n"
        for e in self.delta:
            for s in self.delta[e]:
                string += str(e) + " (" + str(s) + ") -> " + str(self.delta[e][s]) + "\n"
        return string[:-1]

    def __repr__(self):
        return self.__str__()


class FiniteAutomata:
    def __init__(self, transition, initial, final):
        self.transition = transition
        self.initial = State(initial)
        self.final = {State(e) for e in final}

        assert isinstance(self.transition.states, set)
        assert isinstance(self.transition.alphabet, set)
        assert isinstance(self.final, set) and self.final.issubset(self.transition.states)
        assert not isinstance(self.initial, set) and self.initial in self.transition.states

    def __str__(self):
        string = str(self.transition)
        string += "\ninitial = " + str(self.initial) + " ; final = " + str(self.final)
        return string

    def __repr__(self):
        return self.__str__()

    def _read(self, string, state):
        if not len(string) and state in self.final:
            return True
        else:
            for e, rmn in self.transition(state, string):
                if self._read(rmn, e):
                    return True

        return False

    def read(self, string):
        return self._read(string, self.initial)

    def build_dot(self):
        dot = Digraph()
        dot.attr(rankdir='LR', size='8,5')
        dot.node("hidden", style="invisible")

        if self.initial in self.final:
            dot.node(str(self.initial), root="true", shape="doublecircle")
        else:
            dot.node(str(self.initial), root="true")

        dot.edge("hidden", str(self.initial))

        for state in self.transition.states:
            if state in self.final:
                dot.node(str(state), shape="doublecircle")
            else:
                dot.node(str(state))

        for ei in self.transition.delta:
            for symbol in self.transition.delta[ei]:
                for ef in self.transition.delta[ei][symbol]:
                    dot.edge(str(ei), str(ef), label=str(symbol))

        return dot
