from graphviz import Digraph


class State:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Transition:
    def __init__(self):
        self.states = set()
        self.alphabet = set()
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
