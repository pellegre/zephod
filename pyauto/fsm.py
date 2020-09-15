from graphviz import Digraph
from enum import Enum
from networkx.drawing.nx_agraph import to_agraph

import networkx
import re


class State:
    def __init__(self, state):
        if isinstance(state, str):
            self.name = state
            groups = re.search("([a-zA-Z]+)([0-9]+)", self.name).groups()
            self.prefix = groups[0]
            self.number = int(groups[1])
        else:
            self.name = state.name
            self.prefix = state.prefix
            self.number = state.number

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Transition:
    EPSILON = "$"

    def __init__(self):
        self.states = set()
        self.alphabet = {Transition.EPSILON}
        self.delta = {}

    def add(self, ei, ef, symbols):
        assertion = [isinstance(s, str) for s in symbols]
        assert sum(assertion) == len(assertion)

        initial_state, final_state = State(ei), State(ef)

        self.states.add(initial_state)
        self.states.add(final_state)
        self.alphabet.update(symbols)

        if initial_state not in self.delta:
            self.delta[initial_state] = {}

        for s in symbols:
            if s not in self.delta[initial_state]:
                self.delta[initial_state][s] = set()

            self.delta[initial_state][s].add(final_state)

    def join(self, other):
        for ei in other.delta:
            for symbol in other.delta[ei]:
                for ef in other.delta[ei][symbol]:
                    self.add(ei, ef, symbol)

    def __call__(self, state, string):
        assert state in self.states
        state_value = state
        if isinstance(state_value, str):
            state_value = State(state_value)

        next_states = set()
        if state_value in self.delta:
            if len(string):
                symbol = string[0]
                assert symbol in self.alphabet

                if symbol in self.delta[state_value]:
                    next_states.update({(e, string[1:]) for e in self.delta[state_value][symbol]})

            if Transition.EPSILON in self.delta[state_value]:
                next_states.update({(e, string[:]) for e in self.delta[state_value][Transition.EPSILON]})

        return next_states

    def __str__(self):
        string = "states = " + str(self.states) + " ; alphabet = " + str(self.alphabet) + "\n"
        for e in self.delta:
            for s in self.delta[e]:
                string += str(e) + " (" + str(s) + ") -> " + str(self.delta[e][s]) + "\n"
        return string[:-1]

    def __repr__(self):
        return self.__str__()

    def max_state(self):
        return max(self.states, key=lambda e: e.number)

    def rebase(self, base):
        transition = Transition()

        for ei in self.delta:
            new_ei = ei.prefix + str(ei.number + base)
            for symbol in self.delta[ei]:
                for ef in self.delta[ei][symbol]:
                    new_ef = ef.prefix + str(ef.number + base)
                    transition.add(new_ei, new_ef, symbol)

        return transition


class FiniteAutomata:
    class NodeType:
        INITIAL = "initial"
        FINAL = "final"
        NONE = "none"

    def __init__(self, transition, initial, final):
        self.transition = transition
        self.initial = initial
        if isinstance(self.initial, str):
            self.initial = State(self.initial)
        self.final = {State(e) if isinstance(e, str) else e for e in final}

        assert isinstance(self.transition.states, set)
        assert isinstance(self.transition.alphabet, set)
        assert isinstance(self.final, set) and self.final.issubset(self.transition.states)
        assert not isinstance(self.initial, set) and self.initial in self.transition.states

        self.g = self.build_graph()

    def __str__(self):
        string = str(self.transition)
        string += "\ninitial = " + str(self.initial) + " ; final = " + str(self.final)
        return string

    def __repr__(self):
        return self.__str__()

    def __add__(self, p):
        this = self.rebase(1)
        other = p.rebase(this.transition.max_state().number + 1)

        transition = Transition()
        transition.join(this.transition)
        transition.join(other.transition)

        transition.add(self.initial, this.initial, Transition.EPSILON)
        transition.add(self.initial, other.initial, Transition.EPSILON)

        final = self.initial.prefix + str(other.transition.max_state().number + 1)
        for fe in this.final:
            transition.add(fe, final, Transition.EPSILON)
        for fe in other.final:
            transition.add(fe, final, Transition.EPSILON)

        return FiniteAutomata(transition, self.initial, {final})

    def __invert__(self):
        this = self.rebase(1)
        final = self.initial.prefix + str(this.transition.max_state().number + 1)

        transition = Transition()
        transition.join(this.transition)

        transition.add(self.initial, this.initial, Transition.EPSILON)
        transition.add(self.initial, final, Transition.EPSILON)

        for fe in this.final:
            transition.add(fe, final, Transition.EPSILON)
            transition.add(fe, this.initial, Transition.EPSILON)

        return FiniteAutomata(transition, self.initial, {final})

    def __or__(self, p):
        other = p.rebase(self.transition.max_state().number + 1)

        transition = Transition()
        transition.join(self.transition)
        transition.join(other.transition)

        for fe in self.final:
            transition.add(fe, other.initial, Transition.EPSILON)

        return FiniteAutomata(transition, self.initial, other.final)

    def parse(self, string, state):
        if not len(string) and state in self.final:
            return True
        else:
            for e, rest in self.transition(state, string):
                if self.parse(rest, e):
                    return True

        return False

    def read(self, string):
        return self.parse(string, self.initial)

    def rebase(self, base):
        initial = self.initial.prefix + str(self.initial.number + base)
        final = {e.prefix + str(e.number + base) for e in self.final}
        return FiniteAutomata(self.transition.rebase(base), initial, final)

    def build_graph(self):
        g = networkx.DiGraph()
        g.add_node(self.initial, type=FiniteAutomata.NodeType.INITIAL)

        for each in self.transition.states:
            if each in self.final:
                g.add_node(each, type=FiniteAutomata.NodeType.FINAL)
            else:
                g.add_node(each, type=FiniteAutomata.NodeType.NONE)

        for ei in self.transition.delta:
            for symbol in self.transition.delta[ei]:
                for ef in self.transition.delta[ei][symbol]:
                    g.add_edge(ei, ef, symbol=symbol)

        return g

    @staticmethod
    def get_colors():
        return ["lightskyblue4", "green4", "indianred4", "lavenderblush4", "olivedrab4"
                  "purple4", "steelblue4", "hotpink4",
                  "orangered4", "turquoise4"]

    def build_dot(self):
        colors = self.get_colors()

        a = to_agraph(self.g)
        a.graph_attr["rankdir"] = "LR"
        a.graph_attr["size"] = 8.5
        a.node_attr["fontsize"] = 7
        a.edge_attr["fontsize"] = 8
        a.edge_attr["penwidth"] = 0.4
        a.edge_attr["arrowsize"] = 0.5

        symbol_colors = {}
        for each in self.transition.delta:
            for symbol in sorted(set(self.transition.delta[each])):
                if symbol not in symbol_colors:
                    symbol_colors[symbol] = colors.pop(0)

                    if not len(colors):
                        colors = self.get_colors()

                a.add_node(symbol, color=symbol_colors[symbol])

        initial = a.get_node(str(self.initial))
        initial.attr["root"] = "true"
        if self.initial in self.final:
            initial.attr["shape"] = "doublecircle"

        for state in filter(lambda n: n in self.final, self.transition.states):
            node = a.get_node(str(state))
            node.attr["shape"] = "doublecircle"

        a.add_node("hidden", style="invisible")
        a.add_edge("hidden", str(self.initial))

        a.layout("dot")

        for ei in self.transition.delta:
            for symbol in self.transition.delta[ei]:
                for ef in self.transition.delta[ei][symbol]:
                    edge = a.get_edge(str(ei), str(ef))
                    edge.attr["xlabel"] = str(symbol)
                    edge.attr["fontcolor"] = symbol_colors[symbol]
                    edge.attr["color"] = symbol_colors[symbol]

        return a


class Z(FiniteAutomata):
    def __init__(self, expression):
        transition = Transition()

        state = 0
        for i, z in enumerate(expression):
            transition.add("z" + str(state), "z" + str(state + 1), z)
            state += 1
            if i < len(expression) - 1:
                transition.add("z" + str(state), "z" + str(state + 1), Transition.EPSILON)
                state += 1

        initial = "z0"
        final = {"z" + str(state)}

        super().__init__(transition, initial, final)
