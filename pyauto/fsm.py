from graphviz import Digraph
from enum import Enum
from networkx.drawing.nx_agraph import to_agraph

import networkx
import re


class State:
    def __init__(self, state):
        if isinstance(state, str):
            groups = re.search("([a-zA-Z]+)([0-9]+)", state).groups()
            self.prefix = groups[0]
            self.number = int(groups[1])
        else:
            self.prefix = state.prefix
            self.number = state.number

        # self.name = "$" + self.prefix + "_" + str(self.number) + "$"
        self.name = self.prefix + str(self.number)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __lt__(self, other):
        assert isinstance(other, State)
        return self.number < other.number

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

        self.epsilon_closure = {}
        if self.has_epsilon():
            for each in self.g.nodes:
                self.build_epsilon_closure(initial=each, node=each)

        self.state_power_set = {}
        if self.is_non_deterministic() and not self.has_epsilon():
            self.build_power_state_set({self.initial})

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
            if fe != this.initial:
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

    @staticmethod
    def _get_symbol_from_edge(g, edge):
        return g.get_edge_data(edge[0], edge[1])["symbol"]

    @staticmethod
    def _add_edge_to_graph(g, ei, ef, symbol):
        edge = g.get_edge_data(ei, ef)
        if edge is not None:
            edge["symbol"].append(symbol)
        else:
            g.add_edge(ei, ef, symbol=[symbol])

    def build_power_state_set(self, state_set):
        assert not self.has_epsilon()
        tuple_state = tuple(sorted(state_set))

        self.state_power_set[tuple_state] = {s: set() for s in self.transition.alphabet if s != Transition.EPSILON}

        for state in state_set:
            for edge in self.g.out_edges(state):
                for symbol in self._get_symbol_from_edge(self.g, edge):
                    self.state_power_set[tuple_state][symbol].add(edge[1])

        for s in self.state_power_set[tuple_state]:
            added_sets = tuple(sorted(self.state_power_set[tuple_state][s]))
            if len(added_sets) and added_sets not in self.state_power_set:
                self.build_power_state_set(self.state_power_set[tuple_state][s])

    def is_non_deterministic(self):
        is_ndfa = False

        for node in self.g.nodes:
            out_symbol = []
            for edge in self.g.out_edges(node):
                out_symbol += self._get_symbol_from_edge(self.g, edge)

            if len(out_symbol) != len(set(out_symbol)):
                is_ndfa = True

        return is_ndfa

    def build_epsilon_closure(self, initial, node):
        if initial not in self.epsilon_closure:
            self.epsilon_closure[initial] = {initial}
        else:
            self.epsilon_closure[initial].add(node)

        for edge in self.g.out_edges(node):
            if Transition.EPSILON in self._get_symbol_from_edge(self.g, edge):
                self.epsilon_closure[initial].add(node)
                if edge[1] not in self.epsilon_closure[initial]:
                    self.build_epsilon_closure(initial, edge[1])

    def strip_epsilon(self):
        g = networkx.DiGraph()
        final = [state for state in self.epsilon_closure
                 if len(set(self.epsilon_closure[state]).intersection(self.final))]

        for f in final:
            g.add_node(f)

        for p in self.epsilon_closure:
            for eclose in filter(lambda state: state in self.transition.delta, self.epsilon_closure[p]):
                for q in self.epsilon_closure:
                    for symbol in filter(lambda s: s != Transition.EPSILON, self.transition.delta[eclose]):
                        if q in self.transition.delta[eclose][symbol]:
                            self._add_edge_to_graph(g, p, q, symbol)

        for state in list(g.nodes):
            try:
                networkx.dijkstra_path(g, self.initial, state)
            except networkx.exception.NetworkXNoPath:
                g.remove_node(state)
                if state in final:
                    final.remove(state)

        transition = Transition()
        for edge in g.edges:
            for symbol in self._get_symbol_from_edge(g, edge):
                transition.add(edge[0], edge[1], symbol)

        return FiniteAutomata(transition, self.initial, final)

    def has_epsilon(self):
        for edge in self.g.edges:
            if Transition.EPSILON in self._get_symbol_from_edge(self.g, edge):
                return True

        return False

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
                    self._add_edge_to_graph(g, ei, ef, symbol)

        return g

    @staticmethod
    def get_colors():
        return ["lightskyblue4", "green4", "indianred4", "lavenderblush4", "olivedrab4",
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

        if self.has_epsilon():
            important_nodes, important_to_final_nodes = [], []
            for each in self.g.nodes:
                in_symbol = []
                for edge in self.g.in_edges(node):
                    in_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != Transition.EPSILON]

                out_symbol = []
                for edge in self.g.out_edges(node):
                    out_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != Transition.EPSILON]

                if len(in_symbol) >= 1 and not len(out_symbol):
                    important_nodes.append(each)

                    for neighbor in self.g.neighbors(each):
                        if neighbor in self.final:
                            important_to_final_nodes.append(each)

            if len(important_nodes) > 0:
                important_nodes.append(self.initial)

            if len(important_nodes) > 0:
                a.add_node("important", fillcolor="darksalmon", style="filled")

            for state in important_nodes:
                node = a.get_node(str(state))
                node.attr["fillcolor"] = "darksalmon"
                node.attr["style"] = "filled"

            if len(important_to_final_nodes) > 0:
                a.add_node("important\n(to final)", fillcolor="tomato", style="filled")

            for state in important_to_final_nodes:
                node = a.get_node(str(state))
                node.attr["fillcolor"] = "tomato"
                node.attr["style"] = "filled"

        a.add_node("hidden", style="invisible")
        a.add_edge("hidden", str(self.initial))

        a.layout("dot")

        edges = {ei: {} for ei in self.transition.delta}
        for ei in self.transition.delta:
            for symbol in self.transition.delta[ei]:
                for ef in self.transition.delta[ei][symbol]:
                    if ef not in edges[ei]:
                        edges[ei][ef] = str(symbol)
                    else:
                        edges[ei][ef] = edges[ei][ef] + "," + str(symbol)

        for ei in edges:
            for ef in edges[ei]:
                symbol = edges[ei][ef]

                edge = a.get_edge(str(ei), str(ef))
                edge.attr["label"] = symbol
                if symbol not in symbol_colors and "," in symbol:
                    edge.attr["fontcolor"] = symbol_colors[symbol.split(",")[0]]
                    edge.attr["color"] = symbol_colors[symbol.split(",")[0]]
                else:
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
