from networkx.drawing.nx_agraph import to_agraph

from pyauto.delta import *

import networkx
import math
import copy


class FDATransition(Transition):
    def __init__(self, fda, **kwargs):
        super().__init__(**kwargs)
        self.fda = fda

    def __call__(self, tape):
        pointer = tape.pointer()

        buffer = tape.copy()
        buffer.states = [self.fda.initial]
        buffer.pointers = [pointer]

        parsed = self.fda(buffer)
        if parsed.error and parsed.state() in self.fda.final:
            tape.read(self.target, parsed.pointer() - pointer)
        else:
            tape.error = True

        return tape

    def symbol(self):
        return "FDA" + str(hex(id(self)))


class FADelta(Delta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_transition(self, source, target, symbols):
        if source not in self.transitions:
            self.transitions[source] = []

        transition_symbols = []
        for s in symbols:
            if s is NullTransition.SYMBOL:
                transition = NullTransition(source=source, target=target)
                self.transitions[source].append(transition)

            elif isinstance(s, str) and len(s) == 1:
                transition = CharTransition(character=s, source=source, target=target)
                self.transitions[source].append(transition)

            elif isinstance(s, FiniteAutomata):
                transition = FDATransition(fda=s.minimal(), source=source, target=target)
                self.transitions[source].append(transition)

            else:
                raise RuntimeError("can't interpret symbol", s)

            transition_symbols.append(transition.symbol())
            self.alphabet.add(transition.symbol())

        return transition_symbols


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
        if self.has_null_transitions():
            for each in self.g.nodes:
                self._build_epsilon_closure(initial=each, node=each)

        self.state_power_set = {}
        if self.is_non_deterministic() and not self.has_null_transitions():
            self._build_power_state_set({self.initial})

        self.minimization_steps = {}

    def __str__(self):
        string = str(self.transition)
        string += "\ninitial = " + str(self.initial) + " ; final = " + str(self.final)
        return string

    def __repr__(self):
        return self.__str__()

    def __add__(self, p):
        this = self.rebase(1)
        other = p.rebase(this.transition.max_state().number + 1)

        transition = FADelta()
        transition.join(this.transition)
        transition.join(other.transition)

        transition.add(self.initial, this.initial, NullTransition.SYMBOL)
        transition.add(self.initial, other.initial, NullTransition.SYMBOL)

        final = self.initial.prefix + str(other.transition.max_state().number + 1)
        for fe in this.final:
            transition.add(fe, final, NullTransition.SYMBOL)
        for fe in other.final:
            transition.add(fe, final, NullTransition.SYMBOL)

        return FiniteAutomata(transition, self.initial, {final})

    def __invert__(self):
        this = self.rebase(1)
        final = self.initial.prefix + str(this.transition.max_state().number + 1)

        transition = FADelta()
        transition.join(this.transition)
        transition.add(self.initial, this.initial, NullTransition.SYMBOL)
        transition.add(self.initial, final, NullTransition.SYMBOL)

        for fe in this.final:
            transition.add(fe, final, NullTransition.SYMBOL)
            if fe != this.initial:
                transition.add(fe, this.initial, NullTransition.SYMBOL)

        return FiniteAutomata(transition, self.initial, {final})

    def __or__(self, p):
        other = p.rebase(self.transition.max_state().number + 1)

        transition = FADelta()
        transition.join(self.transition)
        transition.join(other.transition)

        for fe in self.final:
            transition.add(fe, other.initial, NullTransition.SYMBOL)

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

    def _build_epsilon_closure(self, initial, node):
        if initial not in self.epsilon_closure:
            self.epsilon_closure[initial] = {initial}
        else:
            self.epsilon_closure[initial].add(node)

        for edge in self.g.out_edges(node):
            if NullTransition.SYMBOL in self._get_symbol_from_edge(self.g, edge):
                self.epsilon_closure[initial].add(node)
                if edge[1] not in self.epsilon_closure[initial]:
                    self._build_epsilon_closure(initial, edge[1])

    def _build_power_state_set(self, state_set):
        assert not self.has_null_transitions()
        tuple_state = tuple(sorted(state_set))
        self.state_power_set[tuple_state] = {}

        for state in state_set:
            for edge in self.g.out_edges(state):
                for symbol in self._get_symbol_from_edge(self.g, edge):
                    if symbol not in self.state_power_set[tuple_state]:
                        self.state_power_set[tuple_state][symbol] = {edge[1]}
                    else:
                        self.state_power_set[tuple_state][symbol].add(edge[1])

        for s in self.state_power_set[tuple_state]:
            added_sets = tuple(sorted(self.state_power_set[tuple_state][s]))
            if len(added_sets) and added_sets not in self.state_power_set:
                self._build_power_state_set(self.state_power_set[tuple_state][s])

    def _partition_by_symbol(self, pi, pi_current, symbol, delta):
        groups = {}

        for state in pi:
            if state in self.transition.delta and symbol in self.transition.delta[state]:
                e = self.transition.delta[state][symbol]
                assert len(e) == 1
                index = list(map(lambda g: e.issubset(g), pi_current)).index(True)
            else:
                index = math.inf

            if index not in groups:
                groups[index] = {state}
            else:
                groups[index].add(state)

            if symbol not in delta:
                delta[symbol] = {state: index}
            else:
                delta[symbol][state] = index

        return list(groups.values())

    def minimal(self):
        if self.has_null_transitions():
            return self.remove_null_transitions().get_deterministic_automata().minimize_automata()
        elif self.is_non_deterministic():
            return self.get_deterministic_automata().minimize_automata()
        return self.minimize_automata()

    def minimize_automata(self):
        assert not self.has_null_transitions()

        g = self.g.copy()
        for state in filter(lambda s: s not in self.final, list(g.nodes)):
            reach_final = False
            for final in self.final:
                try:
                    networkx.dijkstra_path(g, state, final)
                    reach_final = True
                except networkx.exception.NetworkXNoPath:
                    continue

            if not reach_final:
                g.remove_node(state)

        pi_next, pi_current, pi_global = [{e for e in g.nodes if e not in self.final}, self.final], [], []

        step_number = 0
        while not(set(sorted([tuple(sorted(p)) for p in pi_global])) ==
                  set(sorted([tuple(sorted(p)) for p in pi_next]))):

            pi_global = copy.deepcopy(pi_next)
            self.minimization_steps[step_number] = {"pi": pi_next, "delta": {}}

            for symbol in filter(lambda s: s != NullTransition.SYMBOL, self.transition.alphabet):
                pi_current, pi_next = pi_next, []

                for pi in pi_current:
                    pi_next += self._partition_by_symbol(pi, pi_current, symbol,
                                                         self.minimization_steps[step_number]["delta"])

            step_number += 1

        last_step = max(self.minimization_steps.keys())
        pi_final = self.minimization_steps[last_step]["pi"]

        states_map, final, initial = {}, [], None
        prefix = chr((ord(self.initial.prefix[0]) - ord('a') - 1) % (ord('z') - ord('a') + 1) + ord('a'))

        delta = {}
        for i, state in enumerate(pi_final):
            tuple_state = tuple(sorted(state))
            states_map[tuple_state] = prefix + str(i)
            assert state.issubset(self.final) or not len(state.intersection(self.final))

            if state.issubset(self.final):
                final.append(states_map[tuple_state])

            if self.initial in state:
                initial = states_map[tuple_state]

            symbols = self.minimization_steps[last_step]["delta"]
            delta[tuple_state] = {s: [] for s in symbols}
            for symbol in symbols:
                for e in filter(lambda t: t in state, self.minimization_steps[last_step]["delta"][symbol]):
                    delta[tuple_state][symbol].append(self.minimization_steps[last_step]["delta"][symbol][e])

        assert initial

        transition = FADelta()
        for state in delta:
            for symbol in delta[state]:
                group = set(delta[state][symbol])
                assert len(group) == 1

                g = group.pop()
                if g is not math.inf:
                    tuple_state = tuple(sorted(pi_final[g]))
                    transition.add(states_map[state], states_map[tuple_state], symbol)

        return FiniteAutomata(transition, initial, final)

    def get_deterministic_automata(self):
        assert not self.has_null_transitions()

        if not self.is_non_deterministic():
            return FiniteAutomata(self.transition, self.initial, self.final)

        assert len(self.state_power_set) > 0

        states_map = {}
        prefix = chr((ord(self.initial.prefix[0]) - ord('a') - 1) % (ord('z') - ord('a') + 1) + ord('a'))
        for i, each in enumerate(self.state_power_set):
            states_map[each] = prefix + str(i)

        transition, final = FADelta(), set()
        for each in self.state_power_set:
            for symbol in self.state_power_set[each]:
                states = self.state_power_set[each][symbol]

                tuple_state = tuple(sorted(states))
                transition.add(states_map[each], states_map[tuple_state], symbol)

            tuple_current_state = tuple(sorted(each))
            if len(self.final.intersection(each)):
                final.add(states_map[tuple_current_state])

        return FiniteAutomata(transition, states_map[tuple({self.initial})], final)

    def is_non_deterministic(self):
        is_ndfa = False

        for node in self.g.nodes:
            out_symbol = []
            for edge in self.g.out_edges(node):
                out_symbol += self._get_symbol_from_edge(self.g, edge)

            if len(out_symbol) != len(set(out_symbol)):
                is_ndfa = True

        return is_ndfa

    def remove_null_transitions(self):
        assert self.has_null_transitions()

        g = networkx.DiGraph()
        final = [state for state in self.epsilon_closure
                 if len(set(self.epsilon_closure[state]).intersection(self.final))]

        for f in final:
            g.add_node(f)

        for p in self.epsilon_closure:
            for eclose in filter(lambda state: state in self.transition.delta, self.epsilon_closure[p]):
                for q in self.epsilon_closure:
                    for symbol in filter(lambda s: s != NullTransition.SYMBOL, self.transition.delta[eclose]):
                        if q in self.transition.delta[eclose][symbol]:
                            self._add_edge_to_graph(g, p, q, symbol)

        for state in list(g.nodes):
            try:
                networkx.dijkstra_path(g, self.initial, state)
            except networkx.exception.NetworkXNoPath:
                g.remove_node(state)
                if state in final:
                    final.remove(state)

        transition = FADelta()
        for edge in g.edges:
            for symbol in self._get_symbol_from_edge(g, edge):
                transition.add(edge[0], edge[1], symbol)

        return FiniteAutomata(transition, self.initial, final)

    def has_null_transitions(self):
        for edge in self.g.edges:
            if NullTransition.SYMBOL in self._get_symbol_from_edge(self.g, edge):
                return True

        return False

    def __call__(self, tape):
        if not len(tape.head()) and self.initial in self.final:
            done_buffer = tape.copy()
            return done_buffer

        buffers, final_buffer = [tape], None
        while not all(map(lambda b: not len(b.head()) and b.error, buffers)):
            parsed = set()
            for buffer in filter(lambda b: not b.error, buffers):
                parsed.update(self.transition(buffer))

            if len(parsed):
                buffers = list(parsed)

                consumed_in_final = list(map(lambda b: not len(b.head()) and b.state() in self.final, buffers))
                if any(consumed_in_final):
                    final_buffer = buffers[consumed_in_final.index(True)]

            else:
                consumed_in_final = list(map(lambda b: b.state() in self.final, buffers))

                if any(consumed_in_final):
                    final_buffer = buffers[consumed_in_final.index(True)]

                break

        if final_buffer:
            return final_buffer
        else:
            done_buffer = tape.copy()
            done_buffer.error = True

            return done_buffer

    def read(self, string):
        buffer = self(Buffer(data=string, initial=self.initial))
        return buffer.state() in self.final and not len(buffer.head())

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

        if self.has_null_transitions():
            important_nodes, important_to_final_nodes = [], []
            for each in self.g.nodes:
                in_symbol = []
                for edge in self.g.in_edges(node):
                    in_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != NullTransition.SYMBOL]

                out_symbol = []
                for edge in self.g.out_edges(node):
                    out_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != NullTransition.SYMBOL]

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
        transition = FADelta()

        state = 0
        for i, z in enumerate(expression):
            transition.add("z" + str(state), "z" + str(state + 1), z)
            state += 1
            if i < len(expression) - 1:
                transition.add("z" + str(state), "z" + str(state + 1), NullTransition.SYMBOL)
                state += 1

        initial = "z0"
        final = {"z" + str(state)}

        super().__init__(transition, initial, final)
