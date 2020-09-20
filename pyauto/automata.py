from pyauto.delta import *
from networkx.drawing.nx_agraph import to_agraph

import networkx


class NullTransition(Transition):
    SYMBOL = "$"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _consume(self, tape):
        tape.read(self.target, 0)

    def symbol(self):
        return NullTransition.SYMBOL


class Automata:
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
                else:
                    final_buffer = max(buffers, key=lambda b: b.pointer())

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

    def build_graph(self):
        g = networkx.DiGraph()
        g.add_node(self.initial, type=Automata.NodeType.INITIAL)

        for each in self.transition.states:
            if each in self.final:
                g.add_node(each, type=Automata.NodeType.FINAL)
            else:
                g.add_node(each, type=Automata.NodeType.NONE)

        for ei in self.transition.delta:
            for symbol in self.transition.delta[ei]:
                for ef in self.transition.delta[ei][symbol]:
                    self._add_edge_to_graph(g, ei, ef, symbol)

        return g

    def get_colors(self):
        return ["black"]

    @staticmethod
    def get_latex_node(state):
        if "_" not in state.prefix:
            return str(state.prefix) + "_{" + str(state.number) + "}"
        else:
            tokens = state.prefix.split("_")
            state = state.prefix[0]
            return str(state) + "_{" + str(tokens[1]) + "}^{" + tokens[2] + "}"

    def build_dot(self, labels=True, tex=False, layout="dot"):
        colors = self.get_colors()
        null_color = "gray"

        a = to_agraph(self.g)
        a.graph_attr["rankdir"] = "LR"
        a.graph_attr["size"] = 8.5
        a.node_attr["fontsize"] = 7
        a.edge_attr["fontsize"] = 8
        a.edge_attr["penwidth"] = 0.4
        a.edge_attr["arrowsize"] = 0.5

        if tex:
            a.node_attr["texmode"] = "math"
            a.edge_attr["texmode"] = "math"

        symbol_colors = {}
        for symbol in sorted(list(self.transition.alphabet)):
            if symbol not in symbol_colors and NullTransition.SYMBOL not in symbol:
                symbol_colors[symbol] = colors.pop(0)

                if not len(colors):
                    colors = self.get_colors()
            else:
                symbol_colors[symbol] = null_color

            if labels:
                a.add_node(symbol, color=symbol_colors[symbol])
                if NullTransition.SYMBOL in symbol and tex:
                    node = a.get_node(symbol)
                    node.attr["texlbl"] = symbol.replace(NullTransition.SYMBOL, "$\epsilon$")

        initial = a.get_node(str(self.initial))
        initial.attr["root"] = "true"

        if tex:
            initial.attr["label"] = self.get_latex_node(self.initial)

        if self.initial in self.final:
            initial.attr["shape"] = "doublecircle"

        for state in self.transition.states:
            node = a.get_node(str(state))
            if state in self.final:
                node.attr["shape"] = "doublecircle"
                if tex:
                    node.attr["label"] = self.get_latex_node(state)
            else:
                if tex:
                    node.attr["label"] = self.get_latex_node(state)

        if self.has_null_transitions():
            important_nodes, important_to_final_nodes = [], []
            for each in self.g.nodes:

                in_symbol = []
                for edge in self.g.in_edges(each):
                    in_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != NullTransition.SYMBOL]

                out_symbol = []
                for edge in self.g.out_edges(each):
                    out_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != NullTransition.SYMBOL]

                if len(in_symbol) >= 1 and not len(out_symbol):
                    important_nodes.append(each)

                    for neighbor in self.g.neighbors(each):
                        if neighbor in self.final:
                            important_to_final_nodes.append(each)

            if len(important_nodes) > 0:
                important_nodes.append(self.initial)

            for state in important_nodes:
                each = a.get_node(str(state))
                each.attr["fillcolor"] = "gray"
                each.attr["style"] = "filled"

            for state in important_to_final_nodes:
                each = a.get_node(str(state))
                each.attr["fillcolor"] = "orange"
                each.attr["style"] = "filled"

        a.add_node("hidden", style="invisible")
        a.add_edge("hidden", str(self.initial))

        a.layout(layout)

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
                if NullTransition.SYMBOL in symbol and tex:
                    edge.attr["texlbl"] = symbol.replace(NullTransition.SYMBOL, "$\epsilon$")
                    edge.attr["label"] = "  "
                else:
                    edge.attr["label"] = symbol
                if symbol not in symbol_colors and "," in symbol:
                    edge.attr["fontcolor"] = symbol_colors[symbol.split(",")[0]]
                    edge.attr["color"] = symbol_colors[symbol.split(",")[0]]
                else:
                    edge.attr["fontcolor"] = symbol_colors[symbol]
                    edge.attr["color"] = symbol_colors[symbol]

        return a