from zephod.pushdown import *

import networkx
import math
import copy
import shutil
import functools


# --------------------------------------------------------------------
#
# finite automata transitions
#
# --------------------------------------------------------------------

class FANullTransition(Transition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, action=FANullReadAction())

    def symbol(self):
        return Transition.NULL


class FAReadTransition(Transition):
    def __init__(self, character, **kwargs):
        super().__init__(**kwargs, action=FAReadAction(on_symbol=character))
        self.character = character

    def symbol(self):
        return self.character


class RegexTapeAction(TapeAction):
    def __init__(self, fda):
        super().__init__(on_symbol=None)
        self.fda = fda

    def action(self, tape: Tape):
        pointer = tape.pointer()

        parsed = self.fda(Buffer(data=tape.data(), initial=self.fda.initial, pointer=pointer))

        if parsed.state() in self.fda.final:
            delta = parsed.pointer() - pointer

            if not delta:
                tape.none()

            for move in range(delta):
                tape.right()

    def __str__(self):
        if self.fda.regex:
            return self.fda.regex
        else:
            return str(hex(id(self)))


class FARegexAction(InputAction):
    def __init__(self, fda):
        self.fda = fda

        super().__init__(actions={
            Tape.N(0): [RegexTapeAction(fda=self.fda)]
        })

    def __str__(self):
        if self.fda.regex:
            return self.fda.regex
        else:
            return str(hex(id(self)))


class FDATransition(Transition):
    def __init__(self, fda, **kwargs):
        super().__init__(**kwargs, action=FARegexAction(fda=fda))
        self.fda = fda

    def symbol(self):
        if self.fda.regex:
            return self.fda.regex
        else:
            return str(hex(id(self)))


# --------------------------------------------------------------------
#
# FA delta function
#
# --------------------------------------------------------------------


class FADelta(Delta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_transition(self, source, target, symbols):
        if source not in self.transitions:
            self.transitions[source] = []

        transition_symbols = []
        for s in symbols:
            if s is Transition.NULL:
                transition = FANullTransition(source=source, target=target)

            elif isinstance(s, str) and len(s) == 1:
                transition = FAReadTransition(character=s, source=source, target=target)

            elif isinstance(s, FiniteAutomata):
                transition = FDATransition(fda=s.minimal(), source=source, target=target)

            else:
                raise RuntimeError("can't interpret symbol", s)

            if source in self.transitions:
                if any([transition == t for t in self.transitions[transition.source]]):
                    print("[+] skipping repeated transition", transition)
                    return None

            self.transitions[source].append(transition)

            transition_symbols.append(transition.symbol())
            self.alphabet.add(transition.symbol())

        return transition_symbols


class FiniteAutomata(Automata):
    def __init__(self, transition, initial, final):
        assert isinstance(transition, FADelta)
        super().__init__(transition=transition, initial=initial, final=final)

        self.regex = None

        self.epsilon_closure = {}
        if self.has_null_transitions():
            for each in self.g.nodes:
                self._build_epsilon_closure(initial=each, node=each)

        self.state_power_set = {}
        if self.is_non_deterministic() and not self.has_null_transitions():
            self._build_power_state_set({self.initial})

        self.minimization_steps = {}

    def __add__(self, p):
        this = self.rebase(1)
        other = p.rebase(this.transition.max_state().number + 1)

        transition = FADelta()
        transition.join(this.transition)
        transition.join(other.transition)

        transition.add(self.initial, this.initial, Transition.NULL)
        transition.add(self.initial, other.initial, Transition.NULL)

        final = self.initial.prefix + str(other.transition.max_state().number + 1)
        for fe in this.final:
            transition.add(fe, final, Transition.NULL)
        for fe in other.final:
            transition.add(fe, final, Transition.NULL)

        fda = FiniteAutomata(transition, self.initial, {final})
        fda.regex = "(" + str(self.regex) + " + " + str(p.regex) + ")"

        return fda

    def __invert__(self):
        this = self.rebase(1)
        final = self.initial.prefix + str(this.transition.max_state().number + 1)

        transition = FADelta()
        transition.join(this.transition)
        transition.add(self.initial, this.initial, Transition.NULL)
        transition.add(self.initial, final, Transition.NULL)

        for fe in this.final:
            transition.add(fe, final, Transition.NULL)
            if fe != this.initial:
                transition.add(fe, this.initial, Transition.NULL)

        fda = FiniteAutomata(transition, self.initial, {final})

        if self.regex[0] != "(" and len(self.regex) > 1:
            fda.regex = "(" + self.regex + ")*"
        else:
            fda.regex = self.regex + "*"

        return fda

    def __or__(self, p):
        other = p.rebase(self.transition.max_state().number + 1)

        transition = FADelta()
        transition.join(self.transition)
        transition.join(other.transition)

        for fe in self.final:
            transition.add(fe, other.initial, Transition.NULL)

        fda = FiniteAutomata(transition, self.initial, other.final)
        fda.regex = str(self.regex) + str(p.regex)

        return fda

    def _build_a_graph(self, a):
        if self.has_null_transitions():
            important_nodes, important_to_final_nodes = [], []
            for each in self.g.nodes:

                in_symbol = []
                for edge in self.g.in_edges(each):
                    in_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != Transition.NULL]

                out_symbol = []
                for edge in self.g.out_edges(each):
                    out_symbol += [s for s in self._get_symbol_from_edge(self.g, edge) if s != Transition.NULL]

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

    def _build_epsilon_closure(self, initial, node):
        if initial not in self.epsilon_closure:
            self.epsilon_closure[initial] = {initial}
        else:
            self.epsilon_closure[initial].add(node)

        for edge in self.g.out_edges(node):
            if Transition.NULL in self._get_symbol_from_edge(self.g, edge):
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

    def minimal(self):
        if self.has_null_transitions():
            return self.remove_null_transitions().get_deterministic().strip_redundant().minimize()
        elif self.is_non_deterministic():
            return self.get_deterministic().strip_redundant().minimize()
        return self.strip_redundant().minimize()

    def strip_redundant(self):
        states_to_remove = set()

        for state in filter(lambda s: s not in self.final, self.g.nodes):
            reach_final = False
            for final in self.final:
                if networkx.has_path(self.g, state, final):
                    reach_final = True

            if not reach_final:
                states_to_remove.add(state)

        for state in self.g:
            if not networkx.has_path(self.g, self.initial, state):
                states_to_remove.add(state)

        final = self.final.copy()
        final = final.difference(states_to_remove)

        transition = FADelta()
        for state in self.transition.delta:

            if state not in states_to_remove:

                for symbol in self.transition.delta[state]:
                    target = self.transition.delta[state][symbol]

                    for each in target:
                        if each not in states_to_remove:
                            transition.add(state, each, symbol)

        fda = FiniteAutomata(transition, self.initial, final)
        fda.regex = self.regex

        return fda

    def minimize(self):
        assert not self.has_null_transitions() and not self.is_non_deterministic()

        p = {frozenset(self.transition.states.difference(self.final)), frozenset(self.final)}
        w = p.copy()

        while len(w):
            partition = w.pop()

            for a in self.transition.alphabet:
                x = frozenset({s for s in self.transition.states
                               if s in self.transition.delta and a in self.transition.delta[s]
                               and not len(self.transition.delta[s][a].intersection(partition))})

                for y in p.copy():
                    intersection, difference = frozenset(y.intersection(x)), frozenset(y.difference(x))
                    if len(intersection) and len(difference):
                        p.remove(y)
                        p.update({intersection, difference})

                        if y in w:
                            w.remove(y)
                            w.update({intersection, difference})
                        else:
                            if intersection <= difference:
                                w.add(intersection)
                            else:
                                w.add(difference)

        states = {each: State(self.initial.prefix + str(i)) for i, each in enumerate(filter(lambda t: len(t), p))}

        initial, final, delta = None, set(), FADelta()
        for each in states:
            is_final = each.issubset(self.final)

            assert is_final or not len(each.intersection(self.final))

            if is_final:
                final.add(states[each])

            if self.initial in each:
                initial = states[each]

            for a in self.transition.alphabet:
                target = {s for source in self.transition.delta
                          if (source in each and a in self.transition.delta[source])
                          for s in self.transition.delta[source][a]}

                if len(target):
                    target = states[next(iter(filter(lambda z: target.issubset(z), p)))]
                    delta.add(states[each], target, a)

        return FiniteAutomata(transition=delta, initial=initial, final=final)

    def get_deterministic(self, with_states_map=False):
        assert not self.has_null_transitions()

        if not self.is_non_deterministic():
            fda = FiniteAutomata(self.transition, self.initial, self.final)
            fda.regex = self.regex

            return fda

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

        if with_states_map:
            fda = FiniteAutomata(transition, states_map[tuple({self.initial})], final), states_map
            fda.regex = self.regex

            return fda

        fda = FiniteAutomata(transition, states_map[tuple({self.initial})], final)
        fda.regex = self.regex

        return fda

    def is_non_deterministic(self, node=None, visited=None):
        # initialize stack
        if node is None:
            node = self.initial
            visited = set()

        transition_symbols = [symbol for edge in self.g.out_edges(node)
                              for symbol in self._get_symbol_from_edge(self.g, edge)]

        # visited node
        visited.add(node)

        # check for current and children, if any
        if any([transition_symbols.count(s) > 1 for s in transition_symbols]) or \
                (Transition.NULL in transition_symbols and len(transition_symbols) > 1):
            return True
        else:
            for child in filter(lambda c: c not in visited, self.g[node]):
                if self.is_non_deterministic(node=child, visited=visited):
                    return True

        return False

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
                    for symbol in filter(lambda s: s != Transition.NULL, self.transition.delta[eclose]):
                        if q in self.transition.delta[eclose][symbol]:
                            self._add_edge_to_graph(g, p, q, symbol)

        for state in list(g.nodes):
            if not networkx.has_path(g, self.initial, state):
                g.remove_node(state)
                if state in final:
                    final.remove(state)

        transition = FADelta()
        for edge in g.edges:
            for symbol in self._get_symbol_from_edge(g, edge):
                transition.add(edge[0], edge[1], symbol)

        fda = FiniteAutomata(transition, self.initial, final)
        fda.regex = self.regex

        return fda

    def has_null_transitions(self):
        for edge in self.g.edges:
            if Transition.NULL in self._get_symbol_from_edge(self.g, edge):
                return True

        return False

    def get_pushdown_automata(self):
        initial = copy.deepcopy(self.initial)
        final = self.final.copy()

        transition = PDADelta()
        for state in self.transition.delta:
            for symbol in self.transition.delta[state]:
                target = self.transition.delta[state][symbol]

                for each in target:
                    transition.add(state, each,
                                   {
                                       (symbol, Stack.EMPTY): Null()
                                   })

        return PushdownAutomata(transition, initial, final)

    def read(self, string):
        buffer = self(Buffer(data=string, initial=self.initial))
        return buffer.state() in self.final and self.is_done(buffer)

    def debug(self, string):
        buffer = Buffer(data=string, initial=self.initial)
        columns = shutil.get_terminal_size((80, 20)).columns

        size = len(str(buffer).split('\n')[0])
        right = size - (2 * len(string) + 14)
        print('-'.join(['' for _ in range(columns)]))

        print()
        print(("{:" + str(size - right) + "}").format("initial (" + str(self.initial) + ")") +
              ("{:" + str(right) + "}").format("final " + str(self.final) + ""))
        print('='.join(['' for _ in range(columns)]))

        print()
        print(buffer)
        print()
        buffer = self(buffer, debug=True)

        accepted = buffer.state() in self.final and self.is_done(buffer)
        print()
        print("{:25}".format("accepted ---->  (" + str(accepted) + ")"))
        print()

    def rebase(self, base):
        initial = self.initial.prefix + str(self.initial.number + base)
        final = {e.prefix + str(e.number + base) for e in self.final}
        fda = FiniteAutomata(self.transition.rebase(base), initial, final)
        fda.regex = self.regex

        return fda

    def get_colors(self):
        return ["purple", "red", "blue", "orange", "brown", "cyan", "green"]


class Z(FiniteAutomata):
    def __init__(self, expression):
        transition = FADelta()

        state = 0
        for i, z in enumerate(expression):
            transition.add("z" + str(state), "z" + str(state + 1), z)
            state += 1
            if i < len(expression) - 1:
                transition.add("z" + str(state), "z" + str(state + 1), Transition.NULL)
                state += 1

        initial = "z0"
        final = {"z" + str(state)}

        super().__init__(transition, initial, final)

        self.regex = expression
