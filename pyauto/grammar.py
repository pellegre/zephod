from pyauto.automata.finite import *


class Grammar:
    @staticmethod
    def get_non_terminal_from_counter(counter):
        return chr((counter - ord('A') - 1) % (ord('R') - ord('A') + 1) + ord('A'))

    @staticmethod
    def build_from_finite_automata(automata):
        if automata.has_null_transitions():
            raise RuntimeError("can't build from automata with null transitions")

        g, counter = copy.deepcopy(automata.g), ord('R') + 1
        for node in g.nodes:
            if node == automata.initial:
                if len(g.in_edges(node)) > 0:
                    networkx.set_node_attributes(g, {node: Grammar.get_non_terminal_from_counter(counter)},
                                                 "non_terminal")
                    counter -= 1
            elif node in automata.final:
                if len(g.out_edges(node)) > 0:
                    networkx.set_node_attributes(g, {node: Grammar.get_non_terminal_from_counter(counter)},
                                                 "non_terminal")
                    counter -= 1
            else:
                networkx.set_node_attributes(g, {node: Grammar.get_non_terminal_from_counter(counter)}, "non_terminal")
                counter -= 1

        non_terminal = set(networkx.get_node_attributes(g, "non_terminal").values())
        grammar = Grammar(non_terminal=non_terminal, terminal=automata.transition.alphabet)
        for node in g.nodes:
            for edge in g.out_edges(node):
                left, right = edge[0], edge[1]
                symbol = g.get_edge_data(left, right)["symbol"]

                if node == automata.initial:
                    non_terminal_left = ["S"]
                    if "non_terminal" in g.nodes[left]:
                        non_terminal_left.append(g.nodes[left]["non_terminal"])

                    if node in automata.final:
                        grammar.add("S", Transition.NULL)

                else:
                    non_terminal_left = [g.nodes[left]["non_terminal"]]

                if right in automata.final:
                    for s in symbol:
                        for each in non_terminal_left:
                            grammar.add(each, s)

                if "non_terminal" in g.nodes[right]:
                    non_terminal_right = g.nodes[right]["non_terminal"]

                    for s in symbol:
                        for each in non_terminal_left:
                            grammar.add(each, s + non_terminal_right)

        return grammar

    def __init__(self, terminal, non_terminal, start="S"):
        self.terminal, self.non_terminal = terminal, non_terminal
        self.start, self.rules = start, {}

        assert isinstance(self.terminal, set)
        assert isinstance(self.non_terminal, set)
        assert isinstance(self.start, str)

    def __str__(self):
        string = "T = " + str(self.terminal) + " ; N = " + str(self.non_terminal) + "\n"
        for ll in self.rules:
            for r in self.rules[ll]:
                string += ll + " -> " + r + "\n"
        return string[:-1]

    def __repr__(self):
        return self.__str__()

    def _get_rules_in_string(self, string):
        rules, rules_in_string = [r for r in self.rules if r in string], []

        for each in rules:
            for right in self.rules[each]:
                rules_in_string.append((each, right))

        return rules_in_string

    def _run_from_string(self, string, length, debug=False):
        rules_in_string = self._get_rules_in_string(string)

        if not len(rules_in_string):
            has_terminal = any(map(lambda s: s in self.non_terminal, string))

            if not has_terminal and len(string) >= length:

                if debug:
                    print("--->", string)

                return string

            return str()

        else:
            while len(rules_in_string):
                produced = self._run_random_rule(string, length, rules_in_string)

                run = self._run_from_string(produced, length, debug)
                if len(run):

                    if debug:
                        print(produced)

                    return run
                
            return str()

    def _run_random_rule(self, string, length, rules_in_string):
        if not len(rules_in_string):
            return string

        rules_with_non_terminal, rules_with_terminal = [], []
        for each, right in rules_in_string:
            if any(map(lambda c: c in self.non_terminal, right)):
                rules_with_non_terminal.append((each, right))
            else:
                rules_with_terminal.append((each, right))

        rules_with_non_terminal = sorted(rules_with_non_terminal, reverse=True,
                                         key=lambda r: sum([r.count(c) for c in self.non_terminal]))

        ordered_rules = rules_with_non_terminal + rules_with_terminal

        if len(ordered_rules):
            if len(string) > length or not len(rules_with_non_terminal):
                rule = random.randint(0, len(ordered_rules) - 1)
            else:
                rule = random.randint(0, len(rules_with_non_terminal) - 1)

            left, right = ordered_rules[rule]
            i = string.find(left)

            rules_in_string.remove((left, right))

            return string[:i] + right + string[i + len(left):]

        return string

    def _sanity_check(self):
        assert self.start in self.rules
        assert all(map(lambda n: n in self.rules.keys(), self.non_terminal))

    def add(self, left, right):
        assert self.start not in right

        left_sanity = list(map(lambda s: s in self.terminal or s in self.non_terminal or s == self.start, left))
        right_sanity = list(map(lambda s: s in self.terminal or s in self.non_terminal or s == self.start, right))

        assert sum(left_sanity) == len(left)
        if right != Transition.NULL:
            assert sum(right_sanity) == len(right)

        if left not in self.rules:
            self.rules[left] = [right]
            return True
        else:
            if right not in self.rules[left]:
                self.rules[left].append(right)
                return True

        return False

    def is_regular(self):
        is_regular = all(map(lambda s: s in self.non_terminal or s == self.start and len(s) == 1, self.rules))

        for ll in self.rules:
            is_regular = is_regular & \
                         all(map(lambda r: (len(r) == 1 and r in self.terminal) or
                                           (len(r) == 2 and r[0] in self.terminal and r[1] in self.non_terminal),
                                 self.rules[ll]))

        return is_regular

    def is_context_free(self):
        is_context_free = all(map(lambda s: s in self.non_terminal or s == self.start and len(s) == 1, self.rules))

        return is_context_free and not self.is_regular()

    def get_finite_automata(self):
        if not self.is_regular():
            raise RuntimeError("can't build FDA from the current rules")

        initial = "z0"
        states_map = {n: "z" + str(i + 1) for i, n in enumerate(self.non_terminal)}
        final = "z" + str(len(self.non_terminal) + 1)

        states_map[self.start], states_map[final] = initial, final

        transition = FADelta()
        for n in self.rules:
            state = states_map[n]
            for r in self.rules[n]:
                if len(r) == 1:
                    transition.add(state, final, r)
                elif len(r) == 2:
                    transition.add(state, states_map[r[1]], r[0])

        return FiniteAutomata(transition, initial, {final})

    def __call__(self, length=1, debug=False):
        self._sanity_check()

        rule = random.randint(0, len(self.rules[self.start]) - 1)
        string = self.rules[self.start][rule]

        return self._run_from_string(string, length=length, debug=debug)

    def debug(self, length):
        return [self._run_from_string(self.rules[self.start][rule], length=length, debug=True)
                for rule in range(len(self.rules[self.start]))]


class OpenGrammar(Grammar):
    def __init__(self, start="S"):
        super().__init__(terminal=set(), non_terminal=set(), start=start)
        self.non_terminal_counter = ord('S')

    def _sanity_check(self):
        pass

    def get_non_terminal(self):
        non_terminal = Grammar.get_non_terminal_from_counter(self.non_terminal_counter)
        self.non_terminal_counter -= 1
        return non_terminal

    def add(self, left, right):
        for e in left:
            if e.isupper() and e != self.start:
                self.non_terminal.add(e)
            elif e != self.start:
                self.terminal.add(e)

        for e in right:
            if e.isupper():
                self.non_terminal.add(e)
            else:
                self.terminal.add(e)

        if left not in self.rules:
            self.rules[left] = [right]
        else:
            if right not in self.rules[left]:
                self.rules[left].append(right)
