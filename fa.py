from pyauto.language import *
from pyauto.automata.finite import *
from pyauto.automata.pushdown import *
from pyauto.automata.turing import *
from pyauto.grammar import *

from utils.function import *


def nfsm_example():
    transition = FADelta()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    nfsm = FiniteAutomata(transition, "e0", {"e2", "e4"})

    g = Grammar.build_from_finite_automata(nfsm.minimal())
    data = g(length=16)
    nfsm.debug(data)

    AutomataPlotter.plot(nfsm)
    AutomataPlotter.plot(nfsm.minimal())

    print(nfsm)


def pda_example():
    grammar = Grammar(non_terminal={"A"}, terminal={"a", "b", "c"})
    grammar.add("S", "A")
    grammar.add("A", "aAa")
    grammar.add("A", "bAb")
    grammar.add("A", "c")

    transition = PDADelta()

    transition.add("z0", "z0",
                   {
                       ("a", Stack.EMPTY): Push(obj="X"),
                       ("a", "X"): Push(obj="X"),
                       ("a", "Y"): Push(obj="X"),

                       ("b", Stack.EMPTY): Push(obj="Y"),
                       ("b", "Y"): Push(obj="Y"),
                       ("b", "X"): Push(obj="Y")
                   })

    transition.add("z0", "z1",
                   {
                       ("c", Stack.EMPTY): Null(),
                       ("c", "X"): Null(),
                       ("c", "Y"): Null()
                   })

    transition.add("z1", "z1",
                   {
                       ("a", "X"): Pop(),
                       ("b", "Y"): Pop(),
                   })

    transition.add("z1", "z2",
                   {
                       ("$", Stack.EMPTY): Null()
                   })

    pda = PushdownAutomata(transition, initial="z0", final={"z2"})

    pda.debug(grammar.enumerate(length=15).pop())

    AutomataPlotter.plot(pda)
    AutomataPlotter.tikz(pda, filename="to_text", output=".")

    print(pda)


def regex_example():
    expr = (~Z("aaa") | ~Z("bb") | (~Z("cd") + Z("ab"))).minimal()

    transition = FADelta()
    transition.add("z0", "z1", {~Z("aaa")})
    transition.add("z1", "z2", {~Z("bb")})
    transition.add("z2", "y1", {Transition.NULL})
    transition.add("y1", "z3", {(~Z("cd") + Z("ab"))})

    fda = FiniteAutomata(transition, initial="z0", final={"z0", "z1", "z2", "z3"})

    g = Grammar.build_from_finite_automata(expr)
    data = g(length=16)
    fda.debug(data)

    AutomataPlotter.plot(fda)

    print(fda)


def turing_machine_example():
    transition = TuringDelta()

    transition.add("e0", "e1", {
        C(0): A("a", move=Stay()),
        C(1): A(Tape.BLANK, new="X", move=Right())
    })

    transition.add("e0", "e1", {
        C(0): A("b", move=Stay()),
        C(1): A(Tape.BLANK, new="X", move=Right())
    })

    transition.add("e0", "e3", {
        C(0): A("c", move=Right()),
        C(1): A(Tape.BLANK, move=Right())
    })

    # ---

    transition.add("e1", "e1", {
        C(0): A("a", move=Right()),
        C(1): A(Tape.BLANK, new="a", move=Right())
    })

    transition.add("e1", "e1", {
        C(0): A("b", move=Right()),
        C(1): A(Tape.BLANK, new="b", move=Right())
    })

    transition.add("e1", "e2", {
        C(0): A("c", move=Stay()),
        C(1): A(Tape.BLANK, move=Left())
    })

    # ---

    transition.add("e2", "e2", {
        C(0): A("c", move=Stay()),
        C(1): A("a", move=Left())
    })

    transition.add("e2", "e2", {
        C(0): A("c", move=Stay()),
        C(1): A("b", move=Left())
    })

    transition.add("e2", "e3", {
        C(0): A("c", move=Right()),
        C(1): A("X", move=Right())
    })

    # ---

    transition.add("e3", "e3", {
        C(0): A("b", move=Right()),
        C(1): A("b", move=Right())
    })

    transition.add("e3", "e3", {
        C(0): A("a", move=Right()),
        C(1): A("a", move=Right())
    })

    transition.add("e3", "e4", {
        C(0): A(Tape.BLANK, move=Stay()),
        C(1): A(Tape.BLANK, move=Stay())
    })

    # ---
    transition.add_tape()
    transition.add_tape()
    transition.add_tape()

    turing = TuringMachine(initial="e0", final={"e4"}, transition=transition)

    turing.debug("abbabbacabbabba")

    print(turing)


def turing_function_example():
    x, y, z, w = symbols("x y z w")

    function = FunctionMachine(expression=3 * x - 4 * z - 7 * w + 9 * y, domain=[x >= 0, y > x, z >= 0, w >= 0])

    function.info()

    assert function.run_machine({x: 3, y: 5, z: 6, w: 2})
    assert function.run_machine({x: 3, y: 5, z: 6, w: 0})

    function = FunctionMachine(expression=9 * x - 3 * y, domain=[x >= 0, y >= 0])

    function.info()

    assert function.run_machine({x: 7, y: 3})
    assert function.run_machine({x: 8, y: 0})
    assert function.run_machine({x: 8, y: 6})


def ll1_grammar():
    grammar = OpenGrammar()

    grammar.add("S", "L")
    grammar.add("S", "aB")

    grammar.add("B", "$")
    grammar.add("B", "aL")
    grammar.add("B", "ea")

    grammar.add("L", "$")
    grammar.add("L", "d")
    grammar.add("L", "aL")

    grammar.add("P", "d")
    grammar.add("P", "ed")

    print(grammar)

    print("-----")

    lang1 = set(sorted(grammar.enumerate(length=15), key=lambda w: len(w)))

    grammar = OpenGrammar()
    grammar.add("S", "$")

    grammar.add("S", "Z")
    grammar.add("S", "Y")

    grammar.add("Z", "M")

    grammar.add("M", "aN")

    grammar.add("N", "aN")
    grammar.add("N", "$")

    grammar.add("N", "ea")
    grammar.add("N", "Y")

    grammar.add("Y", "d")

    print(grammar)
    lang2 = set(sorted(grammar.enumerate(length=15), key=lambda w: len(w)))

    print(lang2)

    print(lang1.difference(lang2))


class LanguageGrammar:
    @staticmethod
    def get_non_terminal_from_counter(counter):
        return chr((counter - ord('A') - 1) % (ord('R') - ord('A') + 1) + ord('A'))

    def __init__(self, language, non_terminal='S'):
        self.language = LanguageFormula.normalize(language)

        self.initial = non_terminal
        self.non_terminal_counter = ord(self.initial)

        self.non_terminal_for_group, self.non_terminal_for_blocks = {}, {}

        for each in self.language.expression_partition:
            for i, expr in enumerate(self.language.expression_partition[each]):
                if each not in self.non_terminal_for_blocks:
                    self.non_terminal_for_blocks[each] = {}

                self.non_terminal_for_blocks[each][i] = self._get_non_terminal()

            self.non_terminal_for_group[each] = self.non_terminal_for_blocks[each][0]

        self.grammar = OpenGrammar()
        self._generate_grammar()

    def info(self):
        print("\n[+] language")
        self.language.info()

        print("\n[+] grammar")
        print("[+] initial", self.initial)
        print("[+] non terminal for group", self.non_terminal_for_group)
        print("[+] non terminal for blocks", self.non_terminal_for_blocks)

        print(self.grammar)

    def _generate_grammar(self):
        expression_groups = [self.language.symbols_partition[set(expr.exp.free_symbols).pop()]
                             for expr in self.language.expression if isinstance(expr, Pow)]

        groups = list()
        for each in filter(lambda e: e not in groups, expression_groups):
            groups.append(each)

        self.grammar.add(self.initial, ''.join([self.non_terminal_for_group[g] for g in groups]))

        for each in groups:
            non_terminal = self.non_terminal_for_group[each]

            self._add_rules_for_group(non_terminal, each)

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _add_rules_for_group(self, non_terminal, group_set, block=0):
        blocks = self.language.expression_partition[group_set]
        if block < len(blocks):
            expr = blocks[block]
            assert isinstance(expr, Pow)

            self.grammar.add(non_terminal, ''.join([str(blocks[block].base), non_terminal] +
                                                   [self.non_terminal_for_blocks[group_set][b] for b in
                                                    range(block + 1, len(blocks))]))

            next_non_terminal = self._get_non_terminal()

            self.grammar.add(non_terminal, ''.join([str(blocks[block].base) + str(blocks[block + 1].base),
                                                    next_non_terminal] + [self.non_terminal_for_blocks[group_set][b]
                                                                          for b in range(block + 2, len(blocks))]))

    def _get_non_terminal(self):
        self.non_terminal_counter -= 1
        return self.get_non_terminal_from_counter(self.non_terminal_counter)


class MachineBuilder:
    def __init__(self, planner, prefix="z"):
        self.planner = planner
        self.language = LanguageFormula.normalize(self.planner.language)

        self.transition = TuringDelta(tapes=1)

        self.prefix = prefix
        self.initial_state = State(self.prefix + "0")

        self.state_counter = 0

        self.state_description = {}

        for _ in self.planner.tapes:
            self.transition.add_tape()

        self.minimal = self._get_minimum_indices()

        if not len(self.minimal):
            raise RuntimeError("inconsistent initial conditions, check for negative values")

        self.block_from_state, self.state_from_block = {}, {}
        for i in range(0, len(self.language.expression)):
            state = self._get_new_state(description="state for parsing expression block " + str(i) +
                                                    " = " + str(self.language.expression[i]))
            self.block_from_state[state] = i
            self.state_from_block[i] = state

        self.parsed_state = self._get_new_state(description="state after parsing all blocks")

        self.state_from_block[TuringPlanner.END_BLOCK] = self.parsed_state

        for block in self.planner.explore_exit_blocks(0):
            state = self.state_from_block[block]

            next_symbol = self._get_symbol_for_state(state)

            delta = self._get_blank_delta()

            delta[C(0)] = A(next_symbol, move=Stay())
            for tape in self.planner.tapes:
                delta[C(tape)] = A(Tape.BLANK, new="X", move=Right())

            self.transition.add(self.initial_state, state, delta)

        self.final_state = self._build_planner(self.initial_state)

        self.turing = TuringMachine(initial=self.initial_state, transition=self.transition,
                                    final={self.final_state})

    def info(self):
        print("[+] expression", self.language.expression)
        print("[+] minimal", self.minimal)
        print("[+] block state", self.block_from_state)
        print("[+] final state", self.final_state)
        print("[+] conditions", self.language.conditions)

        print("[+] state description")
        for state in self.state_description:
            print("[+] state " + str(state) + " -> " + self.state_description[state])

        print("[*] turing machine")
        print(self.turing.transition)

        if self.turing.is_non_deterministic():
            non_deterministic_deltas = dict()
            print("[+] non deterministic :", self.turing.is_non_deterministic(deltas=non_deterministic_deltas))
            if len(non_deterministic_deltas):
                for state in non_deterministic_deltas:
                    for each in non_deterministic_deltas[state]:
                        print("[+] -", each)

        else:
            print("[*] the turing machine is deterministic :)")

        self.planner.info()

    def _build_planner(self, initial_state):
        next_state = initial_state

        for plan in self.planner.machine_plan:
            if isinstance(plan, BlockPlan):
                current_block = plan.block
                state = self.state_from_block[current_block]

                final_states = [self.state_from_block[block] for block in self.planner.exit_blocks[current_block]]

                if isinstance(plan, ParseLoneSymbol):
                    self._parse_lone_symbol(self._get_word_for_block(current_block), state, final_states)

                elif isinstance(plan, ParseAccumulate):
                    self._parse_block_and_count(plan.tape, self._get_word_for_block(current_block),
                                                state, final_states)

                elif isinstance(plan, ParseEqual):
                    self._parse_block_exact_equal(plan.tape, self._get_word_for_block(current_block),
                                                  state, final_states)

                else:
                    raise RuntimeError("unhandled plan " + str(plan))

                next_state = self.parsed_state

            elif isinstance(plan, OperationPlan):
                source_tape = plan.source_tape
                target_tape = plan.target_tape

                if isinstance(plan, Accumulate):
                    next_state = self._accumulate_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareGreater):
                    next_state = self._verify_greater_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareStrictGreater):
                    next_state = self._verify_strict_greater_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareUnequal):
                    next_state = self._verify_unequal_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareEqual):
                    next_state = self._verify_equal_counters(next_state, source_tape, target_tape)

                else:
                    raise RuntimeError("unhandled plan " + str(plan))

            else:
                raise RuntimeError("unhandled plan " + str(plan))

        return next_state

    def _accumulate_counters(self, initial_state, tape, result_tape):
        next_state = initial_state

        left_state = self._get_new_state(description="moving left while accumulating C" +
                                                     str(tape) + " on " + C(result_tape))

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A(Tape.BLANK, move=Left())
        self.transition.add(next_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, new="Z", move=Right())
        delta[C(tape)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state(description="hit X while accumulating C" +
                                                      str(tape) + " on " + C(result_tape))

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        next_state = self._get_new_state(description="rewind " + C(tape) + " after accumulating")

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, next_state, delta)

        return next_state

    def _verify_strict_greater_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (greater) " +
                                                     C(tape_a) + " > " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (greater) " + C(tape_a) + " > " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (greater) " +
                                          C(tape_a) + " > " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _verify_greater_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (greater / equal) " +
                                                     C(tape_a) + " >= " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (greater / equal) " + C(tape_a) + " >= " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("X", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (greater / equal) " +
                                          C(tape_a) + " >= " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _verify_equal_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (equality) " +
                                                     C(tape_a) + " == " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (equality) " + C(tape_a) + " == " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("X", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (equality) " +
                                          C(tape_a) + " == " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _verify_unequal_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (unequality) " +
                                                     C(tape_a) + " != " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (unequality) " + C(tape_a) + " != " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("X", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (unequality) " +
                                          C(tape_a) + " != " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _parse_block_exact_equal(self, counter_tape, word, initial_state, final_states):
        block = self.block_from_state[initial_state]

        rewind_state = self._get_new_state(description="rewind " + C(counter_tape) + " after parsing block " +
                                                       str(block) + " (" + str(self.language.expression[block]) + ")")

        for each in final_states:
            next_symbol = self._get_symbol_for_state(each)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())
            delta[C(counter_tape)] = A("X", move=Right())
            self.transition.add(initial_state, rewind_state, delta)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())
            delta[C(counter_tape)] = A("Z", move=Right())
            self.transition.add(rewind_state, rewind_state, delta)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())
            delta[C(counter_tape)] = A(Tape.BLANK, move=Stay())
            self.transition.add(rewind_state, each, delta)

        current_state = initial_state

        delta = self._get_blank_delta()
        delta[C(0)] = A(word[0], move=Stay())
        delta[C(counter_tape)] = A(Tape.BLANK, move=Left())
        self.transition.add(current_state, current_state, delta)

        for i in range(0, len(word)):
            delta = self._get_blank_delta()

            if i == len(word) - 1:
                next_state = initial_state
                delta[C(counter_tape)] = A("Z", move=Left())

            else:
                delta[C(counter_tape)] = A("Z", move=Stay())
                next_state = self._get_new_state(description="parsed letter " + word[i] +
                                                             " of word " + word + " in block " + str(block) +
                                                             " while verifying counter in C" + str(counter_tape))

            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

    def _parse_block_and_count(self, counter_tape, word, initial_state, final_states):
        block = self.block_from_state[initial_state]

        expr = self.language.expression[block]

        assert isinstance(expr, Pow)
        assert len(expr.exp.free_symbols) == 1

        current_state = initial_state

        for i in range(0, len(word)):
            delta = self._get_blank_delta()

            if i == len(word) - 1:
                next_state = initial_state
                delta[C(counter_tape)] = A(Tape.BLANK, new="Z", move=Right())

            else:
                next_state = self._get_new_state(description="parsing letter " + word[i] +
                                                             " of word " + word + " in block " + str(block) +
                                                             " while counting " + str(counter_tape) + " first time")

            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

        for each in final_states:
            next_symbol = self._get_symbol_for_state(each)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())

            if any([m[expr.exp] == 0 for m in self.minimal]):
                self.transition.add(initial_state, each, delta)
            else:
                self.transition.add(current_state, each, delta)

    def _parse_lone_symbol(self, word, initial_state, final_states):
        current_state = initial_state

        for i in range(0, len(word)):
            next_state = self._get_new_state(description="parsing letter " + word[i] +
                                                         " of lone word " + word)

            delta = self._get_blank_delta()
            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

        for each in final_states:
            delta = self._get_blank_delta()
            delta[C(0)] = A(self._get_symbol_for_state(each), move=Stay())
            self.transition.add(current_state, each, delta)

    def _get_symbol_for_state(self, state):
        if state == self.parsed_state:
            return Tape.BLANK

        word = self._get_word_for_state(state)

        return word[0]

    def _get_word_for_state(self, state):
        block = self.block_from_state[state]
        return self._get_word_for_block(block)

    def _get_word_for_block(self, block):
        expr = self.language.expression[block]

        if isinstance(expr, Pow):
            assert isinstance(expr.exp, Symbol)

            word = expr.base
            assert isinstance(word, Symbol)

            return str(word)
        else:
            assert isinstance(expr, Symbol)
            return str(expr)

    def _get_symbol_for_block(self, block):
        if block == len(self.language.expression):
            return Tape.BLANK

        word = self._get_word_for_block(block)

        return word[0]

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _get_blank_delta(self):
        return {C(tape): A(Tape.BLANK, move=Stay()) for tape in [0] + self.planner.tapes}

    def _get_new_state(self, description):
        self.state_counter += 1

        new_state = State(self.initial_state.prefix + str(self.state_counter))

        self.state_description[new_state] = description

        return new_state


class LanguageMachine(MachineBuilder):
    def __init__(self, language):
        super().__init__(planner=TuringPlanner(language=LanguageFormula.normalize(language)))


class TuringParser:
    def __init__(self, language):
        self.language = language

        self.language_machines = list()

        if not isinstance(language, Language):
            raise RuntimeError("can't build machine from " + str(type(language)))

        if isinstance(language, LanguageFormula):
            language_machine = LanguageMachine(language=language)

            self.language_machines.append(language_machine)
            self.turing = language_machine.turing

        elif isinstance(language, LanguageUnion):

            planners = []
            for l in language.languages:
                planners.append(TuringPlanner(language=LanguageFormula.normalize(l)))

            max_tapes = max([max(p.tapes) for p in planners])

            for planner in planners:
                for i in range(len(planner.tapes), max_tapes):
                    planner.tapes.append(i + 1)

            for planner in planners:
                language_machine = MachineBuilder(planner=planner)

                self.language_machines.append(language_machine)

            combined = self._join_machines()

            final_states = {trans.target for state in combined.transitions
                            for trans in combined.transitions[state] if trans.target not in combined.transitions}

            self.turing = TuringMachine(initial=self.language_machines[0].turing.initial,
                                        transition=combined, final=final_states)

    def info(self):
        for i, each in enumerate(self.language_machines):
            print("[*] language machine " + str(i))
            each.info()
            print()

        print("[*] turing machine")
        print(self.turing.transition)

        if self.turing.is_non_deterministic():
            non_deterministic_deltas = dict()
            print("[+] non deterministic :", self.turing.is_non_deterministic(deltas=non_deterministic_deltas))
            if len(non_deterministic_deltas):
                for state in non_deterministic_deltas:
                    for each in non_deterministic_deltas[state]:
                        print("[+] -", each)

        else:
            print("[*] the turing machine is deterministic :)")

    @staticmethod
    def _check_equal_prefix(deltas):
        prefix = [transition.source.prefix for m in deltas for s in deltas[m] for transition in deltas[m][s]]
        return len(prefix) and prefix.count(prefix[0]) == len(prefix)

    @staticmethod
    def _check_equal_states(deltas):
        number = [transition.source.number for m in deltas for s in deltas[m] for transition in deltas[m][s]]
        return len(number) and number.count(number[0]) == len(number)

    @staticmethod
    def _check_equal_transitions(deltas):
        transitions = {m: set(deltas[m][s]) for m in deltas for s in deltas[m]}
        return reduce(lambda s, t: transitions[s] == transitions[t], transitions)

    def _reprefix_transition(self, state, transition, prefix, first=None):
        if first is None:
            first = state

        if state in transition:
            for each in transition[state]:
                if each.target != first:
                    previous_state = copy.deepcopy(each.target)
                    each.target = State(prefix + str(each.target.number))

                    if each.source != first:
                        each.source = State(prefix + str(each.source.number))

                    self._reprefix_transition(previous_state, transition, prefix, first)

    def _join_machines(self):
        combined = TuringDelta(tapes=max(self.language_machines,
                                         key=lambda l: l.turing.transition.tapes).transition.tapes)

        for machine in self.language_machines:
            for i in range(machine.turing.transition.tapes, combined.tapes):
                machine.turing.transition.add_tape()

        deltas = [copy.deepcopy(machine.turing.transition) for machine in self.language_machines]

        for delta in deltas:
            for j in range(delta.tapes, combined.tapes):
                delta.add_tape()

        delta_stack = {i: delta for i, delta in enumerate(deltas)}

        for block in range(0, len(max(self.language_machines,
                                  key=lambda l: len(l.language.expression)).language.expression)):

            for i in list(delta_stack):
                if len(self.language_machines[i].language.expression) == block:
                    del delta_stack[i]

            if len(delta_stack) == 1:
                break

            for k in filter(lambda d: d in delta_stack, list(delta_stack)):
                expr = self.language_machines[k].language.expression[block]
                minimal = sorted(m[expr.exp] for m in self.language_machines[k].minimal)

                for m in filter(lambda d: d != k and d in delta_stack, list(delta_stack)):
                    other = self.language_machines[m].language.expression[block]

                    should_reprefix = False

                    if isinstance(other, Pow) and isinstance(expr, Pow):
                        other_minimal = sorted(m[expr.exp] for m in self.language_machines[m].minimal)

                        if len(other_minimal) != len(minimal) or \
                                any([o != e for o, e in zip(other_minimal, minimal)]) or other.base != expr.base:
                            should_reprefix = True

                    elif isinstance(other, Symbol) and isinstance(expr, Symbol):
                        if other != expr:
                            should_reprefix = True

                    else:
                        should_reprefix = True

                    if should_reprefix:
                        machine = self.language_machines[m]
                        current_prefix = machine.prefix

                        state = self._get_state_for_block(m, block - 1)

                        self._reprefix_transition(state, deltas[m].transitions, chr(ord(current_prefix) - m))

                        del delta_stack[m]

        for i in range(0, len(deltas)):
            for each in deltas[i].transitions:
                for transition in deltas[i].transitions[each]:
                    combined.merge_transition(transition)

        return combined

    def _get_state_for_block(self, m, block):
        if block < 0:
            return self.language_machines[m].initial_state

        else:
            return self.language_machines[m].state_from_block[block]


def test_language_turing_machine_1():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k >= 0, m >= 0, n >= 0])

    cfl.info()

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)
        lang_machine = TuringParser(language=cfl)
        lang_machine.info()

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)

        assert read_status


def test_language_turing_machine_2():
    print("[+] FD ")
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k >= 0, m >= 0, Eq(n, m)])

    cfl.info()
    lang_machine = TuringParser(language=cfl)

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_3():
    a, e, b, c, aa, ccc, bb = symbols("a e b c aa ccc bb")
    m, k, n, p, q = symbols("m k n p q")

    cfl_with_ones = LanguageFormula(expression=[aa, aa ** k, aa ** q, e ** n, b, b ** k,
                                                bb ** p, c ** k, ccc ** m, c ** n, b ** m],
                                    conditions=[k >= 0, m >= 0, Eq(n, m + 1), Eq(p + 1, k), Eq(q, m)])

    cfl = LanguageFormula(expression=[aa, aa ** k, aa ** q, e ** n, b, b ** k,
                                      bb ** p, c ** k, ccc ** m, c ** n, b ** m],
                          conditions=[k >= 0, m >= 0, Eq(n, m), Eq(p, k), Eq(q, m)])

    cfl.info()
    difference = set(cfl_with_ones.enumerate_strings(length=25)).difference(cfl.enumerate_strings(length=25))

    lang_machine = TuringParser(language=cfl)

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    for data in difference:
        print("[+] testing string from ones language", data)
        lang_machine.info()

        read_status = lang_machine.turing.read(data)

        if read_status:
            lang_machine.turing.debug(data)

        assert not read_status


def test_language_turing_machine_4():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl_with_null = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                                    conditions=[k >= 0, m >= 0, n >= 0])

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k > 0, m > 0, n > 0])
    cfl.info()

    difference = set(cfl_with_null.enumerate_strings(length=25)).difference(cfl.enumerate_strings(length=25))

    lang_machine = TuringParser(language=cfl)

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)
        lang_machine.info()

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)

        assert read_status

    for data in difference:
        print("[+] testing string from nulled language", data)
        lang_machine.info()

        read_status = lang_machine.turing.read(data)

        if read_status:
            lang_machine.turing.debug(data)

        assert not read_status


def test_language_turing_machine_5():
    a, e, b, c, aa, ccc, ab, d, f = symbols("a e b c aa ccc ab d f")
    m, k, n, q = symbols("m k n q")

    cfl = LanguageFormula(expression=[e ** q, a ** k, e ** n, ab ** k, b ** k, c ** m, b ** m, ccc ** n, f ** q],
                          conditions=[k > 0, q > 0, Eq(k + q, m + n)])
    cfl.info()

    print("\n[+] language machine")

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_6():
    a, e, b, c, aa, ccc, ab, d, f = symbols("a e b c aa ccc ab d f")
    m, k, n, q, r, s = symbols("m k n q r s")

    cfl = LanguageFormula(expression=[e ** q, a ** k, e ** n, ab ** k, b ** r,
                                      c ** m, b ** m, ccc ** n, f ** q, aa ** s],
                          conditions=[k > 0, q > 0, Eq(k, m + n), Eq(r + s, q)])
    cfl.info()

    print("\n[+] language machine")

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_7():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n < m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n <= m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n > m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n >= m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_8():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** (2 * k + 1), e ** n, b ** (k + 2), c ** (k + 3 * m)],
                          conditions=[k >= 0, m > 0, n < m])

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_9():
    n, j = symbols("n j")
    ac, b, c, d = symbols("ac b c d")

    lang = LanguageFormula(expression=[ac ** (n + 1), b ** j, c ** n, d ** j], conditions=[n >= 0, j > n])

    print(lang.enumerate_strings(length=15))

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_10():
    n, k, d = symbols("n k d")
    a, d, b, c, f = symbols("a d b c f")

    lang = LanguageFormula(expression=[a ** (k + n), d ** n, b ** (k + 1), c ** k, f ** d],
                           conditions=[n >= 0, k >= 0, d < n, d >= 0])

    print(lang.enumerate_strings(length=15))

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_11():
    s, n, k = symbols("s n k")
    a, d, b, e = symbols("a d b e")

    lang = LanguageFormula(expression=[a ** (2 * n), d ** (s + 1), b ** k, e ** n],
                           conditions=[s >= 0, n > 0, k > 0, ~Eq(n, k)])

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_union_turing_machine_12():
    n, j, i = symbols("n j i")
    aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i], conditions=[n > 0, j > n, i > 0])
    lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** (2 * n + 1), e ** i], conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=15):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_union_turing_machine_13():
    n, j, i, r = symbols("n j i r")
    aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])
    lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** (2 * n + 1), e ** i, c ** n, c ** i],
                             conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=25):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_union_turing_machine_14():
    n, j, i, r = symbols("n j i r")
    aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])
    lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** (2 * n), e ** i, c ** n, c ** i],
                             conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=25):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])

    lang_b = LanguageFormula(expression=[b ** n, a ** j, d ** (2 * n), e ** i, c ** n, c ** i],
                             conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=25):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    # lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
    #                          conditions=[n > 0, j > n, i > 0, r >= 0])
    #
    # lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j],
    #                          conditions=[n > 0, j > n, i > 0, r >= 0])
    #
    # lang = lang_a + lang_b
    #
    # lang.info()
    #
    # lang_machine = TuringParser(language=lang)
    #
    # lang_machine.info()
    #
    # for data in lang.enumerate_strings(length=25):
    #     print("[+] testing string", data)
    #     read_status = lang_machine.turing.read(data)
    #
    #     if not read_status:
    #         lang_machine.turing.debug(data)
    #         lang_machine.info()
    #
    #     assert read_status


def testing_turing_language():
    test_language_turing_machine_1()
    test_language_turing_machine_2()
    test_language_turing_machine_3()
    test_language_turing_machine_4()
    test_language_turing_machine_5()
    test_language_turing_machine_6()
    test_language_turing_machine_7()
    test_language_turing_machine_8()
    test_language_turing_machine_9()
    test_language_turing_machine_10()
    test_language_turing_machine_11()

    test_language_union_turing_machine_12()
    test_language_union_turing_machine_13()
    test_language_union_turing_machine_14()

    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** (2 * k + 2), e ** n, b ** (k + 1), c ** (k + 3 * m)],
                          conditions=[k >= 0, m >= 0, n > m])

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    lang_machine.turing.debug("aaaaaaeeeeebccccc")  # it shouldn't detect it
    assert not lang_machine.turing.read("aaaaaaeeeeebccccc")  # it shouldn't detect it


class MachinePlan:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__str__()


class BlockPlan(MachinePlan):
    def __init__(self, block, exit_blocks):
        self.block = block
        self.exit_blocks = exit_blocks

    def __str__(self):
        return "block " + str(self.block) + " -> " + str(self.exit_blocks)

    def __repr__(self):
        return self.__str__()


class ParseLoneSymbol(BlockPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "ParseLoneSymbol on " + super().__str__()


class ParseAccumulate(BlockPlan):
    def __init__(self, tape, **kwargs):
        super().__init__(**kwargs)
        self.tape = tape

    def __str__(self):
        return "ParseAccumulate on " + super().__str__() + " in tape " + str(self.tape)


class ParseEqual(BlockPlan):
    def __init__(self, tape, **kwargs):
        super().__init__(**kwargs)
        self.tape = tape

    def __str__(self):
        return "ParseEqual on " + super().__str__() + " in tape " + str(self.tape)


class OperationPlan(MachinePlan):
    def __init__(self, source_tape, target_tape):
        self.source_tape, self.target_tape = source_tape, target_tape

    def __str__(self):
        return "tape " + str(self.source_tape) + " and " + str(self.target_tape)

    def __repr__(self):
        return self.__str__()


class RewindAll(MachinePlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Rewind all tapes"


class Accumulate(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Accumulate on " + super().__str__()


class CompareGreater(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareGreater on " + super().__str__()


class CompareStrictGreater(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareStrictGreater on " + super().__str__()


class CompareUnequal(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareUnequal on " + super().__str__()


class CompareEqual(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareEqual on " + super().__str__()


class TuringPlanner:
    END_BLOCK = -1

    def __init__(self, language: LanguageFormula):
        self.language = language

        self.symbol_tape = dict()
        self.tapes = list()

        expression_symbols = list()
        for each in self.language.expression:
            if isinstance(each, Pow):
                if each.exp not in expression_symbols:
                    expression_symbols.append(each.exp)

        for i, each in enumerate(expression_symbols):
            self.tapes.append(i + 1)
            self.symbol_tape[each] = i + 1

        self.machine_plan = list()
        self.exit_blocks = dict()

        for i in range(0, len(self.language.expression)):
            self.exit_blocks[i] = self._get_exit_blocks(i)

        self._process_blocks(current=0)

        self._verify_conditions()

    def info(self):
        print("\n[+] language")
        self.language.info()

        print("\n[+] turing planner")
        print("[+] exit blocks", self.exit_blocks)
        print("[+] symbol tapes", self.symbol_tape)

        for i, each in enumerate(self.machine_plan):
            print(i, each)

    def _process_blocks(self, current=0, symbol_stack=None):
        if symbol_stack is None:
            symbol_stack = set()

        if current < len(self.language.expression):
            expr = self.language.expression[current]

            if isinstance(expr, Symbol):
                self.machine_plan.append(ParseLoneSymbol(block=current, exit_blocks=self.exit_blocks[current]))

            elif isinstance(expr, Pow):
                if expr.exp not in symbol_stack:
                    symbol_stack.add(expr.exp)

                    self.machine_plan.append(ParseAccumulate(tape=self.symbol_tape[expr.exp],
                                                             block=current, exit_blocks=self.exit_blocks[current]))

                else:
                    self.machine_plan.append(ParseEqual(tape=self.symbol_tape[expr.exp],
                                                        block=current, exit_blocks=self.exit_blocks[current]))

            self._process_blocks(current + 1, symbol_stack)

    def _verify_conditions(self):
        if not all([len(c.free_symbols) == 1 for c in self.language.conditions]):

            for c in filter(lambda c: len(c.free_symbols) > 1, self.language.conditions):
                if not isinstance(c.rhs, Symbol):
                    symbol, rest = c.rhs.args[0], c.rhs.args[1:]
                    target_tape = self.symbol_tape[symbol]

                    for each in rest:
                        source_tape = self.symbol_tape[each]
                        self.machine_plan.append(Accumulate(target_tape=target_tape, source_tape=source_tape))

                    self.symbol_tape[c.rhs] = target_tape

                if not isinstance(c.lhs, Symbol):
                    symbol, rest = c.lhs.args[0], c.lhs.args[1:]
                    target_tape = self.symbol_tape[symbol]

                    for each in rest:
                        source_tape = self.symbol_tape[each]
                        self.machine_plan.append(Accumulate(target_tape=target_tape, source_tape=source_tape))

                    self.symbol_tape[c.lhs] = target_tape

                if isinstance(c, Eq):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareEqual(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, StrictGreaterThan):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareStrictGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, GreaterThan):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, StrictLessThan):
                    source_tape = self.symbol_tape[c.rhs]
                    target_tape = self.symbol_tape[c.lhs]

                    self.machine_plan.append(CompareStrictGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, LessThan):
                    source_tape = self.symbol_tape[c.rhs]
                    target_tape = self.symbol_tape[c.lhs]

                    self.machine_plan.append(CompareGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, Unequality):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareUnequal(target_tape=target_tape, source_tape=source_tape))

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _get_exit_blocks(self, block):
        if block < len(self.language.expression) - 1:
            expr = self.language.expression[block]

            if isinstance(expr, Symbol):
                next_expr = self.language.expression[block + 1]

                if isinstance(next_expr, Symbol):
                    return [block + 1]

                else:
                    minimal = self._get_minimum_indices()

                    if all([m[next_expr.exp] for m in minimal]) > 0:
                        return [block + 1]

                    else:
                        return self.explore_exit_blocks(block + 1)

            elif isinstance(expr, Pow):
                next_expr = self.language.expression[block + 1]

                if isinstance(next_expr, Symbol) or expr.exp == next_expr.exp:
                    return [block + 1]

                else:
                    minimal = self._get_minimum_indices({expr.exp: 1})

                    if all([m[next_expr.exp] for m in minimal]) > 0:
                        return [block + 1]

                    else:
                        return self.explore_exit_blocks(block + 1)

            else:
                raise RuntimeError("unrecognized expression " + str(expr))

        else:
            return [TuringPlanner.END_BLOCK]

    def explore_exit_blocks(self, block):
        stack, exit_blocks = set(), list()

        for b in range(block, len(self.language.expression) + 1):
            if b < len(self.language.expression):
                expr = self.language.expression[b]

                if isinstance(expr, Symbol):
                    exit_blocks.append(b)
                    break

                if expr.exp not in stack:
                    exit_blocks.append(b)

                minimal = self._get_minimum_indices()

                if all([m[expr.exp] for m in minimal]) > 0:
                    break

                stack.add(expr.exp)

            else:
                exit_blocks.append(TuringPlanner.END_BLOCK)

        return exit_blocks


def main():
    print("[+] FD ")
    testing_turing_language()


main()
