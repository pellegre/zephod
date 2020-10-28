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


class LanguageMachine:
    def __init__(self, language, prefix="z"):
        self.language = LanguageFormula.normalize(language)

        self.transition = TuringDelta(tapes=1)

        self.prefix = prefix
        self.initial_state = State(self.prefix + "0")

        self.state_counter, self.tape_counter = 0, 0
        self.symbol_tape = {}

        self.state_description = {}

        for each in self.language.symbols:
            tape = self._get_new_tape()
            self.symbol_tape[each] = tape

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

        for state in self._get_next_states(0):
            next_symbol = self._get_symbol_for_state(state)

            delta = self._get_blank_delta()

            delta[C(0)] = A(next_symbol, move=Stay())
            for tape in range(1, self.tape_counter + 1):
                delta[C(tape)] = A(Tape.BLANK, new="X", move=Right())

            self.transition.add(self.initial_state, state, delta)

        self.visited_counters = set()

        self._parse_block(block=0)

        self.final_state = self._verify_conditions(initial_state=self.parsed_state)

        self.turing = TuringMachine(initial=self.initial_state, transition=self.transition,
                                    final={self.final_state})

    def info(self):
        print("[+] expression", self.language.expression)
        print("[+] minimal", self.minimal)
        print("[+] block state", self.block_from_state)
        print("[+] final state", self.final_state)
        print("[+] symbol tape", self.symbol_tape)
        print("[+] conditions", self.language.conditions)

        for block in range(0, len(self.language.expression)):
            print("[+] block", block, ",", self.state_from_block[block],
                  " -> ", self._get_next_states(block + 1))

        print("[+] state description")
        for state in self.state_description:
            print("[+] state " + str(state) + " -> " + self.state_description[state])

    def _verify_conditions(self, initial_state):
        if all([len(c.free_symbols) == 1 for c in self.language.conditions]):
            return initial_state

        else:
            next_state = initial_state

            for c in filter(lambda c: len(c.free_symbols) > 1, self.language.conditions):
                if not isinstance(c.rhs, Symbol):
                    next_state = self._accumulate_counters(next_state, c.rhs)

                if not isinstance(c.lhs, Symbol):
                    next_state = self._accumulate_counters(next_state, c.lhs)

                if isinstance(c, Eq):
                    next_state = self._verify_equal_counters(next_state, c.lhs, c.rhs)

                elif isinstance(c, StrictGreaterThan):
                    next_state = self._verify_strict_greater_counters(next_state, c.lhs, c.rhs)

                elif isinstance(c, GreaterThan):
                    next_state = self._verify_greater_counters(next_state, c.lhs, c.rhs)

                elif isinstance(c, StrictLessThan):
                    next_state = self._verify_strict_greater_counters(next_state, c.rhs, c.lhs)

                elif isinstance(c, LessThan):
                    next_state = self._verify_greater_counters(next_state, c.rhs, c.lhs)

                elif isinstance(c, Unequality):
                    next_state = self._verify_unequal_counters(next_state, c.rhs, c.lhs)

        return next_state

    def _accumulate_counters(self, initial_state, expr):
        assert all([isinstance(e, Symbol) for e in expr.args])

        symbol, rest = expr.args[0], expr.args[1:]
        result_tape = self._get_tape_for_symbol(symbol)

        self.symbol_tape[expr] = result_tape

        next_state = initial_state

        for each in rest:
            tape = self._get_tape_for_symbol(each)

            left_state = self._get_new_state(description="moving left while accumulating " +
                                                         str(expr) + " on " + C(result_tape))

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
            delta[C(tape)] = A(Tape.BLANK, move=Left())
            self.transition.add(next_state, left_state, delta)

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, new="Z", move=Right())
            delta[C(tape)] = A("Z", move=Left())
            self.transition.add(left_state, left_state, delta)

            right_state = self._get_new_state(description="hit X while accumulating " +
                                                          str(expr) + " on " + C(result_tape))

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
            delta[C(tape)] = A("X", move=Right())
            self.transition.add(left_state, right_state, delta)

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
            delta[C(tape)] = A("Z", move=Right())
            self.transition.add(right_state, right_state, delta)

            next_state = self._get_new_state(description="rewind " + C(tape) + " after accumulating " + str(expr))

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
            delta[C(tape)] = A(Tape.BLANK, move=Stay())
            self.transition.add(right_state, next_state, delta)

        return next_state

    def _verify_strict_greater_counters(self, initial_state, a, b):
        tape_a = self._get_tape_for_symbol(a)
        tape_b = self._get_tape_for_symbol(b)

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

    def _verify_greater_counters(self, initial_state, a, b):
        tape_a = self._get_tape_for_symbol(a)
        tape_b = self._get_tape_for_symbol(b)

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

    def _verify_equal_counters(self, initial_state, a, b):
        tape_a = self._get_tape_for_symbol(a)
        tape_b = self._get_tape_for_symbol(b)

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

    def _verify_unequal_counters(self, initial_state, a, b):
        tape_a = self._get_tape_for_symbol(a)
        tape_b = self._get_tape_for_symbol(b)

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

    def _parse_block(self, block):
        state = self.state_from_block[block]
        word = self._get_word_for_state(state)

        next_states = self._get_next_states(block + 1)

        if isinstance(self.language.expression[block], Symbol):
            self._parse_lone_symbol(word, state, next_states)

        elif isinstance(self.language.expression[block], Pow):
            counter = self._get_counter_for_block(block)

            if counter not in self.visited_counters:
                self._parse_block_and_count(word, state, next_states)
                self.visited_counters.add(counter)
            else:
                self._parse_block_exact_equal(word, state, next_states)

        if not (len(next_states) == 1 and self.parsed_state in next_states):
            self._parse_block(block + 1)

    def _parse_block_exact_equal(self, word, initial_state, final_states):
        block = self.block_from_state[initial_state]

        counter = self._get_counter_for_block(block)
        counter_tape = self._get_tape_for_symbol(counter)

        assert counter in self.visited_counters

        next_counter = self._get_counter_for_block(block + 1)
        rewind_state = self._get_new_state(description="rewind " + C(counter_tape) + " after parsing block " +
                                           str(block) + " (" + str(self.language.expression[block]) + ")")

        minimal = self._get_minimum_indices(constraints={counter: 1})

        if isinstance(next_counter, Symbol) and all([m[next_counter] > 0 for m in minimal]):
            next_symbol = self._get_symbol_for_block(block + 1)

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
            self.transition.add(rewind_state, self.state_from_block[block + 1], delta)

        else:
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
                                                             " while verifying counter " + str(counter))

            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

    def _parse_block_and_count(self, word, initial_state, final_states):
        block = self.block_from_state[initial_state]

        expr = self.language.expression[block]

        assert isinstance(expr, Pow)
        assert len(expr.exp.free_symbols) == 1

        counter = expr.exp.free_symbols.pop()
        counter_tape = self._get_tape_for_symbol(counter)

        current_state = initial_state

        for i in range(0, len(word)):
            delta = self._get_blank_delta()

            if i == len(word) - 1:
                next_state = initial_state
                delta[C(counter_tape)] = A(Tape.BLANK, new="Z", move=Right())

            else:
                next_state = self._get_new_state(description="parsing letter " + word[i] +
                                                             " of word " + word + " in block " + str(block) +
                                                             " while counting " + str(counter) + " first time")

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

    def _get_tape_for_symbol(self, symbol):
        return self.symbol_tape[symbol]

    def _get_counter_for_block(self, block):
        if block == len(self.language.expression):
            return Number(1)

        expr = self.language.expression[block]

        if isinstance(expr, Pow):
            assert len(expr.exp.free_symbols) == 1

            counter = expr.exp.free_symbols.pop()

            return counter

        else:
            return Number(1)

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
        return {C(tape): A(Tape.BLANK, move=Stay()) for tape in range(0, self.tape_counter + 1)}

    def _get_new_tape(self):
        self.tape_counter += 1
        self.transition.add_tape()

        return self.tape_counter

    def _get_new_state(self, description):
        self.state_counter += 1

        new_state = State(self.initial_state.prefix + str(self.state_counter))

        self.state_description[new_state] = description

        return new_state

    def _get_next_states(self, index=0):
        if index < len(self.language.expression):
            state = self.state_from_block[index]
            expr = self.language.expression[index]

            if isinstance(expr, Pow):
                assert isinstance(expr.exp, Symbol)

                word = expr.base
                assert isinstance(word, Symbol)

                if all([m[expr.exp] > 0 for m in self.minimal]):
                    return [state]
                else:
                    upstream = self._get_next_states(index + 1)
                    return [state] + upstream

            else:
                assert isinstance(expr, Symbol)

                return [state]
        else:
            return [self.parsed_state]


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
            for i, language in enumerate(language.languages):
                language_machine = LanguageMachine(language=language)

                self.language_machines.append(language_machine)

            combined = TuringDelta(tapes=max(self.language_machines,
                                             key=lambda l: l.turing.transition.tapes).transition.tapes)

            for machine in self.language_machines:
                for i in range(machine.turing.transition.tapes, combined.tapes):
                    machine.turing.transition.add_tape()

            deltas = [copy.deepcopy(machine.turing.transition) for machine in self.language_machines]

            for delta in deltas:
                for j in range(delta.tapes, combined.tapes):
                    delta.add_tape()

            self._join_machines(state=0, deltas=deltas, combined=combined)

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

    def _check_equal_prefix(self, deltas):
        prefix = [transition.source.prefix for m in deltas for s in deltas[m] for transition in deltas[m][s]]
        return len(prefix) and prefix.count(prefix[0]) == len(prefix)

    def _check_equal_states(self, deltas):
        number = [transition.source.number for m in deltas for s in deltas[m] for transition in deltas[m][s]]
        return len(number) and number.count(number[0]) == len(number)

    def _check_equal_transitions(self, deltas):
        transitions = {m: set(deltas[m][s]) for m in deltas for s in deltas[m]}
        return reduce(lambda s, t: transitions[s] == transitions[t], transitions)

    def _reprefix_transition(self, state, transition, prefix, first=None):
        if not first:
            first = state
            print(" first ---> " + str(state) + " " + prefix)

        if state in transition:
            for each in transition[state]:
                if each.target != first:
                    previous_state = copy.deepcopy(each.target)
                    each.target = State(prefix + str(each.target.number))

                    if each.source != first:
                        each.source = State(prefix + str(each.source.number))

                    print(" ---> each.target != state " + str(each.target) + " " + str(state) + "   " + str(each))

                    self._reprefix_transition(previous_state, transition, prefix, first)

    def _join_machines(self, state, deltas, combined):
        transitions = {}
        current_state = State(self.language_machines[0].prefix + str(state))

        for i in range(len(self.language_machines)):
            transition = deltas[i]

            if current_state in transition.delta:
                transitions[i] = {current_state: transition.transitions[current_state]}

        print("_check_equal_prefix", self._check_equal_prefix(transitions))
        print("_check_equal_states", self._check_equal_states(transitions))

        if self._check_equal_prefix(transitions) and self._check_equal_states(transitions):
            first_machine = next(iter(transitions))

            for each in transitions[first_machine]:
                for transition in transitions[first_machine][each]:
                    combined.merge_transition(transition)

            # split = False
            # block = self.language_machines[0].block_from_state[current_state]
            # block_expression = [m.language.expression[block] for m in self.language_machines]

            if not self._check_equal_transitions(transitions):
                for i in range(1, len(transitions)):
                    machine = self.language_machines[i]
                    current_prefix = machine.prefix

                    current_state = State(machine.prefix + str(state))
                    self._reprefix_transition(current_state, deltas[i].transitions, chr(ord(current_prefix) - i))

        for m in transitions:
            for each in transitions[m]:
                for transition in transitions[m][each]:
                    combined.merge_transition(transition)

        if len(transitions) > 0:
            self._join_machines(state + 1, deltas, combined)


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

    cfl = LanguageFormula(expression=[a ** (2*k + 1), e ** n, b ** (k + 2), c ** (k + 3*m)],
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

    lang = LanguageFormula(expression=[ac**(n + 1), b**j, c**n, d**j], conditions=[n >= 0, j >n])

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

    lang = LanguageFormula(expression=[a**(k + n), d**n, b**(k + 1), c**k, f**d],
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

    lang = LanguageFormula(expression=[a**(2*n), d**(s + 1), b**k, e**n], conditions=[s >= 0, n > 0, k > 0, ~Eq(n, k)])

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_12():
    s, t, n, j = symbols("s t n j")
    s1, s2, s3, s4, s5 = symbols("1 2 3 4 5")

    lang = LanguageFormula(expression=[s1 ** (2*s + 1), s2 ** j, s3 ** n, s4 ** (t + 2*n), s5 ** (n + 2)],
                           conditions=[s > 0, t >= 0, n >= 0, j >= 0, j < t, s > j])

    print(lang.enumerate_strings(length=15))
    print(lang.get_index_space(length=15))

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

    lang_a = LanguageFormula(expression=[b**n, a**j, c**n, d**i], conditions=[n > 0, j > n, i > 0])
    lang_b = LanguageFormula(expression=[b**n, a**j, c**(2*n), e**i], conditions=[n > 0, j > n, i > 0])

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

    lang_a = LanguageFormula(expression=[b**n, a**j, c**n, d**i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])
    lang_b = LanguageFormula(expression=[b**n, a**j, c**(2*n + 1), e**i, c**n, c ** i],
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


def main():
    print("[+] FD ")

    # n, j, i, r = symbols("n j i r")
    # aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")
    #
    # lang_a = LanguageFormula(expression=[b**n, a**j, c**n, d**i],
    #                          conditions=[n > 0, j > n, i > 0])
    # lang_b = LanguageFormula(expression=[b**n, a**j, c**(2*n), e**i],
    #                          conditions=[n > 0, j > n, i > 0])
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

    # test_language_turing_machine_1()
    # test_language_turing_machine_2()
    # test_language_turing_machine_3()
    # test_language_turing_machine_4()
    # test_language_turing_machine_5()
    # test_language_turing_machine_6()
    # test_language_turing_machine_7()
    # test_language_turing_machine_8()
    # test_language_turing_machine_9()
    # test_language_turing_machine_10()
    # test_language_turing_machine_11()

    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** (2*k + 2), e ** n, b ** (k + 1), c ** (k + 3*m)],
                          conditions=[k >= 0, m >= 0, n > m])

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    # lang_machine.turing.debug("aaaaaaeeeeebccccc")

    # lang_machine.info()
    #
    # for data in cfl.enumerate_strings(length=20):
    #     print("[+] testing string", data)
    #     read_status = lang_machine.turing.read(data)
    #
    #     if not read_status:
    #         lang_machine.turing.debug(data)
    #         lang_machine.info()
    #
    #     assert read_status


main()
