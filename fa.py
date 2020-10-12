from pyauto.language import *
from pyauto.automata.finite import *
from pyauto.automata.pushdown import *
from pyauto.automata.turing import *


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

    pda.debug(grammar(length=15))

    AutomataPlotter.plot(pda)

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


class TuringFunction:
    def __init__(self, variables):
        self.expression = variables
        self.symbol_tapes, self.tape_symbols, self.tape_counter = dict(), dict(), 1

        self.symbols = variables

        self.tape_counter = 0
        self.state_counter = 0

        self.initial = State("z0")
        self.turing_input = self._get_copy_input_machine()

    def get_unary_input(self, values):
        assert len(values) == len(self.symbols)

        unary_input = []
        for tape in range(1, len(self.tape_symbols) + 1):
            symbol = self.tape_symbols[tape]
            value = values[symbol]
            unary_input += ['1' for _ in range(value)] + ['0']

        return ''.join(unary_input[:-1])

    def info(self):
        print("[+] Turing function")
        print("[+] - input symbols")
        for symbol in self.symbol_tapes:
            print("  === symbol", symbol, "is at tape", C(self.symbol_tapes[symbol]))

    def _get_blank_delta(self, tapes=None):
        if not tapes:
            tapes = self.tape_counter
        return {C(tape): A(Tape.BLANK, move=Stay()) for tape in range(0, tapes + 1)}

    def _get_copy_input_machine(self):
        transition = TuringDelta()

        input_symbols = self.symbols.copy()
        tape = self._add_to_tape(input_symbols[0])
        state = self._get_new_state()

        # ---- X mark

        delta = self._get_blank_delta()

        delta[C(0)] = A("1", move=Stay())
        delta[C(tape)] = A(Tape.BLANK, new="X", move=Right())

        transition.add(self.initial, state, delta)

        machine = TuringMachine(initial=self.initial, transition=transition, final={state})

        return self._update_copy_machine(self.symbols.copy(), machine)

    def _update_copy_machine(self, input_symbols: list, machine: TuringMachine):
        transition = machine.transition

        symbol = input_symbols.pop(0)
        tape = self.symbol_tapes[symbol]

        assert len(machine.final) == 1
        state = machine.final.pop()
        final_state = self._get_new_state()

        # ---- copy input

        delta = self._get_blank_delta()
        delta[C(0)] = A("1", move=Right())
        delta[C(tape)] = A(Tape.BLANK, new="1", move=Right())

        transition.add(state, state, delta)

        if len(input_symbols):
            # ---- detect next input

            next_tape = self._add_to_tape(input_symbols[0])
            transition.add_tape()

            delta = self._get_blank_delta()
            delta[C(0)] = A("0", move=Right())
            delta[C(next_tape)] = A(Tape.BLANK, new="X", move=Right())

            transition.add(state, final_state, delta)

            # ---- create machine

            machine = TuringMachine(initial=machine.initial, transition=transition,
                                    final={final_state})

            return self._update_copy_machine(input_symbols, machine)

        else:
            # ---- final machine

            delta = self._get_blank_delta()
            transition.add(state, final_state, delta)

            # ---- final machine

            machine = TuringMachine(initial=machine.initial, transition=transition,
                                    final={final_state})

            return machine

    def _add_to_tape(self, expression):
        self.tape_counter += 1

        self.symbol_tapes[expression] = self.tape_counter
        self.tape_symbols[self.tape_counter] = expression

        return self.tape_counter

    def _get_new_state(self):
        self.state_counter += 1

        return self.initial.prefix + str(self.state_counter)


def main():
    print("[+] FD ")

    x, y, w, z = symbols("x y w z")
    function = TuringFunction(variables=[z, y, w, x])

    unary_input = function.get_unary_input({x: 5, y: 3, w: 4, z: 9})

    function.turing_input.debug(unary_input)
    function.info()

    # grammar = OpenGrammar()
    #
    # grammar.add("S", "A")
    #
    # grammar.add("A", "aABC")
    # grammar.add("A", "aBC")
    #
    # grammar.add("CB", "BC")
    #
    # grammar.add("aB", "ab")
    # grammar.add("bB", "bb")
    # grammar.add("bC", "bc")
    # grammar.add("cC", "cc")
    #
    # print(grammar(length=10))
    #
    # exit()


main()
