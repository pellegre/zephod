from pyauto.delta import *
from pyauto.automata.base import *
from pyauto.tape import *

from sympy.logic.boolalg import *
from sympy.assumptions.refine import *

from functools import reduce

from itertools import product


import shutil


# --------------------------------------------------------------------
#
# Turing machine transitions
#
# --------------------------------------------------------------------

class Right(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return MoveRightAction(**kwargs)


class Left(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return MoveLeftAction(**kwargs)


class Stay(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return NoneAction(**kwargs)


class A(Action):
    def __init__(self, symbol, move, new=None, **kwargs):
        self.symbol = symbol
        self.move = move
        self.new = new
        super().__init__(**kwargs)

    def _get_action(self, **kwargs):
        assert isinstance(self.move, Action)
        if not self.new:
            return [self.move._get_action(on_symbol=self.symbol)]

        else:
            return [WriteAction(on_symbol=self.symbol, new_symbol=self.new),
                    self.move._get_action(on_symbol=self.new)]


def C(number):
    return Tape.N(number)

# --------------------------------------------------------------------
#
# Turing machine function
#
# --------------------------------------------------------------------


class TuringDelta(Delta):
    def __init__(self, tapes=None):
        super().__init__()
        self.tapes = tapes

        # state management
        self.state_counter = 0
        self.state_description = {}

    def get_new_state(self, prefix, description):
        self.state_counter += 1

        new_state = State(prefix + str(self.state_counter))

        self.state_description[new_state] = description

        return new_state

    def get_blank_delta(self):
        return {C(tape): A(Tape.BLANK, move=Stay()) for tape in range(0, self.tapes)}

    def merge_transition(self, transition):
        if transition.source in self.transitions:
            if any([transition == t for t in self.transitions[transition.source]]):
                print("[+] skipping repeated transition", transition)
                return

        if transition.source not in self.transitions:
            self.transitions[transition.source] = []

        self.states.add(transition.source)
        self.states.add(transition.target)

        self.transitions[transition.source].append(transition)

        if transition.source not in self.delta:
            self.delta[transition.source] = {}

        transition_symbol = str(transition)

        if transition_symbol not in self.delta[transition.source]:
            self.delta[transition.source][transition_symbol] = set()

        self.delta[transition.source][transition_symbol].add(transition.target)

    def add_tape(self):
        if not self.tapes:
            raise RuntimeError("delta function not initialized")

        for each in self.transitions:
            for transition in self.transitions[each]:
                transition.action.actions[Tape.N(self.tapes)] = [NoneAction(on_symbol=Tape.BLANK)]

        self.tapes += 1

    def _add_transition(self, source, target, delta):
        if not self.tapes:
            self.tapes = len(delta)
        else:
            if self.tapes != len(delta):
                raise RuntimeError("invalid number of tapes (" + str(len(delta)) + ") - expected " + str(self.tapes))

        if source not in self.transitions:
            self.transitions[source] = []

        transition_symbols, actions = [], {}
        for tape in delta:
            action = delta[tape]
            actions[tape] = action.get()

            if action.symbol != Tape.BLANK:
                self.alphabet.add(action.symbol)

        transition = Transition(source=source, target=target, action=InputAction(actions=actions))

        if source in self.transitions:
            if any([transition == t for t in self.transitions[transition.source]]):
                print("[+] skipping repeated transition", transition)
                return None

        self.transitions[source].append(transition)

        transition_symbols.append(str(transition))

        return transition_symbols

# --------------------------------------------------------------------
#
# Turing machine
#
# --------------------------------------------------------------------


class TuringMachine(Automata):
    def __init__(self, transition, initial, final):
        assert isinstance(transition, TuringDelta)
        super().__init__(transition=transition, initial=initial, final=final)

    @staticmethod
    def _get_null_color():
        return "black"

    def _build_a_graph(self, a):
        a.edge_attr["minlen"] = "2"

    def is_non_deterministic(self, deltas=None):
        is_non_deterministic = False

        for each in self.transition.transitions:
            for transition in self.transition.transitions[each]:
                for other in filter(lambda t: t is not transition, self.transition.transitions[each]):

                    if all([t[0].on_symbol == o[0].on_symbol
                            for t, o in zip(transition.action.actions.values(), other.action.actions.values())]):
                        is_non_deterministic = True

                        if deltas is not None:
                            if each not in deltas:
                                deltas[each] = set()

                            deltas[each].add(transition)
                            deltas[each].add(other)

        return is_non_deterministic

    def read(self, string):
        buffer = self(Input(data=string, initial=self.initial, tapes=self.transition.tapes - 1))
        return buffer.state() in self.final and self.is_done(buffer)

    def debug_input(self, buffer):
        columns = shutil.get_terminal_size((80, 20)).columns

        size = len(str(buffer).split('\n')[0])
        right = size - (2 * len(buffer.data()) + 14)

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

        return buffer

    def debug(self, string):
        buffer = Input(data=string, initial=self.initial, tapes=self.transition.tapes - 1)
        return self.debug_input(buffer)

