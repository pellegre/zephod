from pyauto.delta import *
from pyauto.automata.base import *
from pyauto.tape import *

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
    def __init__(self):
        super().__init__()
        self.tapes = None

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

            self.alphabet.add(action.symbol)

        transition = Transition(source=source, target=target, action=InputAction(actions=actions))
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

    def read(self, string):
        buffer = self(Input(data=string, initial=self.initial, tapes=self.transition.tapes - 1))
        return buffer.state() in self.final and self.is_done(buffer)

    def debug(self, string):
        buffer = Input(data=string, initial=self.initial, tapes=self.transition.tapes - 1)
        columns = shutil.get_terminal_size((80, 20)).columns

        size = len(str(buffer).split('\n')[0])
        right = size - (2 * len(string) + 14)

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