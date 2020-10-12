from pyauto.delta import *
from pyauto.automata.base import *
from pyauto.tape import *

import shutil


# --------------------------------------------------------------------
#
# pushdown automata transitions
#
# --------------------------------------------------------------------


class PDANullTransition(Transition):
    def __init__(self, stack_action, **kwargs):
        super().__init__(**kwargs, action=PDANullAction(stack_action=stack_action))
        self.stack_action = stack_action

    def symbol(self):
        return Transition.NULL + "," + self.stack_action.on_symbol + "/" + self.stack_action.symbol()


class PDAReadTransition(Transition):
    def __init__(self, character, stack_action, **kwargs):
        super().__init__(**kwargs, action=PDAReadAction(on_symbol=character, stack_action=stack_action))
        self.character, self.stack_action = character, stack_action

    def symbol(self):
        return self.character + "," + self.stack_action.on_symbol + "/" + self.stack_action.symbol()


# --------------------------------------------------------------------
#
# action wrappers
#
# --------------------------------------------------------------------


class Push(Action):
    def __init__(self, obj, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj

    def _get_action(self, **kwargs):
        return PushStackAction(new_symbol=self.obj, **kwargs)


class Pop(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return PopStackAction(**kwargs)


class Null(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return NullStackAction(**kwargs)


# --------------------------------------------------------------------
#
# PDA delta function
#
# --------------------------------------------------------------------


class PDADelta(Delta):
    def __init__(self):
        self.stack_alphabet = set()
        super().__init__()

    def _add_transition(self, source, target, delta):
        if source not in self.transitions:
            self.transitions[source] = []

        transition_symbols = []
        for transition in delta:
            symbol = transition[0]
            stack_symbol = transition[1]

            action = delta[transition]

            if (len(stack_symbol) == 1 or stack_symbol == Stack.EMPTY) and len(symbol) == 1 and \
                    isinstance(symbol, str) and isinstance(stack_symbol, str):
                if symbol is Transition.NULL:
                    transition = PDANullTransition(source=source, target=target,
                                                   stack_action=action.get(on_symbol=stack_symbol))
                    self.transitions[source].append(transition)
                else:
                    transition = PDAReadTransition(source=source, target=target, character=symbol,
                                                   stack_action=action.get(on_symbol=stack_symbol))
                    self.transitions[source].append(transition)
            else:
                raise RuntimeError("can't handle transition " + str(transition))

            transition_symbols.append(transition.symbol())

            self.alphabet.add(transition.symbol())
            self.stack_alphabet.add(stack_symbol)

        return transition_symbols


class PushdownAutomata(Automata):
    def __init__(self, transition, initial, final):
        assert isinstance(transition, PDADelta)
        super().__init__(transition=transition, initial=initial, final=final)

    @staticmethod
    def _get_null_color():
        return "black"

    def _build_a_graph(self, a):
        a.edge_attr["minlen"] = "2"

    def read(self, string):
        buffer = self(Stack(data=string, initial=self.initial))
        return buffer.state() in self.final and self.is_done(buffer)

    def debug(self, string):
        buffer = Stack(data=string, initial=self.initial)
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
