from pyauto.delta import *
from pyauto.automata import *

import shutil

# --------------------------------------------------------------------
#
# stack buffer
#
# --------------------------------------------------------------------


class Stack:
    EMPTY = "Z0"


class StackBuffer(Buffer):
    def __init__(self, **kwargs):
        self.stack = list()
        super().__init__(**kwargs)

    def peek(self):
        if len(self.stack):
            return self.stack[-1]
        return Stack.EMPTY

    def pop(self):
        if len(self.stack):
            return self.stack.pop()

        raise RuntimeError("empty stack")

    def push(self, obj):
        self.stack.append(obj)

    def _state_transition(self):
        if len(self.states) > 1:
            delta_pointer = self.pointer() - self.pointers[-2]
            if delta_pointer:
                symbol = self.buffer[self.pointers[-2]]
            else:
                symbol = "$"

            return "(" + str(self.states[-2]) + ", " + symbol + ") -> " + str(self.states[-1])
        else:
            return str(self.states[-1])

    def __str__(self):
        spaces = ' '.join(['' for _ in range(16)])
        string = ' '.join(self.buffer) + spaces + \
                 "{:30}".format("state = [" + str(self._state_transition()) + "] ") + str(self.stack) + "\n"
        string += ' '.join([' ' if i != self.pointer() else '^' for i in range(self.pointer() + 1)])

        return string

    def _copy(self, **kwargs):
        stack_buffer = StackBuffer(data=self.data(), **kwargs)
        stack_buffer.stack = self.stack.copy()
        return stack_buffer

# --------------------------------------------------------------------
#
# stack actions
#
# --------------------------------------------------------------------


class StackAction:
    def __init__(self, on_symbol):
        self.on_symbol = on_symbol

    def __call__(self, buffer):
        raise RuntimeError("__call__ not implemented")

    def symbol(self):
        raise RuntimeError("symbol not implemented")


class PushAction(StackAction):
    def __init__(self, obj, **kwargs):
        self.obj = obj
        super().__init__(**kwargs)

    def __call__(self, buffer):
        buffer.push(self.obj)

    def symbol(self):
        return self.obj + self.on_symbol


class PopAction(StackAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, buffer):
        buffer.pop()

    def symbol(self):
        return NullTransition.SYMBOL


class NullAction(StackAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, buffer):
        pass

    def symbol(self):
        return self.on_symbol

# --------------------------------------------------------------------
#
# action wrappers
#
# --------------------------------------------------------------------


class Action:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _get_action(self, **kwargs):
        raise RuntimeError("_get_action not implemented")

    def get(self, **kwargs):
        return self._get_action(**self.kwargs, **kwargs)


class Push(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return PushAction(**kwargs)


class Pop(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return PopAction(**kwargs)


class Null(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_action(**kwargs):
        return NullAction(**kwargs)

# --------------------------------------------------------------------
#
# transition function
#
# --------------------------------------------------------------------


class StackCharTransition(Transition):
    def __init__(self, character, action, **kwargs):
        self.character = character
        self.action = action
        super().__init__(**kwargs)

    def _consume(self, tape):
        if len(tape.head()) and \
                tape.head()[0] == self.character and \
                tape.peek() == self.action.on_symbol:

            self.action(tape)
            tape.read(self.target, 1)
        else:
            tape.error = True

    def symbol(self):
        return self.character + "," + self.action.on_symbol + "/" + self.action.symbol()


class StackNullTransition(Transition):
    def __init__(self, action, **kwargs):
        self.action = action
        super().__init__(**kwargs)

    def _consume(self, tape):
        if tape.peek() == self.action.on_symbol:
            self.action(tape)
            tape.read(self.target, 0)
        else:
            tape.error = True

    def symbol(self):
        return NullTransition.SYMBOL + "," + self.action.on_symbol + "/" + self.action.symbol()


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
                if symbol is NullTransition.SYMBOL:
                    transition = StackNullTransition(source=source, target=target,
                                                     action=action.get(on_symbol=stack_symbol))
                    self.transitions[source].append(transition)
                else:
                    transition = StackCharTransition(source=source, target=target, character=symbol,
                                                     action=action.get(on_symbol=stack_symbol))
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
        buffer = self(StackBuffer(data=string, initial=self.initial))
        return buffer.state() in self.final and not len(buffer.head())

    def debug(self, string):
        buffer = StackBuffer(data=string, initial=self.initial)
        columns = shutil.get_terminal_size((80, 20)).columns

        size = len(str(buffer).split('\n')[0])
        right = size - (2 * len(string) + 14)

        print()
        print(("{:" + str(size-right) + "}").format("initial (" + str(self.initial) + ")") +
              ("{:" + str(right) + "}").format("final " + str(self.final) + ""))
        print('='.join(['' for _ in range(columns)]))

        print()
        print(buffer)
        print()
        buffer = self(buffer, debug=True)

        accepted = buffer.state() in self.final and not len(buffer.head())
        print()
        print("{:25}".format("accepted ---->  (" + str(accepted) + ")"))
        print()
