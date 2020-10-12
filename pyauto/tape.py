class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --------------------------------------------------------------------
#
# tape object
#
# --------------------------------------------------------------------


class Tape:
    BLANK = "B0"

    def __init__(self, name, data=None, pointer=0):
        self.name = name

        if data:
            self.buffer = [c for c in data]
        else:
            self.buffer = [Tape.BLANK]

        self.previous = self.buffer.copy()
        self.pointers = [pointer]
        self.last_action = None

    def copy(self):
        tape = Tape(name=self.name)
        tape.buffer = self.buffer.copy()
        tape.pointers = self.pointers.copy()
        tape.previous = self.previous.copy()

        return tape

    def data(self):
        return ''.join(self.buffer)

    def pointer(self):
        return self.pointers[-1]

    def right(self):
        self.previous = self.buffer.copy()
        self.pointers.append(self.pointer() + 1)

        if self.pointer() == len(self.buffer):
            self.buffer.append(Tape.BLANK)
            self.previous.append(Tape.BLANK)

    def left(self):
        self.previous = self.buffer.copy()
        if not self.pointer():
            raise RuntimeError("can't go below zero on tape")

        self.pointers.append(self.pointer() - 1)

    def none(self):
        self.previous = self.buffer.copy()
        self.pointers.append(self.pointer())

    def read(self):
        self.previous = self.buffer.copy()
        return self.buffer[self.pointer()]

    def write(self, character):
        self.previous = self.buffer.copy()
        self.buffer[self.pointer()] = character

    def _color_print(self, i):
        if self.buffer[i] is Tape.BLANK:
            character = '_'
        else:
            character = self.buffer[i]

        if i == self.pointer():
            return Colors.BOLD + Colors.YELLOW + character + Colors.END

        return Colors.BOLD + character + Colors.END

    def __str__(self):
        output = Colors.BLUE + Colors.BOLD + '[' + self.name + '] ' + Colors.END + \
                 ' '.join([self._color_print(i) for i in range(len(self.buffer))])

        return output

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def N(number):
        return "T" + str(number)


# --------------------------------------------------------------------
#
# input object (input data is on the first tape, and might have
# additional ones)
#
# --------------------------------------------------------------------

class Input:
    def __init__(self, initial, pointer=0, data=None, tapes=0):
        self.initial = initial

        self.states = [initial]
        self.tapes = {Tape.N(0): Tape(name=Tape.N(0), data=data, pointer=pointer)}

        for tape in range(tapes):
            tape_name = "T" + str(tape + 1)
            self.tapes[tape_name] = Tape(name=tape_name)

        self.error = False
        self.error_action = None

    def copy(self, input_type=None):
        if not input_type:
            input_type = Input

        inp = input_type(initial=self.initial)
        inp.states = self.states.copy()
        inp.tapes = {}

        for tape in self.tapes:
            inp.tapes[tape] = self.tapes[tape].copy()

        inp.error = self.error

        return inp

    def state(self):
        return self.states[-1]

    def data(self, tape=Tape.N(0)):
        return self.tapes[tape].data()

    def pointer(self, tape=Tape.N(0)):
        return self.tapes[tape].pointer()

    def head(self, tape=Tape.N(0)):
        return self.tapes[tape].read()

    @staticmethod
    def _print_symbol(symbol):
        if symbol == Tape.BLANK:
            return "_"

        return symbol

    @staticmethod
    def _print_coordinates(symbol, ptr):
        return Input._print_symbol(symbol) + "(" + str(ptr) + ")"

    def _state_print(self):
        if len(self.states) > 1:
            from_symbol, to_symbol = str(), str()

            for t in self.tapes:
                tape = self.tapes[t]
                from_symbol += " " + self._print_coordinates(tape.last_action.read_symbol, tape.pointers[-2]) + ","
                to_symbol += " " + self._print_coordinates(tape.read(), tape.pointers[-1]) + ","

            return "[" + str(self.states[-2]) + "," + from_symbol[:-1] + "] to " + \
                   "[" + str(self.states[-1]) + "," + to_symbol[:-1] + "]"
        else:
            return str(self.states[-1])

    def __str__(self):
        spaces = ' '.join(['' for _ in range(16)])

        error_state, state = str(), str()
        if self.error and not self.error_action:
            error_state = Colors.BOLD + Colors.RED + "bad state" + Colors.END

        elif self.error:
            action = self.error_action[1]
            error_state = Colors.BOLD + Colors.RED + str(action) + " - head was at " + action.read_symbol + Colors.END

        else:
            state = Colors.BOLD + Colors.GREEN + "{:30}".format(self._state_print()) + Colors.END

        output = str(self.tapes[Tape.N(0)]) + spaces + state + error_state + "\n"

        for tape in filter(lambda t: t != Tape.N(0), self.tapes):
            output += str(self.tapes[tape]) + "\n"

        return output[:-1]

    def __repr__(self):
        return self.__str__()

# --------------------------------------------------------------------
#
# tape actions
#
# --------------------------------------------------------------------


class TapeAction:
    def __init__(self, on_symbol):
        self.on_symbol = on_symbol
        self.read_symbol = None

    def action(self, tape: Tape):
        raise RuntimeError("action not implemented")

    def peek_tape(self, tape):
        return tape.read()

    def __call__(self, tape):
        tape.last_action = self
        self.read_symbol = self.peek_tape(tape)

        if not self.on_symbol:
            self.action(tape)
            return True

        if self.read_symbol == self.on_symbol:
            self.action(tape)
            return True

        return False

    def __str__(self):
        output = "(on = " + self.on_symbol + ")"
        return output


class NoneAction(TapeAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action(self, tape: Tape):
        tape.none()

    def __str__(self):
        output = "none " + super().__str__()
        return output


class WriteAction(TapeAction):
    def __init__(self, new_symbol, **kwargs):
        super().__init__(**kwargs)
        self.new_symbol = new_symbol

    def action(self, tape: Tape):
        tape.write(self.new_symbol)

    def __str__(self):
        output = "write(" + self.new_symbol + ") " + super().__str__()
        return output

    def __repr__(self):
        return self.__str__()


class MoveRightAction(TapeAction):
    def __init__(self, moves=1, **kwargs):
        super().__init__(**kwargs)
        self.moves = moves

    def action(self, tape: Tape):
        for move in range(self.moves):
            tape.right()

    def __str__(self):
        output = "move right " + super().__str__()
        return output


class MoveLeftAction(TapeAction):
    def __init__(self, moves=1, **kwargs):
        super().__init__(**kwargs)
        self.moves = moves

    def action(self, tape: Tape):
        for move in range(self.moves):
            tape.left()

    def __str__(self):
        output = "move left " + super().__str__()
        return output

# --------------------------------------------------------------------
#
# input action (collection of tape's actions)
#
# --------------------------------------------------------------------


class InputAction:
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, source, target, data: Input):
        if len(self.actions) != len(data.tapes):
            raise RuntimeError("invalid set of actions for input")

        consumed = data.copy(type(data))
        consumed.last_action = self

        if source != consumed.state():
            consumed.error = True
            consumed.error_action = None
            return consumed

        else:
            for tape in consumed.tapes:
                for action in self.actions[tape]:
                    good = action(consumed.tapes[tape])

                    if not good:
                        consumed.error = True
                        consumed.error_action = (tape, action)
                        return consumed

        if not consumed.error:
            consumed.states.append(target)

        return consumed


# --------------------------------------------------------------------
#
# stack object (input with 2 tapes)
#
# --------------------------------------------------------------------

class Stack(Input):
    EMPTY = "Z0"
    NULL = "$"

    def __init__(self, initial, data=None):
        super().__init__(initial=initial, data=data, tapes=1)

    @staticmethod
    def peek_tape(tape):
        if tape.pointer() == 0:
            assert tape.read() is Tape.BLANK
            return Stack.EMPTY

        tape.left()
        top = tape.read()
        tape.right()

        return top

    @staticmethod
    def pop(tape):
        tape.left()
        tape.write(Tape.BLANK)

    @staticmethod
    def push(tape, symbol):
        tape.write(symbol)
        tape.right()

    def peek(self):
        return Stack.peek_tape(self.tapes[Tape.N(1)])

    def data(self):
        return self.tapes[Tape.N(0)].data()

    def pointer(self):
        return self.tapes[Tape.N(0)].pointer()

    def head(self):
        return self.tapes[Tape.N(0)].read()


# --------------------------------------------------------------------
#
# stack actions
#
# --------------------------------------------------------------------


class StackAction(TapeAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def peek_tape(self, tape):
        return Stack.peek_tape(tape)


class PopStackAction(StackAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action(self, tape: Tape):
        Stack.pop(tape)

    def __str__(self):
        output = "pop " + super().__str__()
        return output

    @staticmethod
    def symbol():
        return Stack.NULL


class PushStackAction(StackAction):
    def __init__(self, new_symbol, **kwargs):
        super().__init__(**kwargs)
        self.new_symbol = new_symbol

    def action(self, tape: Tape):
        Stack.push(tape, self.new_symbol)

    def __str__(self):
        output = "push(" + self.new_symbol + ") " + super().__str__()
        return output

    def __repr__(self):
        return self.__str__()

    def symbol(self):
        return self.new_symbol + self.on_symbol


class NullStackAction(StackAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action(self, tape: Tape):
        tape.none()

    def __str__(self):
        output = "none " + super().__str__()
        return output

    def symbol(self):
        return self.on_symbol


class PDAReadAction(InputAction):
    def __init__(self, on_symbol, stack_action):
        self.on_symbol = on_symbol
        self.stack_action = stack_action

        assert (isinstance(self.stack_action, PopStackAction) or
                isinstance(self.stack_action, PushStackAction) or
                isinstance(self.stack_action, NullStackAction))

        super().__init__(actions={
            Tape.N(0): [MoveRightAction(on_symbol=self.on_symbol)],
            Tape.N(1): [self.stack_action]
        })


class PDANullAction(InputAction):
    def __init__(self, stack_action):
        self.stack_action = stack_action

        assert (isinstance(self.stack_action, PopStackAction) or
                isinstance(self.stack_action, PushStackAction) or
                isinstance(self.stack_action, NullStackAction))

        super().__init__(actions={
            Tape.N(0): [NoneAction(on_symbol=None)],
            Tape.N(1): [self.stack_action]
        })

# --------------------------------------------------------------------
#
# buffer object (input with a single tape)
#
# --------------------------------------------------------------------


class Buffer(Input):
    def __init__(self, initial, data=None, pointer=0):
        super().__init__(initial=initial, data=data, pointer=pointer)

    def data(self):
        return self.tapes[Tape.N(0)].data()

    def pointer(self):
        return self.tapes[Tape.N(0)].pointer()

    def head(self):
        return self.tapes[Tape.N(0)].read()

# --------------------------------------------------------------------
#
# buffer actions
#
# --------------------------------------------------------------------


class FAReadAction(InputAction):
    def __init__(self, on_symbol):
        self.on_symbol = on_symbol

        super().__init__(actions={
            Tape.N(0): [MoveRightAction(on_symbol=self.on_symbol)]
        })


class FANullReadAction(InputAction):
    def __init__(self):

        super().__init__(actions={
            Tape.N(0): [NoneAction(on_symbol=None)]
        })

# --------------------------------------------------------------------
#
# action wrapper
#
# --------------------------------------------------------------------


class Action:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _get_action(self, **kwargs):
        raise RuntimeError("_get_action not implemented")

    def get(self, **kwargs):
        return self._get_action(**self.kwargs, **kwargs)
