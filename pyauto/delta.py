import re

# --------------------------------------------------------------------
#
# input buffer
#
# --------------------------------------------------------------------


class Input:
    def __init__(self, initial):
        self.states = [State(initial)]
        self.pointers = [0]

        self.done = False

    def __str__(self):
        string = "(pointers = " + str(self.pointers) + " , states = " + str(self.states) + ") @ "
        string += "(done = " + str(self.done) + ") # "
        if not len(self.head()):
            string += "data = " + str(self.data()) + "\n"
        else:
            string += "head = " + str(self.head()) + " (of " + str(self.data()) + ")\n"
        return string

    def __repr__(self):
        return self.__str__()

    def _get_data_from_pointer(self, pointer):
        raise RuntimeError("_get_data_from_pointer not implemented")

    def _check_consumed(self):
        raise RuntimeError("_check_done not implemented")

    def _copy(self, **kwargs):
        raise RuntimeError("_check_done not implemented")

    def copy(self):
        obj = self._copy(initial=self.states[0])
        obj.states = self.states.copy()
        obj.pointers = self.pointers.copy()
        obj.done = self.done

        return obj

    def data(self):
        raise RuntimeError("data not implemented")

    def head(self):
        return self._get_data_from_pointer(self.pointer())

    def read(self, state, count):
        assert not self.done

        self.states.append(state)
        self.pointers.append(self.pointer() + count)

        assert len(self.states) == len(self.pointers)

    def state(self):
        return self.states[-1]

    def pointer(self):
        return self.pointers[-1]


class Buffer(Input):
    def __init__(self, data, **kwargs):
        self.buffer = data
        super().__init__(**kwargs)

    def data(self):
        return self.buffer

    def _copy(self, **kwargs):
        return Buffer(data=self.data(), **kwargs)

    def _get_data_from_pointer(self, pointer):
        return self.buffer[pointer:]

# --------------------------------------------------------------------
#
# consumer
#
# --------------------------------------------------------------------


class Consumer:
    def __init__(self, source, target):
        self.source = State(source)
        self.target = State(target)

    def _consume(self, tape):
        raise RuntimeError("_check_done not implemented")

    def __call__(self, tape):
        if tape.state() == self.source:
            self._consume(tape)
        else:
            tape.done = True

        return tape


class CharConsumer(Consumer):
    def __init__(self, character, **kwargs):
        self.character = character
        super().__init__(**kwargs)

    def _consume(self, tape):
        if len(tape.head()) and tape.head()[0] == self.character:
            tape.read(self.target, 1)
        else:
            tape.done = True


class NullConsumer(Consumer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _consume(self, tape):
        tape.read(self.target, 0)


# --------------------------------------------------------------------
#
# delta transition function
#
# --------------------------------------------------------------------


class State:
    def __init__(self, state):
        if isinstance(state, str):
            groups = re.search("([a-zA-Z]+)([0-9]+)", state).groups()
            self.prefix = groups[0]
            self.number = int(groups[1])
        else:
            self.prefix = state.prefix
            self.number = state.number

        # self.name = "$" + self.prefix + "_" + str(self.number) + "$"
        self.name = self.prefix + str(self.number)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __lt__(self, other):
        assert isinstance(other, State)
        return self.number < other.number

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class Transition:
    EPSILON = "$"

    def __init__(self):
        self.states = set()
        self.alphabet = {Transition.EPSILON}
        self.delta, self.consumers = {}, {}

    def add(self, ei, ef, symbols):
        assertion = [isinstance(s, str) for s in symbols]
        assert sum(assertion) == len(assertion)

        source, target = State(ei), State(ef)

        self.states.add(source)
        self.states.add(target)
        self.alphabet.update(symbols)

        if source not in self.delta:
            self.delta[source] = {}
            self.consumers[source] = []

        for s in symbols:
            if s not in self.delta[source]:
                self.delta[source][s] = set()

            self.delta[source][s].add(target)

            if s == Transition.EPSILON:
                self.consumers[source].append(NullConsumer(source=source, target=target))
            else:
                self.consumers[source].append(CharConsumer(character=s, source=source, target=target))

    def join(self, other):
        for ei in other.delta:
            for symbol in other.delta[ei]:
                for ef in other.delta[ei][symbol]:
                    self.add(ei, ef, symbol)

    def __call__(self, tape):
        if tape.state() not in self.consumers:
            parsed = tape.copy()
            parsed.done = True
            return {parsed}
        else:
            consumers = self.consumers[tape.state()]

            parsed = set()
            for consumer in consumers:
                parsed.add(consumer(tape.copy()))

            if not len(parsed):
                done_buffer = tape.copy()
                done_buffer.done = True

                return {done_buffer}

            return parsed

    def __str__(self):
        string = "states = " + str(self.states) + " ; alphabet = " + str(self.alphabet) + "\n"
        for e in self.delta:
            for s in self.delta[e]:
                string += str(e) + " (" + str(s) + ") -> " + str(self.delta[e][s]) + "\n"
        return string[:-1]

    def __repr__(self):
        return self.__str__()

    def max_state(self):
        return max(self.states, key=lambda e: e.number)

    def rebase(self, base):
        transition = Transition()

        for ei in self.delta:
            new_ei = ei.prefix + str(ei.number + base)
            for symbol in self.delta[ei]:
                for ef in self.delta[ei][symbol]:
                    new_ef = ef.prefix + str(ef.number + base)
                    transition.add(new_ei, new_ef, symbol)

        return transition
