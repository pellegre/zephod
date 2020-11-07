import re
import random


# --------------------------------------------------------------------
#
# transition
#
# --------------------------------------------------------------------


class Transition:
    NULL = "$"

    def __init__(self, source, target, action):
        self.source = source
        self.target = target
        self.action = action

    def __call__(self, tape):
        return self.action(self.source, self.target, tape)

    def symbol(self):
        raise RuntimeError("symbol not implemented")

    def __str__(self):
        return str(self.source) + " -> " + str(self.target) + " with " + str(self.action)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return str(self.source) == str(other.source) and str(self.target) == str(other.target) and \
               str(self.action) == str(other.action)

    def __hash__(self):
        return self.__str__().__hash__()

# --------------------------------------------------------------------
#
# state
#
# --------------------------------------------------------------------


class State:
    def __init__(self, state):
        if isinstance(state, str):
            groups = re.search("([a-zA-Z_$]*)([0-9]*[$]*)", state).groups()
            self.prefix = groups[0]
            if len(groups[1]):
                self.number = int(groups[1])
            else:
                self.number = random.randint(0, 100)
        else:
            self.prefix = state.prefix
            self.number = state.number

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


class ErrorState(State):
    def __init__(self):
        # the devil is in the details
        super().__init__(state="read_error_696969")

# --------------------------------------------------------------------
#
# delta transition function
#
# --------------------------------------------------------------------


class Delta:
    def __init__(self):
        self.states = set()
        self.alphabet = set()
        self.delta, self.transitions = {}, {}

        # state management
        self.state_counter = 0
        self.state_description = {}

    def get_new_state(self, prefix, description):
        self.state_counter += 1

        new_state = State(prefix + str(self.state_counter))

        self.state_description[new_state] = description

        return new_state

    def _add_transition(self, source, target, symbols):
        raise RuntimeError("_add_transition not implemented")

    def add(self, ei, ef, *args, **kwargs):
        source, target = State(ei), State(ef)

        transition_symbols = self._add_transition(source, target, *args, **kwargs)

        if transition_symbols is not None:
            self.states.add(source)
            self.states.add(target)

            if source not in self.delta:
                self.delta[source] = {}

            for s in transition_symbols:
                if s not in self.delta[source]:
                    self.delta[source][s] = set()

                self.delta[source][s].add(target)

    def join(self, other):
        for ei in other.delta:
            for symbol in other.delta[ei]:
                for ef in other.delta[ei][symbol]:
                    self.add(ei, ef, symbol)

    def __call__(self, tape):
        if tape.state() not in self.transitions:
            parsed = tape.copy()
            parsed.error = True
            return {parsed}
        else:
            consumers = self.transitions[tape.state()]
            parsed = set()
            for consumer in consumers:
                consumed = consumer(tape)

                if isinstance(consumed, set):
                    parsed.update(consumed)
                else:
                    parsed.add(consumed)

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
        transition = type(self)()

        for ei in self.delta:
            new_ei = ei.prefix + str(ei.number + base)
            for symbol in self.delta[ei]:
                for ef in self.delta[ei][symbol]:
                    new_ef = ef.prefix + str(ef.number + base)
                    transition.add(new_ei, new_ef, symbol)

        return transition
