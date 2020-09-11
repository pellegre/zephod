
class Transition:
    def __init__(self):
        self.states = set()
        self.alphabet = set()
        self.delta = {}

    def add(self, ei, ef, symbols):
        self.states.add(ei)
        self.states.add(ef)
        self.alphabet.update(symbols)

        if ei not in self.delta:
            self.delta[ei] = {}

        for s in symbols:
            if s not in self.delta[ei]:
                self.delta[ei][s] = set()

            self.delta[ei][s].add(ef)

    def __str__(self):
        string = "states = " + str(self.states) + " ; alphabet = " + str(self.alphabet) + "\n"
        for e in self.delta:
            for s in self.delta[e]:
                string += e + " (" + str(s) + ") -> " + str(self.delta[e][s]) + "\n"
        return string[:-1]

    def __repr__(self):
        return self.__str__()


class FiniteAutomata:
    def __init__(self, transition, initial, final):
        assert isinstance(transition.states, set)
        assert isinstance(transition.alphabet, set)
        assert isinstance(final, set) and final.issubset(transition.states)
        assert not isinstance(initial, set) and initial in transition.states

        self.transition = transition
        self.initial = initial
        self.final = final

    def __str__(self):
        string = str(self.transition)
        string += "\ninitial = " + str(self.initial) + " ; final = " + str(self.final)
        return string

    def __repr__(self):
        return self.__str__()
