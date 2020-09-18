import random


class Grammar:
    def __init__(self, terminal, non_terminal, start="S"):
        self.terminal, self.non_terminal = terminal, non_terminal
        self.start, self.rules = start, {}

        assert isinstance(self.terminal, set)
        assert isinstance(self.non_terminal, set)
        assert isinstance(self.start, str)

    def __str__(self):
        string = "T = " + str(self.terminal) + " ; N = " + str(self.non_terminal) + "\n"
        for l in self.rules:
            for r in self.rules[l]:
                string += l + "  -> " + r + "\n"
        return string[:-1]

    def __repr__(self):
        return self.__str__()

    def _run_random_rule(self, string, length):
        rules = [r for r in self.rules if r != self.start]
        left = rules[random.randint(0, len(rules) - 1)]

        i = string.find(left)
        while i < 0 :
            left = rules[random.randint(0, len(rules) - 1)]
            i = string.find(left)

        rule = random.randint(0, len(self.rules[left]) - 1)
        while len(string) < length and \
                all(map(lambda s: s in self.terminal, self.rules[left][rule])):
            rule = random.randint(0, len(self.rules[left]) - 1)

        return string[:i] + self.rules[left][rule] + string[i+len(left):]

    def _sanity_check(self):
        assert self.start in self.rules

    def add(self, left, right):
        assert self.start not in right

        left_sanity = list(map(lambda s: s in self.terminal or
                                         s in self.non_terminal or s == self.start, left))
        right_sanity = list(map(lambda s: s in self.terminal or
                                          s in self.non_terminal or s == self.start, right))

        assert sum(left_sanity) == len(left)
        assert sum(right_sanity) == len(right)

        if left not in self.rules:
            self.rules[left] = [right]
        else:
            self.rules[left].append(right)

    def is_regular(self):
        is_regular = all(map(lambda s: s in self.non_terminal or s == self.start and len(s) == 1, self.rules))

        for l in self.rules:
            is_regular = is_regular & \
                         all(map(lambda r: (len(r) == 1 and r in self.terminal) or
                                           (len(r) == 2 and r[0] in self.terminal and r[1] in self.non_terminal),
                                 self.rules[l]))

        return is_regular

    def is_context_free(self):
        is_context_free = all(map(lambda s: s in self.non_terminal or s == self.start and len(s) == 1, self.rules))

        return is_context_free and not self.is_regular()

    def __call__(self, length=1):
        rule = random.randint(0, len(self.rules[self.start]) - 1)
        string = self.rules[self.start][rule]

        while any(map(lambda s: s in self.non_terminal, string)):
            string = self._run_random_rule(string, length)

        return string
