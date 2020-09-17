from pyauto.fsm import *

import pandas
import random
import string
import subprocess
import itertools


class AutomataPlotter:
    def __init__(self):
        pass

    @staticmethod
    def get_tmp_filename():
        return "/tmp/" + "".join(random.choice(string.ascii_letters) for _ in range(12))

    @staticmethod
    def plot(z):
        dot = z.build_dot()
        filename = AutomataPlotter.get_tmp_filename() + ".pdf"
        dot.draw(path=filename)
        subprocess.Popen(["xdg-open " + filename], shell=True)


class Constraint:
    def __init__(self, pattern):
        self.pattern = pattern


class NotContained(Constraint):
    def __init__(self, pattern):
        super().__init__(pattern)
        self.read_none = "read_none"
        self.conditions = ["read_" + str(i) + "_" + e for i, e in enumerate(self.pattern[:-1])]

    def get_name(self):
        return self.pattern + "_not_contained"

    def initial(self, condition):
        return self.read_none == condition

    @staticmethod
    def final(condition):
        return True

    def get_conditions(self):
        # first should always be default
        return [self.read_none] + self.conditions

    def next_condition(self, current, symbol):
        if current == self.read_none and symbol == self.pattern[0]:
            return self.conditions[0]
        elif current == self.conditions[-1] and symbol == self.pattern[-1]:
            return "err"

        for i in range(len(self.conditions) - 1):
            if current == self.conditions[i] and symbol == self.pattern[i + 1]:
                return self.conditions[i + 1]

        return self.read_none


class NotEnd(NotContained):
    def __init__(self, pattern):
        super().__init__(pattern)
        self.read_none = "read_none"
        self.conditions = ["read_" + str(i) + "_" + e for i, e in enumerate(self.pattern)]

    def get_name(self):
        return self.pattern + "_not_end"

    def initial(self, condition):
        return self.read_none == condition

    def final(self, condition):
        return condition != self.conditions[-1]

    def get_conditions(self):
        # first should always be default
        return [self.read_none] + self.conditions

    def next_condition(self, current, symbol):
        if current == self.read_none and symbol == self.pattern[0]:
            return self.conditions[0]

        for i in range(len(self.conditions) - 1):
            if current == self.conditions[i] and symbol == self.pattern[i + 1]:
                return self.conditions[i + 1]

        if len(set(self.pattern)) == 1 and current == self.conditions[-1] and symbol == self.pattern[0]:
            return self.conditions[-1]

        return self.read_none


class Parity(Constraint):
    def __init__(self, pattern):
        super().__init__(pattern)
        self.even_condition = "even_" + self.pattern
        self.odd_condition = "odd_" + self.pattern

    def get_name(self):
        return self.pattern + "_parity"

    def initial(self, condition):
        return self.even_condition == condition

    def get_conditions(self):
        # first should always be default
        return [self.even_condition, self.odd_condition]

    def next_condition(self, current, symbol):
        if current == self.even_condition and symbol == self.pattern:
            return self.odd_condition
        elif current == self.odd_condition and symbol == self.pattern:
            return self.even_condition
        return current


class Odd(Parity):
    def __init__(self, pattern):
        super().__init__(pattern)

    def final(self, condition):
        return self.odd_condition == condition


class Even(Parity):
    def __init__(self, pattern):
        super().__init__(pattern)

    def final(self, condition):
        return self.even_condition == condition


class Unit:
    def __init__(self, alphabet, constraints, name):
        self.alphabet = set(sorted(alphabet))
        self.constraints = constraints

        if isinstance(name, str):
            self.units = {name: self.constraints}
        else:
            assert isinstance(name, dict)
            self.units = name

    def __or__(self, other):
        alphabet = set(list(self.alphabet) + list(other.alphabet))
        constraints = self.constraints + other.constraints

        return Unit(alphabet, constraints, name={**self.units, **other.units})

    def get_frame(self, total=False):
        columns, names, conditions = ["states"], [], []

        for each in self.constraints:
            columns.append(each.get_name())
            names.append(each.get_name())
            conditions.append(each.get_conditions())

        for each in self.alphabet:
            columns.append(each)

        frame = pandas.DataFrame(columns=columns)
        for i, each in enumerate(list(itertools.product(*conditions))):
            row = pandas.DataFrame([["z" + str(i)] + list(each) + [pandas.NA] * len(self.alphabet)], columns=columns)

            initial = [c.initial(v) for c, v in zip(self.constraints, row[names].values[0])]
            final = [c.final(v) for c, v in zip(self.constraints, row[names].values[0])]
            if sum(initial) == len(self.constraints) and sum(final) == len(self.constraints):
                row["type"] = FiniteAutomata.NodeType.INITIAL + "/" + FiniteAutomata.NodeType.FINAL
            elif sum(initial) == len(self.constraints):
                row["type"] = FiniteAutomata.NodeType.INITIAL
            elif sum(final) == len(self.constraints):
                row["type"] = FiniteAutomata.NodeType.FINAL
            else:
                row["type"] = FiniteAutomata.NodeType.NONE

            frame = pandas.concat([frame, row])

        for i in range(len(frame)):
            for symbol in self.alphabet:
                next_conditions = [each.next_condition(frame.iloc[i][each.get_name()], symbol)
                                   for each in self.constraints]
                matching_state = frame[frame[names].apply(lambda x:
                                                          list(x) == next_conditions, axis=1)]["states"].values
                if not len(matching_state):
                    frame.iloc[i][symbol] = "err"
                elif len(matching_state) == 1:
                    frame.iloc[i][symbol] = matching_state[0]
                else:
                    raise RuntimeError("more than 1 matching state", matching_state, symbol)

        if total:
            return frame

        return frame[["states", "type"] + list(self.alphabet)]


class FiniteAutomataBuilder:
    def __init__(self):
        pass

    @staticmethod
    def get_finite_automata_from_csv(filename):
        frame = pandas.read_csv(filename)
        return FiniteAutomataBuilder.get_finite_automata_from_frame(frame)

    @staticmethod
    def get_finite_automata_from_frame(frame):
        all_states = set(frame["states"].values)

        transition = Transition()
        final_states, initial_state = set(), None
        for row in range(len(frame)):
            from_state = frame.iloc[row]["states"]
            state_type = frame.iloc[row]["type"]

            for each_type in state_type.split("/"):
                # check for initial and final states
                if each_type == FiniteAutomata.NodeType.INITIAL:
                    if initial_state is not None:
                        raise RuntimeError("more than one initial state", initial_state, from_state)
                    else:
                        initial_state = from_state

                elif each_type == FiniteAutomata.NodeType.FINAL:
                    final_states.add(from_state)

                elif each_type != FiniteAutomata.NodeType.NONE:
                    raise RuntimeError("invalid state type", each_type)

            for symbol in frame.iloc[row].index:
                if symbol not in ["states", "type"]:
                    state = frame.iloc[row][symbol]
                    if state in all_states:
                        transition.add(from_state, state, {symbol})
                    else:
                        if state != "err":
                            print("warning :", state, "not defined state")

        dfsm = FiniteAutomata(transition, initial_state, final_states)
        return dfsm

    @staticmethod
    def get_finite_automata_from_unit(unit):
        return FiniteAutomataBuilder.get_finite_automata_from_frame(unit.get_frame())
