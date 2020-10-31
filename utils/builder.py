import pandas
import random
import string
import subprocess

from pyauto.automata.finite import *
from pyauto.automata.turing import *


class TuringPlotter:
    def __init__(self):
        pass

    @staticmethod
    def resolve_move(move):
        if isinstance(move, MoveLeftAction):
            return "I"
        elif isinstance(move, MoveRightAction):
            return "D"
        else:
            return "N"

    @staticmethod
    def resolve_symbol(symbol):
        if symbol == Tape.BLANK:
            return "B"

        return symbol

    @staticmethod
    def table(machine: TuringMachine):
        delta = machine.transition

        columns = ["state"]
        columns += [Tape.N(i) for i in range(delta.tapes)]

        for i in range(delta.tapes):
            columns += [Tape.N(i) + "-NS", Tape.N(i) + "-M"]

        columns += ["new state"]

        frame = pandas.DataFrame(columns=columns)

        for state in delta.transitions:
            for transition in delta.transitions[state]:
                row = [state]

                for tape in range(delta.tapes):
                    action = transition.action.actions[Tape.N(tape)][0]
                    row.append(TuringPlotter.resolve_symbol(action.on_symbol))

                for tape in range(delta.tapes):
                    actions = transition.action.actions[Tape.N(tape)]

                    if len(actions) > 1:
                        write_action = actions[0]
                        move_action = actions[1]

                        row.append(TuringPlotter.resolve_symbol(write_action.new_symbol))
                        row.append(TuringPlotter.resolve_move(move_action))

                    else:
                        action = actions[0]

                        row.append(TuringPlotter.resolve_symbol(action.on_symbol))
                        row.append(TuringPlotter.resolve_move(action))

                row.append(transition.target)

                row = pandas.DataFrame([row], columns=columns)
                frame = pandas.concat([frame, row])

        return frame

    @staticmethod
    def to_csv(filename, machine: TuringMachine):
        frame = TuringPlotter.table(machine)
        frame.to_csv(filename, index=False)

        print("[+] wrote to", filename)


class AutomataPlotter:
    def __init__(self):
        pass

    @staticmethod
    def get_tmp_filename():
        return "/tmp/" + "".join(random.choice(string.ascii_letters) for _ in range(12))

    @staticmethod
    def plot(z, labels=False, layout="dot"):
        dot = z.build_dot(labels, layout=layout)
        filename = AutomataPlotter.get_tmp_filename() + ".pdf"
        dot.draw(path=filename)
        subprocess.Popen(["xdg-open " + filename], shell=True)

    @staticmethod
    def tikz(z, filename, output, labels=False, layout="dot"):
        dot = z.build_dot(tex=True, labels=labels, layout=layout)
        dot_file = "/tmp/" + filename + ".dot"
        tex_file = "/tmp/" + filename + ".tex"

        with open(dot_file, "w") as f:
            f.write(dot.to_string())
            subprocess.Popen(["dot2tex --crop -ftikz " + dot_file + " > " +
                              tex_file + " && pdflatex --output-directory " + output + " " + tex_file], shell=True)

            f.close()


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

        transition = FADelta()
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
