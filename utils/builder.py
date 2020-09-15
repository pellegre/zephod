from pyauto.fsm import *
import pandas


class FiniteAutomataBuilder:
    def __init__(self):
        pass

    @staticmethod
    def get_finite_automata_from_csv(filename):
        frame = pandas.read_csv(filename)
        all_states = set(frame["states"].values)

        transition = Transition()
        final_states, initial_state = set(), None
        for row in range(len(frame)):
            from_state = frame.iloc[row]["states"]
            state_type = frame.iloc[row]["type"]

            for each_type in state_type.split("/"):
                # check for initial and final states
                if each_type == "initial":
                    if initial_state is not None:
                        raise RuntimeError("more than one initial state", initial_state, from_state)
                    else:
                        initial_state = from_state

                elif each_type == "final":
                    final_states.add(from_state)

                elif each_type != "none":
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
