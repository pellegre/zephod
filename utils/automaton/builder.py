from utils.automaton.planner import *

# --------------------------------------------------------------------
#
# machine builder (from a plan)
#
# --------------------------------------------------------------------


class MachineBuilder:
    def __init__(self, planner, prefix="z"):
        self.planner, self.language = planner, planner.language

        self.transition = TuringDelta(tapes=len(self.planner.tapes) + 1)

        self.prefix, self.initial_state = prefix, State(prefix + "0")

        self.minimal = self._get_minimum_indices()

        if not len(self.minimal):
            raise RuntimeError("inconsistent initial conditions, check for negative values")

        self.block_from_state, self.state_from_block = {}, {}
        for i in range(0, len(self.language.expression)):
            state = self._get_new_state(description="state for parsing expression block " + str(i) +
                                                    " = " + str(self.language.expression[i]))
            self.block_from_state[state] = i
            self.state_from_block[i] = state

        self.parsed_state = self._get_new_state(description="state after parsing all blocks")

        self.state_from_block[TuringPlanner.END_BLOCK] = self.parsed_state

        for block in self.planner.explore_exit_blocks(0):
            state = self.state_from_block[block]

            next_symbol = self._get_symbol_for_state(state)

            delta = self._get_blank_delta()

            delta[T(0)] = A(next_symbol, move=Stay())
            for tape in self.planner.tapes:
                delta[T(tape)] = A(Tape.BLANK, new="X", move=Right())

            self.transition.add(self.initial_state, state, delta)

        self.final_state = self._build_planner(self.initial_state)

        self.turing = TuringMachine(initial=self.initial_state, transition=self.transition,
                                    final={self.final_state})

    def info(self):
        print("[+] expression", self.language.expression)
        print("[+] minimal", self.minimal)
        print("[+] block state", self.block_from_state)
        print("[+] final state", self.final_state)
        print("[+] conditions", self.language.conditions)

        print("[+] state description")
        for state in self.transition.state_description:
            print("[+] state " + str(state) + " -> " + self.transition.state_description[state])

        print("[*] turing machine")
        print(self.turing.transition)

        if self.turing.is_non_deterministic():
            non_deterministic_deltas = dict()
            print("[+] non deterministic :", self.turing.is_non_deterministic(deltas=non_deterministic_deltas))
            if len(non_deterministic_deltas):
                for state in non_deterministic_deltas:
                    for each in non_deterministic_deltas[state]:
                        print("[+] -", each)

        else:
            print("[*] the turing machine is deterministic :)")

        self.planner.info()

    def _build_planner(self, initial_state):
        next_state = initial_state

        for plan in self.planner.machine_plan:
            if isinstance(plan, BlockPlan):
                current_block = plan.block
                state = self.state_from_block[current_block]

                final_states = {self.state_from_block[block]: self._get_symbol_for_block(block)
                                for block in self.planner.exit_blocks[current_block]}

                word = self._get_word_for_block(current_block)

                plan(self.transition, state, final_states, word)

                next_state = self.parsed_state

            elif isinstance(plan, OperationPlan) or isinstance(plan, RewindLeft) or isinstance(plan, WipeTapes):
                next_state = plan(self.transition, next_state)

            else:
                raise RuntimeError("unhandled plan " + str(plan))

        return next_state

    def _get_symbol_for_state(self, state):
        if state == self.parsed_state:
            return Tape.BLANK

        word = self._get_word_for_state(state)

        return word[0]

    def _get_word_for_state(self, state):
        block = self.block_from_state[state]
        return self._get_word_for_block(block)

    def _get_word_for_block(self, block):
        expr = self.language.expression[block]

        if isinstance(expr, Pow):
            assert isinstance(expr.exp, Symbol)

            word = expr.base
            assert isinstance(word, Symbol)

            return str(word)
        else:
            assert isinstance(expr, Symbol)
            return str(expr)

    def _get_symbol_for_block(self, block):
        if block == TuringPlanner.END_BLOCK:
            return Tape.BLANK

        word = self._get_word_for_block(block)

        return word[0]

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _get_blank_delta(self):
        return {T(tape): A(Tape.BLANK, move=Stay()) for tape in [0] + self.planner.tapes}

    def _get_new_state(self, description):
        return self.transition.get_new_state(self.prefix, description)

# --------------------------------------------------------------------
#
# turing machine language recognition
#
# --------------------------------------------------------------------


class LanguageMachine(MachineBuilder):
    def __init__(self, language):
        super().__init__(planner=AutomaticPlanner(language=LanguageFormula.normalize(language)))


class TuringParser:
    def __init__(self, language):
        self.language = language

        self.language_machines = list()

        if not isinstance(language, Language):
            raise RuntimeError("can't build machine from " + str(type(language)))

        if isinstance(language, LanguageFormula):
            planner = AutomaticPlanner(language=LanguageFormula.normalize(language))
            language_machine = MachineBuilder(planner=planner)

            self.language_machines.append(language_machine)
            self.turing = language_machine.turing

        elif isinstance(language, LanguageUnion):

            planners = []
            for ll in language.languages:
                planners.append(AutomaticPlanner(language=LanguageFormula.normalize(ll)))

            max_tapes = max([max(p.tapes) for p in planners])

            for planner in planners:
                for i in range(len(planner.tapes), max_tapes):
                    planner.tapes.append(i + 1)

            for planner in planners:
                language_machine = MachineBuilder(planner=planner)

                self.language_machines.append(language_machine)

            combined = self._join_machines()

            final_states = {trans.target for state in combined.transitions
                            for trans in combined.transitions[state] if trans.target not in combined.transitions}

            self.turing = TuringMachine(initial=self.language_machines[0].turing.initial,
                                        transition=combined, final=final_states)

    def info(self):
        for i, each in enumerate(self.language_machines):
            print("[*] language machine " + str(i))
            each.info()
            print()

        print("[*] turing machine")
        print(self.turing.transition)

        if self.turing.is_non_deterministic():
            non_deterministic_deltas = dict()
            print("[+] non deterministic :", self.turing.is_non_deterministic(deltas=non_deterministic_deltas))
            if len(non_deterministic_deltas):
                for state in non_deterministic_deltas:
                    for each in non_deterministic_deltas[state]:
                        print("[+] -", each)

        else:
            print("[*] the turing machine is deterministic :)")

    @staticmethod
    def _check_equal_prefix(deltas):
        prefix = [transition.source.prefix for m in deltas for s in deltas[m] for transition in deltas[m][s]]
        return len(prefix) and prefix.count(prefix[0]) == len(prefix)

    @staticmethod
    def _check_equal_states(deltas):
        number = [transition.source.number for m in deltas for s in deltas[m] for transition in deltas[m][s]]
        return len(number) and number.count(number[0]) == len(number)

    @staticmethod
    def _check_equal_transitions(deltas):
        transitions = {m: set(deltas[m][s]) for m in deltas for s in deltas[m]}
        return reduce(lambda s, t: transitions[s] == transitions[t], transitions)

    def _reprefix_transition(self, state, transition, prefix, first=None):
        if first is None:
            first = state

        if state in transition:
            for each in transition[state]:
                if each.target != first:
                    previous_state = copy.deepcopy(each.target)
                    each.target = State(prefix + str(each.target.number))

                    if each.source != first:
                        each.source = State(prefix + str(each.source.number))

                    self._reprefix_transition(previous_state, transition, prefix, first)

    def _join_machines(self):
        combined = TuringDelta(tapes=max(self.language_machines,
                                         key=lambda l: l.turing.transition.tapes).transition.tapes)

        for machine in self.language_machines:
            for i in range(machine.turing.transition.tapes, combined.tapes):
                machine.turing.transition.add_tape()

        deltas = [copy.deepcopy(machine.turing.transition) for machine in self.language_machines]

        for delta in deltas:
            for j in range(delta.tapes, combined.tapes):
                delta.add_tape()

        delta_stack = {i: delta for i, delta in enumerate(deltas)}

        for block in range(0, len(max(self.language_machines,
                                  key=lambda l: len(l.language.expression)).language.expression)):

            for i in list(delta_stack):
                if len(self.language_machines[i].language.expression) == block:
                    del delta_stack[i]

            if len(delta_stack) == 1:
                break

            for k in filter(lambda d: d in delta_stack, list(delta_stack)):
                expr = self.language_machines[k].language.expression[block]
                minimal = sorted(m[expr.exp] for m in self.language_machines[k].minimal)

                for m in filter(lambda d: d != k and d in delta_stack, list(delta_stack)):
                    other = self.language_machines[m].language.expression[block]

                    should_reprefix = False

                    if isinstance(other, Pow) and isinstance(expr, Pow):
                        other_minimal = sorted(m[expr.exp] for m in self.language_machines[m].minimal)

                        if len(other_minimal) != len(minimal) or \
                                any([o != e for o, e in zip(other_minimal, minimal)]) or other.base != expr.base:
                            should_reprefix = True

                    elif isinstance(other, Symbol) and isinstance(expr, Symbol):
                        if other != expr:
                            should_reprefix = True

                    else:
                        should_reprefix = True

                    if should_reprefix:
                        machine = self.language_machines[m]
                        current_prefix = machine.prefix

                        state = self._get_state_for_block(m, block - 1)

                        self._reprefix_transition(state, deltas[m].transitions, chr(ord(current_prefix) - m))

                        del delta_stack[m]

        for i in range(0, len(deltas)):
            for each in deltas[i].transitions:
                for transition in deltas[i].transitions[each]:
                    combined.merge_transition(transition)

        return combined

    def _get_state_for_block(self, m, block):
        if block < 0:
            return self.language_machines[m].initial_state

        else:
            return self.language_machines[m].state_from_block[block]
