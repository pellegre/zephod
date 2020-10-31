from pyauto.language import *


# --------------------------------------------------------------------
#
# machine plan declaration
#
# --------------------------------------------------------------------


class MachinePlan:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__str__()


class BlockPlan(MachinePlan):
    def __init__(self, block):
        self.block = block

    def __str__(self):
        return "block " + str(self.block)

    def __repr__(self):
        return self.__str__()


class ParseLoneSymbol(BlockPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "ParseLoneSymbol on " + super().__str__()


class ParseAccumulate(BlockPlan):
    def __init__(self, tape, **kwargs):
        super().__init__(**kwargs)
        self.tape = tape

    def __str__(self):
        return "ParseAccumulate on " + super().__str__() + " in tape " + str(self.tape)


class ParseEqual(BlockPlan):
    def __init__(self, tape, **kwargs):
        super().__init__(**kwargs)
        self.tape = tape

    def __str__(self):
        return "ParseEqual on " + super().__str__() + " in tape " + str(self.tape)


class OperationPlan(MachinePlan):
    def __init__(self, source_tape, target_tape):
        self.source_tape, self.target_tape = source_tape, target_tape

    def __str__(self):
        return "tape " + str(self.source_tape) + " and " + str(self.target_tape)

    def __repr__(self):
        return self.__str__()


class RewindAll(MachinePlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Rewind all tapes"


class Accumulate(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "Accumulate on " + super().__str__()


class CompareGreater(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareGreater on " + super().__str__()


class CompareStrictGreater(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareStrictGreater on " + super().__str__()


class CompareUnequal(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareUnequal on " + super().__str__()


class CompareEqual(OperationPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "CompareEqual on " + super().__str__()


class TuringPlanner:
    END_BLOCK = -1

    def __init__(self, language: LanguageFormula):
        self.language = language

        self.symbol_tape = dict()
        self.tapes = list()

        self.machine_plan = list()
        self.exit_blocks = dict()

        for i in range(0, len(self.language.expression)):
            self.exit_blocks[i] = self._get_exit_blocks(i)

    def info(self):
        print("\n[+] language")
        self.language.info()

        print("\n[+] turing planner")
        print("[+] exit blocks", self.exit_blocks)
        print("[+] symbol tapes", self.symbol_tape)

        for i, each in enumerate(self.machine_plan):
            print(i, each)

    def explore_exit_blocks(self, block):
        stack, exit_blocks = set(), list()

        for b in range(block, len(self.language.expression) + 1):
            if b < len(self.language.expression):
                expr = self.language.expression[b]

                if isinstance(expr, Symbol):
                    exit_blocks.append(b)
                    break

                if expr.exp not in stack:
                    exit_blocks.append(b)

                minimal = self._get_minimum_indices()

                if all([m[expr.exp] for m in minimal]) > 0:
                    break

                stack.add(expr.exp)

            else:
                exit_blocks.append(AutomaticPlanner.END_BLOCK)

        return exit_blocks

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _get_exit_blocks(self, block):
        if block < len(self.language.expression) - 1:
            expr = self.language.expression[block]

            if isinstance(expr, Symbol):
                next_expr = self.language.expression[block + 1]

                if isinstance(next_expr, Symbol):
                    return [block + 1]

                else:
                    minimal = self._get_minimum_indices()

                    if all([m[next_expr.exp] for m in minimal]) > 0:
                        return [block + 1]

                    else:
                        return self.explore_exit_blocks(block + 1)

            elif isinstance(expr, Pow):
                next_expr = self.language.expression[block + 1]

                if isinstance(next_expr, Symbol) or expr.exp == next_expr.exp:
                    return [block + 1]

                else:
                    minimal = self._get_minimum_indices({expr.exp: 1})

                    if all([m[next_expr.exp] for m in minimal]) > 0:
                        return [block + 1]

                    else:
                        return self.explore_exit_blocks(block + 1)

            else:
                raise RuntimeError("unrecognized expression " + str(expr))

        else:
            return [TuringPlanner.END_BLOCK]

# --------------------------------------------------------------------
#
# automatic turing machine planner
#
# --------------------------------------------------------------------


class AutomaticPlanner(TuringPlanner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        expression_symbols = list()
        for each in self.language.expression:
            if isinstance(each, Pow):
                if each.exp not in expression_symbols:
                    expression_symbols.append(each.exp)

        for i, each in enumerate(expression_symbols):
            self.tapes.append(i + 1)
            self.symbol_tape[each] = i + 1

        self._process_blocks(current=0)

        self._verify_conditions()

    def _process_blocks(self, current=0, symbol_stack=None):
        if symbol_stack is None:
            symbol_stack = set()

        if current < len(self.language.expression):
            expr = self.language.expression[current]

            if isinstance(expr, Symbol):
                self.machine_plan.append(ParseLoneSymbol(block=current))

            elif isinstance(expr, Pow):
                if expr.exp not in symbol_stack:
                    symbol_stack.add(expr.exp)

                    self.machine_plan.append(ParseAccumulate(tape=self.symbol_tape[expr.exp],
                                                             block=current))

                else:
                    self.machine_plan.append(ParseEqual(tape=self.symbol_tape[expr.exp],
                                                        block=current))

            self._process_blocks(current + 1, symbol_stack)

    def _verify_conditions(self):
        if not all([len(c.free_symbols) == 1 for c in self.language.conditions]):

            for c in filter(lambda c: len(c.free_symbols) > 1, self.language.conditions):
                if not isinstance(c.rhs, Symbol):
                    symbol, rest = c.rhs.args[0], c.rhs.args[1:]
                    target_tape = self.symbol_tape[symbol]

                    for each in rest:
                        source_tape = self.symbol_tape[each]
                        self.machine_plan.append(Accumulate(target_tape=target_tape, source_tape=source_tape))

                    self.symbol_tape[c.rhs] = target_tape

                if not isinstance(c.lhs, Symbol):
                    symbol, rest = c.lhs.args[0], c.lhs.args[1:]
                    target_tape = self.symbol_tape[symbol]

                    for each in rest:
                        source_tape = self.symbol_tape[each]
                        self.machine_plan.append(Accumulate(target_tape=target_tape, source_tape=source_tape))

                    self.symbol_tape[c.lhs] = target_tape

                if isinstance(c, Eq):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareEqual(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, StrictGreaterThan):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareStrictGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, GreaterThan):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, StrictLessThan):
                    source_tape = self.symbol_tape[c.rhs]
                    target_tape = self.symbol_tape[c.lhs]

                    self.machine_plan.append(CompareStrictGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, LessThan):
                    source_tape = self.symbol_tape[c.rhs]
                    target_tape = self.symbol_tape[c.lhs]

                    self.machine_plan.append(CompareGreater(target_tape=target_tape, source_tape=source_tape))

                elif isinstance(c, Unequality):
                    source_tape = self.symbol_tape[c.lhs]
                    target_tape = self.symbol_tape[c.rhs]

                    self.machine_plan.append(CompareUnequal(target_tape=target_tape, source_tape=source_tape))


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

            delta[C(0)] = A(next_symbol, move=Stay())
            for tape in self.planner.tapes:
                delta[C(tape)] = A(Tape.BLANK, new="X", move=Right())

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

                final_states = [self.state_from_block[block] for block in self.planner.exit_blocks[current_block]]

                if isinstance(plan, ParseLoneSymbol):
                    self._parse_lone_symbol(state, final_states)

                elif isinstance(plan, ParseAccumulate):
                    self._parse_block_and_accumulate(plan.tape, state, final_states)

                elif isinstance(plan, ParseEqual):
                    self._parse_block_exact_equal(plan.tape, state, final_states)

                else:
                    raise RuntimeError("unhandled plan " + str(plan))

                next_state = self.parsed_state

            elif isinstance(plan, OperationPlan):
                source_tape = plan.source_tape
                target_tape = plan.target_tape

                if isinstance(plan, Accumulate):
                    next_state = self._accumulate_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareGreater):
                    next_state = self._verify_greater_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareStrictGreater):
                    next_state = self._verify_strict_greater_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareUnequal):
                    next_state = self._verify_unequal_counters(next_state, source_tape, target_tape)

                elif isinstance(plan, CompareEqual):
                    next_state = self._verify_equal_counters(next_state, source_tape, target_tape)

                else:
                    raise RuntimeError("unhandled plan " + str(plan))

            else:
                raise RuntimeError("unhandled plan " + str(plan))

        return next_state

    def _accumulate_counters(self, initial_state, tape, result_tape):
        next_state = initial_state

        left_state = self._get_new_state(description="moving left while accumulating C" +
                                                     str(tape) + " on " + C(result_tape))

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A(Tape.BLANK, move=Left())
        self.transition.add(next_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, new="Z", move=Right())
        delta[C(tape)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state(description="hit X while accumulating C" +
                                                      str(tape) + " on " + C(result_tape))

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        next_state = self._get_new_state(description="rewind " + C(tape) + " after accumulating")

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
        delta[C(tape)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, next_state, delta)

        return next_state

    def _verify_strict_greater_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (greater) " +
                                                     C(tape_a) + " > " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (greater) " + C(tape_a) + " > " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (greater) " +
                                          C(tape_a) + " > " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _verify_greater_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (greater / equal) " +
                                                     C(tape_a) + " >= " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (greater / equal) " + C(tape_a) + " >= " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("X", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (greater / equal) " +
                                          C(tape_a) + " >= " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _verify_equal_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (equality) " +
                                                     C(tape_a) + " == " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (equality) " + C(tape_a) + " == " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("X", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (equality) " +
                                          C(tape_a) + " == " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _verify_unequal_counters(self, initial_state, tape_a, tape_b):
        left_state = self._get_new_state(description="moving left while comparing (unequality) " +
                                                     C(tape_a) + " != " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Left())
        delta[C(tape_b)] = A(Tape.BLANK, move=Left())
        self.transition.add(initial_state, left_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Left())
        delta[C(tape_b)] = A("Z", move=Left())
        self.transition.add(left_state, left_state, delta)

        right_state = self._get_new_state("hit X after comparing (unequality) " + C(tape_a) + " != " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("X", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("X", move=Right())
        self.transition.add(left_state, right_state, delta)

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A("Z", move=Right())
        delta[C(tape_b)] = A("Z", move=Right())
        self.transition.add(right_state, right_state, delta)

        final_state = self._get_new_state("moving right (rewind) after comparing (unequality) " +
                                          C(tape_a) + " != " + C(tape_b))

        delta = self._get_blank_delta()
        delta[C(tape_a)] = A(Tape.BLANK, move=Stay())
        delta[C(tape_b)] = A(Tape.BLANK, move=Stay())
        self.transition.add(right_state, final_state, delta)

        return final_state

    def _parse_block_exact_equal(self, counter_tape, initial_state, final_states):
        block = self.block_from_state[initial_state]
        word = self._get_word_for_block(block)

        rewind_state = self._get_new_state(description="rewind " + C(counter_tape) + " after parsing block " +
                                                       str(block) + " (" + str(self.language.expression[block]) + ")")

        for each in final_states:
            next_symbol = self._get_symbol_for_state(each)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())
            delta[C(counter_tape)] = A("X", move=Right())
            self.transition.add(initial_state, rewind_state, delta)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())
            delta[C(counter_tape)] = A("Z", move=Right())
            self.transition.add(rewind_state, rewind_state, delta)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())
            delta[C(counter_tape)] = A(Tape.BLANK, move=Stay())
            self.transition.add(rewind_state, each, delta)

        current_state = initial_state

        delta = self._get_blank_delta()
        delta[C(0)] = A(word[0], move=Stay())
        delta[C(counter_tape)] = A(Tape.BLANK, move=Left())
        self.transition.add(current_state, current_state, delta)

        for i in range(0, len(word)):
            delta = self._get_blank_delta()

            if i == len(word) - 1:
                next_state = initial_state
                delta[C(counter_tape)] = A("Z", move=Left())

            else:
                delta[C(counter_tape)] = A("Z", move=Stay())
                next_state = self._get_new_state(description="parsed letter " + word[i] +
                                                             " of word " + word + " in block " + str(block) +
                                                             " while verifying counter in C" + str(counter_tape))

            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

    def _parse_block_and_accumulate(self, counter_tape, initial_state, final_states):
        block = self.block_from_state[initial_state]
        word = self._get_word_for_block(block)

        expr = self.language.expression[block]

        assert isinstance(expr, Pow)
        assert len(expr.exp.free_symbols) == 1

        current_state = initial_state

        for i in range(0, len(word)):
            delta = self._get_blank_delta()

            if i == len(word) - 1:
                next_state = initial_state
                delta[C(counter_tape)] = A(Tape.BLANK, new="Z", move=Right())

            else:
                next_state = self._get_new_state(description="parsing letter " + word[i] +
                                                             " of word " + word + " in block " + str(block) +
                                                             " while counting " + str(counter_tape) + " first time")

            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

        for each in final_states:
            next_symbol = self._get_symbol_for_state(each)

            delta = self._get_blank_delta()
            delta[C(0)] = A(next_symbol, move=Stay())

            if any([m[expr.exp] == 0 for m in self.minimal]):
                self.transition.add(initial_state, each, delta)
            else:
                self.transition.add(current_state, each, delta)

    def _parse_lone_symbol(self, initial_state, final_states):
        current_state = initial_state

        block = self.block_from_state[initial_state]
        word = self._get_word_for_block(block)

        for i in range(0, len(word)):
            next_state = self._get_new_state(description="parsing letter " + word[i] +
                                                         " of lone word " + word)

            delta = self._get_blank_delta()
            delta[C(0)] = A(word[i], move=Right())
            self.transition.add(current_state, next_state, delta)

            current_state = next_state

        for each in final_states:
            delta = self._get_blank_delta()
            delta[C(0)] = A(self._get_symbol_for_state(each), move=Stay())
            self.transition.add(current_state, each, delta)

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
        if block == len(self.language.expression):
            return Tape.BLANK

        word = self._get_word_for_block(block)

        return word[0]

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _get_blank_delta(self):
        return {C(tape): A(Tape.BLANK, move=Stay()) for tape in [0] + self.planner.tapes}

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
