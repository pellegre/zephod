from pyauto.automata.turing import *
import itertools


class FunctionMachine:
    def __init__(self, expression, domain):
        self.expression = expression
        self.domain = domain

        self.symbol_tapes, self.tape_symbols, self.tape_counter = dict(), dict(), 1

        self.symbols = sorted(list(expression.free_symbols), key=lambda s: str(s))

        for symbol in self.symbols:
            if not any([symbol in c.free_symbols for c in self.domain]):
                raise RuntimeError("missing domain for symbol " + str(symbol))

        self.tape_counter = 0
        self.state_counter = 0

        self.initial = State("z0")
        self.result_tape = None

        self.turing_machine = self._get_initial_machine()
        self._copy_initial_input(self.symbols.copy())

        self._calculate_expression(self.expression)

    def info(self):
        print("[+] TURING function")
        print("[+] - input symbols")

        for tape in self.tape_symbols:
            print("  === symbol", self.tape_symbols[tape], "is at tape", C(tape))

    def get_unary_input(self, values):
        assert len(values) == len(self.symbols)

        unary_input = []
        for i, symbol in enumerate(self.symbols):
            value = values[symbol]

            if value == 0 and i == len(self.symbols) - 1:
                unary_input += ['0'] + ['0']
            else:
                unary_input += ['1' for _ in range(value)] + ['0']

        return ''.join(unary_input[:-1])

    def run_machine(self, values):
        unary_input = self.get_unary_input(values)
        assert self.result_tape

        buffer = self.turing_machine.debug(unary_input)

        result = max(0, self.expression.subs(values))
        calculated = buffer.tapes[self.result_tape].buffer.count('1')

        print("[+] - function result for", values, "=", result)
        print("[+] - machine calculated", calculated)

        accepted = buffer.state() in self.turing_machine.final and self.turing_machine.is_done(buffer)
        return result == calculated and accepted

    @staticmethod
    def _factorize_expression(expr):
        if isinstance(expr, Number):
            return int(expr), expr

        try:
            number = expr.args[[isinstance(a, Number) for a in expr.args].index(True)]
            sym = expr.args[[not isinstance(a, Number) for a in expr.args].index(True)]
        except ValueError:
            number = Number(1)
            sym = expr

        return number, sym

    def _replicate_tape_with_factor(self, transition, initial_state, fact, expr, tape, result_tape):
        state = self._get_new_state()

        result_expr = self.tape_symbols[result_tape]

        if fact < 0:
            print("------> initial fact < 0", fact, state)

            movement = A("1", new=Tape.BLANK, move=Left())
            movement_stay = A("1", move=Stay())
            movement_hit_zero = A("Z", move=Stay())

            # ---- get the first one

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, move=Left())
            delta[C(tape)] = A(Tape.BLANK, move=Left())

            transition.add(initial_state, state, delta)

        else:
            print("------> initial fact > 0", fact, state)

            movement = A(Tape.BLANK, new="1", move=Right())
            movement_stay = A(Tape.BLANK, move=Stay())
            movement_hit_zero = None

            delta = self._get_blank_delta()
            delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
            delta[C(tape)] = A(Tape.BLANK, move=Left())

            transition.add(initial_state, state, delta)

        if abs(fact) >= 1:
            for i in range(abs(fact)):
                if i % 2 == 0:
                    print("------> i % 2 == 0", state)
                    # ---- go to left
                    delta = self._get_blank_delta()
                    delta[C(result_tape)] = movement
                    delta[C(tape)] = A("1", move=Left())

                    transition.add(state, state, delta)

                    if movement_hit_zero and not self._is_expression_strictly_positive(result_expr):
                        delta[C(result_tape)] = movement_hit_zero
                        transition.add(state, state, delta)

                    # ---- hit X
                    delta = self._get_blank_delta()
                    delta[C(result_tape)] = movement_stay
                    delta[C(tape)] = A("X", move=Right())

                    next_state = self._get_new_state()
                    transition.add(state, next_state, delta)

                    if movement_hit_zero and not self._is_expression_strictly_positive(result_expr):
                        delta[C(result_tape)] = movement_hit_zero
                        transition.add(state, next_state, delta)

                    # swap states
                    state = next_state

                elif i % 2 == 1:
                    print("------> i % 2 == 1", state)
                    # ---- go to right
                    delta = self._get_blank_delta()
                    delta[C(result_tape)] = movement
                    delta[C(tape)] = A("1", move=Right())

                    transition.add(state, state, delta)

                    if movement_hit_zero and not self._is_expression_strictly_positive(result_expr):
                        delta[C(result_tape)] = movement_hit_zero
                        transition.add(state, state, delta)

                    # ---- hit blank
                    delta = self._get_blank_delta()
                    delta[C(result_tape)] = movement_stay

                    if i == abs(fact) - 1:
                        delta[C(tape)] = A(Tape.BLANK, move=Stay())
                    else:
                        delta[C(tape)] = A(Tape.BLANK, move=Left())

                    next_state = self._get_new_state()
                    transition.add(state, next_state, delta)

                    if movement_hit_zero and not self._is_expression_strictly_positive(result_expr):
                        delta[C(result_tape)] = movement_hit_zero
                        transition.add(state, next_state, delta)

                    # swap states
                    state = next_state

            if abs(fact) % 2 == 1:
                print("------> abs(fact) == 1", state)

                mid_state = self._get_new_state()

                if fact < 0:
                    print("------> fact < 0", state, fact)

                    # ---- hit X, and sync tapes

                    delta = self._get_blank_delta()
                    delta[C(result_tape)] = A("1", move=Right())
                    delta[C(tape)] = A("1", move=Stay())

                    transition.add(state, state, delta)

                    if movement_hit_zero and not self._is_expression_strictly_positive(result_expr):
                        delta[C(result_tape)] = A("Z", move=Right())
                        transition.add(state, state, delta)

                    # ---- hit X, and sync tapes

                    if not self._is_expression_strictly_positive(expr):
                        delta = self._get_blank_delta()
                        delta[C(result_tape)] = A("1", move=Right())
                        delta[C(tape)] = A(Tape.BLANK, move=Stay())

                        transition.add(state, mid_state, delta)

                delta = self._get_blank_delta()
                delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
                delta[C(tape)] = A("1", move=Right())

                transition.add(state, mid_state, delta)

                # -------- move back to the end

                delta = self._get_blank_delta()
                delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
                delta[C(tape)] = A("1", move=Right())

                transition.add(mid_state, mid_state, delta)

                # -------- find the blank
                final_state = self._get_new_state()

                delta = self._get_blank_delta()
                delta[C(result_tape)] = A(Tape.BLANK, move=Stay())
                delta[C(tape)] = A(Tape.BLANK, move=Stay())

                transition.add(mid_state, final_state, delta)

            else:
                if fact < 0:
                    print("------> fact < 0", state, fact)

                    # ---- hit X, and sync tapes

                    delta = self._get_blank_delta()
                    delta[C(result_tape)] = A("1", move=Right())
                    delta[C(tape)] = A(Tape.BLANK, move=Stay())

                    transition.add(state, state, delta)

                final_state = state

            return final_state

    def _calculate_expression(self, expression):
        args = sorted(expression.args, key=lambda w: int(self._factorize_expression(w)[0]), reverse=True)

        # get result tape
        transition = self.turing_machine.transition
        result_tape = self._add_to_tape(expression)
        transition.add_tape()

        self.result_tape = C(result_tape)

        # prepare first step (mark result with Z)
        initial = self.turing_machine.final.pop()
        state = self._get_new_state()

        delta = self._get_blank_delta()
        delta[C(result_tape)] = A(Tape.BLANK, new="Z", move=Right())

        transition.add(initial, state, delta)

        for each in args:
            if not isinstance(each, Number):
                # -------- get factor and expression
                fact, expr = self._factorize_expression(each)

                # -------- prepare the sum
                if expr not in self.symbol_tapes:
                    raise RuntimeError("expression " + str(expr) + " is not in any tape")

                tape = self._get_tape_for_symbol(expr)

                print("-----> state before", expr, state)
                # -------- add (or subtract) ones
                state = self._replicate_tape_with_factor(transition, state, fact, expr, tape, result_tape)
                print("-----> state after", expr, state)

        final_state = self._get_new_state()

        # ---- final machine
        delta = self._get_blank_delta()
        transition.add(state, final_state, delta)

        # ---- final machine

        self.turing_machine = TuringMachine(initial=self.turing_machine.initial, transition=transition,
                                            final={final_state})

    def _add_to_tape(self, expression):
        self.tape_counter += 1

        if expression not in self.symbol_tapes:
            self.symbol_tapes[expression] = [self.tape_counter]
        else:
            self.symbol_tapes[expression].append(self.tape_counter)

        self.tape_symbols[self.tape_counter] = expression

        return self.tape_counter

    def _get_new_state(self):
        self.state_counter += 1

        return State(self.initial.prefix + str(self.state_counter))

    def _get_blank_delta(self):
        return {C(tape): A(Tape.BLANK, move=Stay()) for tape in range(0, self.tape_counter + 1)}

    def _get_tape_for_symbol(self, symbol):
        return self.symbol_tapes[symbol][0]

    def _get_initial_machine(self):
        transition = TuringDelta()

        input_symbols = self.symbols.copy()

        symbol = input_symbols[0]
        tape = self._add_to_tape(symbol)
        state = self._get_new_state()

        # ---- X mark

        delta = self._get_blank_delta()

        delta[C(0)] = A("1", move=Stay())
        delta[C(tape)] = A(Tape.BLANK, new="X", move=Right())

        transition.add(self.initial, state, delta)

        if not self._is_expression_strictly_positive(symbol):
            delta = self._get_blank_delta()

            delta[C(0)] = A("0", move=Stay())
            delta[C(tape)] = A(Tape.BLANK, new="X", move=Right())

            transition.add(self.initial, state, delta)

        return TuringMachine(initial=self.initial, transition=transition, final={state})

    def _copy_initial_input(self, input_symbols: list):
        transition = self.turing_machine.transition

        symbol = input_symbols.pop(0)
        tape = self._get_tape_for_symbol(symbol)

        assert len(self.turing_machine.final) == 1
        state = self.turing_machine.final.pop()

        # ---- copy input

        delta = self._get_blank_delta()
        delta[C(0)] = A("1", move=Right())
        delta[C(tape)] = A(Tape.BLANK, new="1", move=Right())

        transition.add(state, state, delta)

        if len(input_symbols):
            # ---- detect next input

            next_tape = self._add_to_tape(input_symbols[0])
            transition.add_tape()

            final_state = self._get_new_state()

            delta = self._get_blank_delta()
            delta[C(0)] = A("0", move=Right())
            delta[C(next_tape)] = A(Tape.BLANK, new="X", move=Right())

            transition.add(state, final_state, delta)

            # ---- create machine

            self.turing_machine = TuringMachine(initial=self.turing_machine.initial, transition=transition,
                                                final={final_state})

            self._copy_initial_input(input_symbols)

        else:
            final_state = self._get_new_state()

            # ---- final machine
            delta = self._get_blank_delta()
            transition.add(state, final_state, delta)

            if not self._is_expression_strictly_positive(symbol):
                delta = self._get_blank_delta()

                delta[C(0)] = A("0", move=Right())
                delta[C(tape)] = A(Tape.BLANK, move=Stay())

                transition.add(state, final_state, delta)

            # ---- final machine

            self.turing_machine = TuringMachine(initial=self.turing_machine.initial, transition=transition,
                                                final={final_state})

    def _is_expression_strictly_positive(self, expr):
        def get_total_sum(v):
            return reduce(lambda ss, rr: ss + rr, v.values())

        length = 10

        space = [list(range(0, length)) for _ in self.symbols]
        cross = itertools.product(*space)

        values = None
        for each in cross:
            current = {s: v for s, v in zip(self.symbols, each)}
            if not values or get_total_sum(current) < get_total_sum(values):
                if all([c.subs(current) for c in self.domain]):
                    values = current

        return expr.subs(values) > 0
