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

    def __call__(self, *args, **kwargs):
        raise RuntimeError("__call__ not implemented")


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

