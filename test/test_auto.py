from pyauto.language import *
from pyauto.automata.finite import *
from pyauto.automata.pushdown import *

from utils.automaton.builder import *
from utils.automaton.csg import *
from utils.function import *


def test_case_1(inp, plotter=False, run_grammar=False):
    transition = FADelta()
    transition.add("e0", "e1", {"0"})
    transition.add("e0", "e2", {"1"})
    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e1", {"0"})
    transition.add("e1", "e3", {"0"})
    transition.add("e2", "e3", {"1"})
    transition.add("e3", "e3", {"0", "1"})

    dfsm = FiniteAutomata(transition, "e0", {"e3"})
    assert not dfsm.has_null_transitions()

    dfsm_value = dfsm.read(inp)

    transition = FADelta()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    nfsm = FiniteAutomata(transition, "e0", {"e2", "e4"})
    assert not nfsm.has_null_transitions()

    nfsm_value = nfsm.read(inp)
    assert dfsm_value == nfsm_value

    rebased_nfsm = nfsm.rebase(17)
    rebased_nfsm_value = rebased_nfsm.read(inp)
    assert rebased_nfsm_value == nfsm_value

    minimized = nfsm.get_deterministic_automata().minimize_automata()
    assert minimized.read(inp) == nfsm_value

    pda = dfsm.get_pushdown_automata()
    pda_value = pda.read(inp)
    assert pda_value == nfsm_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(nfsm)

        for each in grammar_from_fda.enumerate(length=10):
            assert minimized.read(each)

    if plotter:
        AutomataPlotter.plot(nfsm)
        AutomataPlotter.plot(minimized)

    return rebased_nfsm_value


def test_case_2(inp, plotter=False, run_grammar=False):
    transition = FADelta()
    transition.add("s0", "s0", {"a"})
    transition.add("s0", "s2", {Transition.NULL})
    transition.add("s1", "s2", {"a", "b"})
    transition.add("s2", "s0", {"a"})
    transition.add("s2", "s1", {Transition.NULL})

    nfsm = FiniteAutomata(transition, "s0", {"s1"})
    assert nfsm.has_null_transitions()

    nfsm_value = nfsm.read(inp)
    stripped = nfsm.remove_null_transitions()
    dfsm = stripped.get_deterministic_automata()

    assert not dfsm.is_non_deterministic()
    assert dfsm.read(inp) == nfsm_value

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == nfsm_value

    pda = dfsm.get_pushdown_automata()
    pda_value = pda.read(inp)
    assert pda_value == nfsm_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(stripped)

        for each in grammar_from_fda.enumerate(length=10):
            assert minimized.read(each)

    if plotter:
        AutomataPlotter.plot(nfsm)
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(stripped)
        AutomataPlotter.plot(minimized)

    return nfsm_value


def test_case_3(inp, plotter=False, run_grammar=False):
    expr = (Z("1") + Z("10")) | ~Z("01")
    assert expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plotter:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_4(inp, plotter=False, run_grammar=False):
    expr = Z("00") + (~Z("0") | Z("1"))
    assert expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plotter:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_5(inp, plotter=False, run_grammar=False):
    expr = FiniteAutomataBuilder.get_finite_automata_from_csv("./csv/zuto.csv")
    assert not expr.has_null_transitions()

    expr_value = expr.read(inp)

    dfsm = expr.get_deterministic_automata()
    assert not dfsm.is_non_deterministic()

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == expr_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(minimized)

        for each in grammar_from_fda.enumerate(length=8):
            assert minimized.read(each)

    if plotter:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_9(inp, plotter=False, run_grammar=False):
    expr = Z("00") + (~Z("0") | Z("1"))
    assert expr.has_null_transitions()

    stripped = expr.remove_null_transitions()
    expr_value = stripped.read(inp)

    dfsm = stripped.get_deterministic_automata()
    assert not dfsm.is_non_deterministic()

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == expr_value

    pda = dfsm.get_pushdown_automata()
    pda_value = pda.read(inp)
    assert pda_value == expr_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(minimized)

        for each in grammar_from_fda.enumerate(length=10):
            assert minimized.read(each)

    if plotter:
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_11(inp, plotter=False, run_grammar=False):
    transition = FADelta()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    nfsm = FiniteAutomata(transition, "e0", {"e2", "e4"})

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(nfsm)

        for each in grammar_from_fda.enumerate(length=10):
            assert nfsm.read(each)

    assert not nfsm.has_null_transitions()
    dfsm = nfsm.get_deterministic_automata()

    expr_value = dfsm.read(inp)

    assert not dfsm.is_non_deterministic()

    pda = dfsm.get_pushdown_automata()
    pda_value = pda.read(inp)
    assert pda_value == expr_value

    if plotter:
        AutomataPlotter.plot(dfsm)

    return expr_value


def test_case_12(inp, plotter=False, run_grammar=False):
    transition = FADelta()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    nfsm = FiniteAutomata(transition, "e0", {"e2", "e4"})
    assert not nfsm.has_null_transitions()
    dfsm = nfsm.get_deterministic_automata()

    expr_value = dfsm.read(inp)

    assert not dfsm.is_non_deterministic()

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == expr_value

    pda = minimized.get_pushdown_automata()
    pda_value = pda.read(inp)
    assert pda_value == expr_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(minimized)

        for each in grammar_from_fda.enumerate(length=10):
            assert minimized.read(each)
            assert pda.read(each)

    if plotter:
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_13(inp, plotter=False, run_grammar=False):
    expr = (~(Z("aa") + Z("b")) | (Z("c") + Z("d")) | ~Z("cd"))
    assert expr.has_null_transitions()

    stripped = expr.remove_null_transitions()
    expr_value = stripped.read(inp)

    dfsm = stripped.get_deterministic_automata()
    assert not dfsm.is_non_deterministic()

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == expr_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(minimized)

        for each in grammar_from_fda.enumerate(length=10):
            assert minimized.read(each)

    if plotter:
        AutomataPlotter.plot(expr)
        AutomataPlotter.plot(stripped)
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_14(plotter=False, run_grammar=False):
    g = Grammar(terminal={"0", "1"}, non_terminal={"A", "B", "C"})

    g.add("S", "0A")
    g.add("S", "1B")
    g.add("A", "0C")
    g.add("A", "0")
    g.add("A", "1B")
    g.add("B", "0A")
    g.add("B", "1C")
    g.add("B", "1")
    g.add("C", "0C")
    g.add("C", "0")
    g.add("C", "1C")
    g.add("C", "0")

    fda = g.get_finite_automata()

    for each in g.enumerate(length=10):
        assert fda.read(each)

    grammar_from_fda = Grammar.build_from_finite_automata(fda)

    for each in grammar_from_fda.enumerate(length=10):
        assert fda.read(each)

    if plotter:
        AutomataPlotter.plot(fda)


def test_case_15(plotter=False):
    expr = (~(Z("aa") + Z("b")) | ~Z("cd"))
    assert expr.has_null_transitions()

    stripped = expr.remove_null_transitions()
    expr_value = stripped.read("")

    dfsm = stripped.get_deterministic_automata()
    assert not dfsm.is_non_deterministic()

    minimized = dfsm.minimize_automata()
    assert minimized.read("") == expr_value

    grammar_from_fda = Grammar.build_from_finite_automata(minimized)

    got_null_string = False

    for w in grammar_from_fda.enumerate(length=10):
        assert minimized.read(w)
        if not len(w):
            got_null_string = True

    assert got_null_string

    if plotter:
        AutomataPlotter.plot(minimized)

    return expr_value


def test_buffer():
    buffer = Buffer(data="test_string", initial="z0")
    assert buffer.pointer() == 0
    assert len(buffer.head()) == len(buffer.data())
    buffer.read("z1", 1)

    for i in range(len(buffer.head())):
        buffer.read("z1", 1)

    assert not len(buffer.head())
    assert buffer.pointer() == len(buffer.data())

    buffer = Buffer(data="test_string_", initial="z0")
    assert buffer.pointer() == 0
    assert len(buffer.head()) == len(buffer.data())

    for i in range(len(buffer.head()) // 2):
        buffer.read("z1", 2)

    assert not len(buffer.head())
    assert buffer.pointer() == len(buffer.data())

    assert len(buffer.states) == len(buffer.data()) // 2 + 1
    assert len(buffer.pointers) == len(buffer.data()) // 2 + 1


def test_case_16():
    expr = (~Z("aaa") | ~Z("bb") | ~Z("cd")).minimal()

    transition = FADelta()
    transition.add("z0", "z1", {~Z("aaa")})
    transition.add("z1", "z2", {~Z("bb")})
    transition.add("z2", "z3", {~Z("cd")})

    fda = FiniteAutomata(transition, initial="z0", final={"z0", "z1", "z2", "z3"})

    grammar_from_fda = Grammar.build_from_finite_automata(expr)

    for each in grammar_from_fda.enumerate(length=10):
        assert fda.read(each)


def test_constraints():
    contained = ContainedRule(pattern="acb", closure={"a", "b", "c", "d"}).compile()

    assert contained("read_a_1", "c") == State("read_c_2")
    assert contained("read_c_2", "b") == State("read_acb_3")
    assert contained("read_acb_3", "d") == State("read_acb_3")

    assert contained("read_c_2", "d") == State("read_none_0")
    assert contained("read_c_2", "a") == State("read_none_0")

    not_contained = NotContainedRule(pattern="acb", closure={"a", "b", "c", "d"}).compile()

    assert not_contained("read_none_0", "a") == State("read_a_1")
    assert not_contained("read_a_1", "c") == State("read_c_2")
    assert not_contained("read_c_2", "b") == ErrorState()

    assert not_contained("read_c_2", "a") == State("read_none_0")
    assert not_contained("read_c_2", "c") == State("read_none_0")
    assert not_contained("read_c_2", "d") == State("read_none_0")

    even = EvenRule(pattern="ab", closure={"a", "b", "c", "d"}).compile()

    assert even("read_even_ab_0", "a") == State("read_even_a_1")
    assert even("read_even_a_1", "b") == State("read_odd_ab_0")
    assert even("read_odd_ab_0", "a") == State("read_odd_a_1")
    assert even("read_odd_a_1", "b") == State("read_even_ab_0")

    assert even("read_even_a_1", "a") == State("read_even_ab_0")
    assert even("read_odd_a_1", "a") == State("read_odd_ab_0")

    assert even("read_even_ab_0", "b") == State("read_even_ab_0")
    assert even("read_odd_ab_0", "b") == State("read_odd_ab_0")

    odd = OddRule(pattern="ab", closure={"a", "b", "c", "d"}).compile()

    assert odd("read_even_ab_0", "a") == State("read_even_a_1")
    assert odd("read_even_a_1", "b") == State("read_odd_ab_0")
    assert odd("read_odd_ab_0", "a") == State("read_odd_a_1")
    assert odd("read_odd_a_1", "b") == State("read_even_ab_0")

    assert odd("read_even_a_1", "a") == State("read_even_ab_0")
    assert odd("read_odd_a_1", "a") == State("read_odd_ab_0")

    assert odd("read_even_ab_0", "b") == State("read_even_ab_0")
    assert odd("read_odd_ab_0", "b") == State("read_odd_ab_0")

    lang = RegularLanguage(alphabet={"a", "b", "c", "d"},
                           definition={
                               "rules": [
                                   Even(pattern="d"), Odd(pattern="b"), NotContained(pattern="addc")
                               ],
                               "closure": {"a", "b", "c", "d"}
                           })

    assert lang.fda.read("b")
    assert lang.fda.read("ddb")
    assert lang.fda.read("ddbbb")
    assert not lang.fda.read("")
    assert not lang.fda.read("ddbaddc")
    assert lang.fda.read("ddbadd")
    assert lang.fda.read("dadabddbab")
    assert not lang.fda.read("addc")

    lang = RegularLanguage(alphabet={"a", "b", "c"},
                           definition={
                               "rules": [
                                   Even(pattern="b"), Odd(pattern="c"),
                                   NotContained(pattern="aac"), Order(before={"a", "b"}, after={"a", "c"})
                               ],
                               "closure": {"a", "b", "c"}
                           })

    assert lang.fda.read("bbc")
    assert lang.fda.read("c")
    assert not lang.fda.read("cc")
    assert lang.fda.read("babaccc")
    assert not lang.fda.read("babacccbb")
    assert not lang.fda.read("babaacccbb")
    assert not lang.fda.read("bb")
    assert not lang.fda.read("")
    assert lang.fda.read("bababaaaabbaaaabcacac")
    assert not lang.fda.read("bababaaaabbaaaabcaacaac")

    lang = RegularLanguage(alphabet={"a", "b", "c"},
                           definition={
                               "rules": [
                                   Even(pattern="baa"), Odd(pattern="c"),
                                   NotContained(pattern="aac"), Order(before={"a", "b"}, after={"a", "c"})
                               ],
                               "closure": {"a", "b", "c"}
                           })

    assert lang.fda.read("c")
    assert lang.fda.read("baabaabc")
    assert not lang.fda.read("baabaac")
    assert lang.fda.read("baabaabccc")
    assert lang.fda.read("baaabaaabaabaaabcacac")
    assert not lang.fda.read("baaabaaabaabaaabcacacb")

    lang = RegularLanguage(alphabet={"a", "b", "c", "d"},
                           definition={
                               "rules": [
                                   Even(pattern="baa"),
                                   Order(before={"a", "b"}, after={"a", "c"}),
                                   Odd(pattern="c"), NotContained(pattern="aac"),
                                   Order(before={"a", "b", "c"}, after={"a", "d"}),
                                   Odd(pattern="dd"), NotContained(pattern="ad")
                               ],
                               "closure": {"a", "b", "c", "d"}
                           })

    assert lang.fda.read("cdd")
    assert lang.fda.read("baabaabbbbbcdd")
    assert lang.fda.read("cccdddddd")
    assert lang.fda.read("cacacdddddd")
    assert not lang.fda.read("cacacadddddd")
    assert not lang.fda.read("cacacaddddddb")
    assert not lang.fda.read("cacacaddddcdd")
    assert not lang.fda.read("bacabaabbbbbcdd")
    assert not lang.fda.read("badabaabbbbbcdd")

    # TODO : fix this case
    # lang = RegularLanguage(alphabet={"a", "b", "c"},
    #                        definition={
    #                                "rules": [
    #                                    Even(pattern="b"),
    #                                    Odd(pattern="c"), Odd(pattern="a"),
    #                                    Order(before={"a", "b"}, after={"a", "c"})
    #                                ],
    #                                "closure": {"a", "b", "c"}
    #                            })
    #
    # AutomataPlotter.plot(lang.fda)
    # assert lang.fda.read("bbca")


def test_pda_case_1():
    grammar = Grammar(non_terminal={"A"}, terminal={"a", "b", "c"})
    grammar.add("S", "A")
    grammar.add("A", "aAa")
    grammar.add("A", "bAb")
    grammar.add("A", "c")

    transition = PDADelta()

    transition.add("z0", "z0",
                   {
                        ("a", Stack.EMPTY): Push(obj="X"),
                        ("a", "X"): Push(obj="X"),
                        ("a", "Y"): Push(obj="X"),

                        ("b", Stack.EMPTY): Push(obj="Y"),
                        ("b", "Y"): Push(obj="Y"),
                        ("b", "X"): Push(obj="Y")
                    })

    transition.add("z0", "z1",
                   {
                        ("c", Stack.EMPTY): Null(),
                        ("c", "X"): Null(),
                        ("c", "Y"): Null()
                    })

    transition.add("z1", "z1",
                   {
                        ("a", "X"): Pop(),
                        ("b", "Y"): Pop(),
                    })

    transition.add("z1", "z2",
                   {
                        ("$", Stack.EMPTY): Null()
                    })

    pda = PushdownAutomata(transition, initial="z0", final={"z2"})

    for each in grammar.enumerate(length=10):
        assert pda.read(each)


def test_transitions():
    stack = Stack(initial="s0", data="AAABBBBBCCCCC")

    pda_read = PDAReadTransition(source="s0", target="s1",
                                 character="A", stack_action=PushStackAction(on_symbol=Stack.EMPTY, new_symbol="Z"))
    stack = pda_read(stack)

    print(stack)


def test_input_actions():
    inp = Input(initial="s0", data="AAABBBBBCCCCC", tapes=1)

    action = InputAction(actions={
        Tape.N(0): [WriteAction(on_symbol="A", new_symbol="D"), MoveRightAction(on_symbol="D")],
        Tape.N(1): [WriteAction(on_symbol=Tape.BLANK, new_symbol="Z"), MoveRightAction(on_symbol="Z")]
    })

    print(inp)
    inp = action(source="s0", target="s1", data=inp)
    assert not inp.error
    assert inp.tapes[Tape.N(0)].pointer() == 1
    assert inp.tapes[Tape.N(1)].pointer() == 1

    assert inp.tapes[Tape.N(0)].read() == 'A'
    assert inp.tapes[Tape.N(1)].read() == Tape.BLANK

    assert inp.state() == "s1"

    print(inp)

    action = InputAction(actions={
        Tape.N(0): [MoveLeftAction(on_symbol="A")],
        Tape.N(1): [WriteAction(on_symbol=Tape.BLANK, new_symbol="Z"), MoveLeftAction(on_symbol="Z")]
    })

    inp = action(source="s1", target="s2", data=inp)

    assert not inp.error
    assert inp.tapes[Tape.N(0)].pointer() == 0
    assert inp.tapes[Tape.N(1)].pointer() == 0

    assert inp.tapes[Tape.N(0)].read() == "D"
    assert inp.tapes[Tape.N(1)].read() == "Z"

    assert inp.state() == "s2"

    print(inp)

    action = InputAction(actions={
        Tape.N(0): [NoneAction(on_symbol="D")],
        Tape.N(1): [MoveRightAction(on_symbol="Z")]
    })

    inp = action(source="s2", target="s1", data=inp)

    assert not inp.error
    assert inp.tapes[Tape.N(0)].pointer() == 0
    assert inp.tapes[Tape.N(1)].pointer() == 1

    assert inp.tapes[Tape.N(0)].read() == "D"
    assert inp.tapes[Tape.N(1)].read() == "Z"

    assert inp.state() == "s1"

    print(inp)

    action = InputAction(actions={
        Tape.N(0): [NoneAction(on_symbol="D")],
        Tape.N(1): [MoveRightAction(on_symbol="Z")]
    })

    error_inp = action(source="s2", target="s1", data=inp)
    assert error_inp.error

    print(error_inp)

    action = InputAction(actions={
        Tape.N(0): [NoneAction(on_symbol="Z")],
        Tape.N(1): [MoveRightAction(on_symbol="Z")]
    })

    error_inp = action(source="s1", target="s2", data=inp)
    assert error_inp.error
    assert isinstance(error_inp.error_action[1], NoneAction)
    assert Tape.N(0) == error_inp.error_action[0]

    print(error_inp)


def test_stack_actions():
    stack = Stack(initial="s0", data="AAABBBBBCCCCC")

    action = PDAReadAction(on_symbol="A", stack_action=PushStackAction(on_symbol=Stack.EMPTY, new_symbol="Z"))
    stack = action(source="s0", target="s1", data=stack)

    assert not stack.error
    assert stack.tapes[Tape.N(0)].pointer() == 1
    assert stack.tapes[Tape.N(1)].pointer() == 1

    assert stack.head() == "A"
    assert stack.peek() == "Z"

    assert stack.state() == "s1"

    print(stack)

    action = PDAReadAction(on_symbol="A", stack_action=PushStackAction(on_symbol="Z", new_symbol="Z"))
    stack = action(source="s1", target="s1", data=stack)

    assert not stack.error
    assert stack.tapes[Tape.N(0)].pointer() == 2
    assert stack.tapes[Tape.N(1)].pointer() == 2

    assert stack.head() == "A"
    assert stack.peek() == "Z"

    assert stack.state() == "s1"

    print(stack)

    action = PDANullAction(stack_action=PushStackAction(on_symbol="Z", new_symbol="Y"))
    stack = action(source="s1", target="s1", data=stack)

    assert not stack.error
    assert stack.tapes[Tape.N(0)].pointer() == 2
    assert stack.tapes[Tape.N(1)].pointer() == 3

    assert stack.head() == "A"
    assert stack.peek() == "Y"

    assert stack.state() == "s1"

    print(stack)

    action = PDANullAction(stack_action=PopStackAction(on_symbol="Y"))
    stack = action(source="s1", target="s1", data=stack)

    assert not stack.error
    assert stack.tapes[Tape.N(0)].pointer() == 2
    assert stack.tapes[Tape.N(1)].pointer() == 2

    assert stack.head() == "A"
    assert stack.peek() == "Z"

    assert stack.state() == "s1"

    print(stack)

    action = PDANullAction(stack_action=PopStackAction(on_symbol="Y"))
    stack_error = action(source="s1", target="s1", data=stack)

    assert stack_error.error

    print(stack_error)

    action = PDAReadAction(on_symbol="A", stack_action=NullStackAction(on_symbol="Z"))
    stack = action(source="s1", target="s1", data=stack)

    assert not stack.error
    assert stack.tapes[Tape.N(0)].pointer() == 3
    assert stack.tapes[Tape.N(1)].pointer() == 2

    assert stack.head() == "B"
    assert stack.peek() == "Z"

    assert stack.state() == "s1"

    print(stack)


def test_buffer_actions():
    buffer = Buffer(initial="s0", data="AAABBBBBCCCCC")

    action = FAReadAction(on_symbol="A")
    buffer = action(source="s0", target="s1", data=buffer)

    assert not buffer.error
    assert buffer.tapes[Tape.N(0)].pointer() == 1
    assert buffer.pointer() == 1

    assert buffer.head() == "A"

    assert buffer.state() == "s1"

    print(buffer)

    action = FANullReadAction()
    buffer = action(source="s1", target="s1", data=buffer)

    assert not buffer.error
    assert buffer.tapes[Tape.N(0)].pointer() == 1
    assert buffer.pointer() == 1

    assert buffer.head() == "A"

    assert buffer.state() == "s1"

    print(buffer)

    action = FAReadAction(on_symbol="B")
    buffer_error = action(source="s1", target="s1", data=buffer)

    assert buffer_error.error

    print(buffer_error)

    action = FANullReadAction()
    buffer_error = action(source="s3", target="s1", data=buffer)

    assert buffer_error.error

    print(buffer_error)


def test_case_1_flang():
    a, b, c = symbols("a b c")
    n = symbols("n")
    lang = LanguageFormula(expression=[a**n, b**n, c**n], conditions=[n > 0])
    print(lang.enumerate_strings(length=9))

    grammar = OpenGrammar()

    grammar.add("S", "A")

    grammar.add("A", "aABC")
    grammar.add("A", "aBC")

    grammar.add("CB", "BC")

    grammar.add("aB", "ab")
    grammar.add("bB", "bb")
    grammar.add("bC", "bc")
    grammar.add("cC", "cc")

    assert lang.check_grammar(grammar, length=15)


def test_case_2_flang():
    a, b, c, d = symbols("a b c d")
    n, j = symbols("n j")
    lang = LanguageFormula(expression=[a**n, b**j, c**(2*n), d**j], conditions=[n >= 0, j > 0])

    print(lang.enumerate_strings(length=15))

    grammar = OpenGrammar()

    grammar.add("S", "A")

    grammar.add("A", "aAC")
    grammar.add("A", "B")

    grammar.add("B", "bBD")
    grammar.add("B", "bD")

    grammar.add("DC", "CD")
    grammar.add("bC", "bcc")
    grammar.add("ccC",  "cccc")

    grammar.add("ccD", "ccd")
    grammar.add("dD", "dd")
    grammar.add("bD", "bd")

    assert lang.check_grammar(grammar, length=15)


def context_sensitive_grammar_1():
    a, b, c, ccc = symbols("a b c ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    grammar.add("S", "Z")

    grammar.add("Z", "aZ")
    grammar.add("Z", "a")

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_2():
    a, b, c, ccc = symbols("a b c ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    grammar.add("S", "Z")

    grammar.add("Z", "aZb")
    grammar.add("Z", "ab")

    grammar.enumerate(length=15)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_3():
    a, b, c, ccc = symbols("a b c ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n, c ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    grammar.add("S", "Z")

    grammar.add("Z", "aZYX")
    grammar.add("Z", "abX")

    grammar.add("XY", "YX")

    grammar.add("bY", "bb")

    grammar.add("bX", "bc")
    grammar.add("cX", "cc")

    grammar.enumerate(length=15)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_4():
    a, b, c, ccc = symbols("a b c ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n, ccc ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    grammar.add("S", "Z")

    grammar.add("Z", "aZYX")
    grammar.add("Z", "abX")

    grammar.add("XY", "YX")

    grammar.add("bY", "bb")

    grammar.add("bX", "bccc")
    grammar.add("cccX", "cccccc")

    grammar.enumerate(length=15)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_5():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n, c ** n, d ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")
    buffer.run_rule("A", "aABCD", times=4)

    buffer.run_rule_until("A", "abCD")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    print(grammar)

    assert cfl.check_grammar(grammar, length=9)


def context_sensitive_grammar_6():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** m, b ** m, c ** n, d ** n], conditions=[n >= 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")
    buffer.run_rule("A", "aABCD", times=4)

    buffer.run_rule("A", "X")
    buffer.run_rule("X", "aXb", times=2)
    buffer.run_rule("X", "ab")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    print(grammar)

    assert cfl.check_grammar(grammar, length=9)


def context_sensitive_grammar_7():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** m, b ** m, c ** n, d ** n], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")
    buffer.run_rule("A", "aABCD", times=4)

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("AB", "aXW")

    buffer.run_rule("X", "aXW", times=3)
    buffer.run_rule("X", "W")

    buffer.run_rule("aW", "ab")
    buffer.run_rule_until("bW", "bb")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    print(grammar.enumerate(length=15))

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_8():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** m, b ** n, c ** n, d ** m], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")
    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("AB", "XWB")

    buffer.run_rule("X", "aXW", times=2)

    buffer.run_rule("aX", "aa")

    while True:
        try:
            buffer.run_rule("aW", "aD")

        except RuntimeError:
            break

        try:
            try:
                buffer.run_rule_until("DW", "WD")
            except RuntimeError:
                pass

            buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
            buffer.run_rule_until("DC", "CD")

        except RuntimeError:
            break

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    print(grammar.enumerate(length=10))

    assert cfl.check_grammar(grammar, length=10)


def context_sensitive_grammar_9():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** m, b ** m, c ** n, d ** n], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")
    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("AB", "XWB")

    buffer.run_rule("X", "aXW", times=1)

    buffer.run_rule("aX", "aa")

    while True:
        try:
            buffer.run_rule("aW", "aB")

        except RuntimeError:
            break

        try:
            try:
                buffer.run_rule_until("BW", "WB")
            except RuntimeError:
                pass

        except RuntimeError:
            break

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    print(grammar.enumerate(length=10))

    assert cfl.check_grammar(grammar, length=10)


def context_sensitive_grammar_10():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** m, c ** n, d ** m], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "aABCD")
    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rule_until("A", "X")
    buffer.run_rule("X", "BXD", times=3)
    buffer.run_rule("X", "BD")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    for each in grammar.enumerate(length=7):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=10)


def context_sensitive_grammar_11():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    context_sensitive_grammar_10()

    cfl = LanguageFormula(expression=[a ** n, b ** m, c ** m, d ** m], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "aABCD")

    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rule_until("A", "X")
    buffer.run_rule("X", "XBCD", times=3)
    buffer.run_rule("X", "BCD")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    for each in grammar.enumerate(length=7):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=7)


def context_sensitive_grammar_12():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    context_sensitive_grammar_10()

    cfl = LanguageFormula(expression=[a ** m, b ** m, c ** m, d ** n], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")

    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rule_until("aA", "aX")

    buffer.run_rule("X", "aXBC", times=3)
    buffer.run_rule("X", "aBC")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    for each in grammar.enumerate(length=15):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_13():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    context_sensitive_grammar_10()

    cfl = LanguageFormula(expression=[a ** m, b ** m, c ** n, d ** m], conditions=[n > 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "aABCD")

    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rule_until("aA", "aX")

    buffer.run_rule("X", "aXBD", times=3)
    buffer.run_rule("X", "aBD")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    for each in grammar.enumerate(length=15):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_14():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** m, b ** m, c ** n, d ** m], conditions=[n >= 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")

    buffer.add_rule("S", "X")

    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rule_until("aA", "aX")

    buffer.run_rule("X", "aXBD", times=3)
    buffer.run_rule("X", "aBD")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.add_rule("bD", "bd")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    for each in grammar.enumerate(length=15):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_15():
    print("[+] FD ")

    a, b, c, d, w, x, y = symbols("a b c d w x y")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[a ** m, b ** n, c ** m], conditions=[n >= 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")
    buffer.add_rule("S", "X")

    buffer.run_rule("A", "aABC", times=2)

    buffer.run_rule_until("aA", "aX")

    buffer.run_rule("X", "aXC", times=3)
    buffer.run_rule("X", "aC")

    buffer.run_rules_until([("CB", "BC")])

    buffer.run_rule_until("aB", "ab")

    buffer.add_rule("aC", "ac")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")

    for each in grammar.enumerate(length=15):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=15)

    cfl = LanguageFormula(expression=[x ** j, y ** j, w ** i], conditions=[i >= 0, j > i])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "X")

    buffer.add_rule("S", "Z")

    buffer.run_rule("X", "xXYW", times=2)

    buffer.run_rule_until("xX", "xZ")

    buffer.run_rule("Z", "xZY", times=3)
    buffer.run_rule("Z", "xY")

    buffer.run_rules_until([("WY", "YW")])

    buffer.run_rule("xY", "xy")

    buffer.run_rule_until("yY", "yy")
    buffer.run_rule_until("yW", "yw")

    buffer.run_rule_until("wW", "ww")

    for each in grammar.enumerate(length=15):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_16():
    a, b, c, d, w, x, y = symbols("a b c d w x y")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[a ** m, x ** j, b ** n, c ** m, y ** j, w ** i],
                          conditions=[n > 0, m > n, i > 0, j > i])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "AZ")

    buffer.run_rule("A", "aABC", times=3)

    buffer.run_rule("Z", "ZXYW", times=3)
    buffer.run_rule("Z", "TXYW")

    buffer.run_rule("A", "aABC", times=2)

    buffer.run_rule_until("aA", "aR")

    buffer.run_rule("R", "aRC", times=3)
    buffer.run_rule("R", "aC")

    buffer.run_rule("T", "XTY", times=3)
    buffer.run_rule("XT", "X")

    buffer.run_rules_until([("WX", "XW"), ("YX", "XY")])

    buffer.run_rules_until([("CX", "XC")])

    buffer.run_rules_until([("BX", "XB"), ("CX", "XC")])

    buffer.run_rules_until([("CB", "BC")])

    buffer.run_rules_until([("WY", "YW")])

    buffer.run_rule_until("aX", "ax")
    buffer.run_rule_until("xX", "xx")

    buffer.run_rule_until("xB", "xb")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")

    buffer.run_rule_until("cY", "cy")

    buffer.run_rule_until("yY", "yy")
    buffer.run_rule_until("yW", "yw")

    buffer.run_rule_until("wW", "ww")

    for each in grammar.enumerate(length=14):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=14)


def context_sensitive_grammar_17():
    a, b, c, d, cc = symbols("a b c d cc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** m, b, b ** m, cc, c ** n, d ** m], conditions=[n >= 0, m > n])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "A")

    buffer.add_rule("S", "X")

    buffer.run_rule("A", "aABCD", times=2)

    buffer.run_rule_until("aA", "aX")

    buffer.run_rule("X", "aXBD", times=3)
    buffer.run_rule("X", "aBD")

    buffer.run_rules_until([("DB", "BD"), ("CB", "BC")])
    buffer.run_rule_until("DC", "CD")

    buffer.run_rule_until("aB", "abb")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bccc")

    buffer.add_rule("bD", "bccd")

    buffer.run_rule_until("cC", "cc")
    buffer.run_rule_until("cD", "cd")

    buffer.run_rule_until("dD", "dd")

    for each in grammar.enumerate(length=15):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=15)


def context_sensitive_grammar_18():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    k, m, n = symbols("k m n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, c ** m],
                          conditions=[m >= 0, k >= 0, n > m])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "JM")
    buffer.run_rule("J", "aa")

    buffer.add_rule("S", "aaeb")

    buffer.run_rule("M", "NMC", times=3)
    buffer.run_rule("M", "NLC")

    buffer.run_rule("N", "eN", times=3)

    buffer.run_rule_until("eN", "ee")

    buffer.run_rule_until("eL", "eb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")

    buffer.reset()

    buffer.run_rule("S", "JNL")
    buffer.run_rule("J", "aa")

    buffer.run_rule("N", "eN", times=7)

    buffer.run_rule_until("eN", "ee")

    buffer.run_rule_until("eL", "eb")

    buffer.reset()

    buffer.run_rule("S", "JAZ")
    buffer.run_rule("J", "aa")

    buffer.add_rule("S", "JZ")

    buffer.run_rule("A", "aaABC", times=3)
    buffer.run_rule("A", "aaBC")

    buffer.run_rule("Z", "YZC", times=3)
    buffer.run_rule("Z", "YLC")

    buffer.run_rule("Y", "EY", times=3)

    buffer.run_rule_until("EY", "EE")

    buffer.run_rules_until([("CE", "EC"), ("BE", "EB")])
    buffer.run_rules_until([("CB", "BC")])

    buffer.run_rules_until([("CL", "LC"), ("BL", "LB")])
    buffer.run_rules_until([("CB", "BC")])

    buffer.run_rule_until("aaE", "aae")

    buffer.run_rule_until("eE", "ee")

    buffer.run_rule_until("eL", "eb")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")

    buffer.reset()

    buffer.run_rule("S", "JAYL")
    buffer.add_rule("S", "JAEL")

    buffer.run_rule("J", "aa")

    buffer.run_rule("Y", "EY", times=3)
    buffer.run_rule_until("EY", "EE")

    buffer.run_rule("A", "aaABC", times=3)
    buffer.run_rule("A", "aaBC")

    buffer.run_rules_until([("CB", "BC")])

    buffer.run_rules_until([("CE", "EC"), ("BE", "EB")])
    buffer.run_rules_until([("CL", "LC"), ("BL", "LB")])

    buffer.run_rule_until("aaE", "aae")

    buffer.run_rule_until("eE", "ee")

    buffer.run_rule_until("eL", "eb")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")

    for each in grammar.enumerate(length=10):
        grammar.print_stack(each)

    assert cfl.check_grammar(grammar, length=10)

    print(grammar)


def context_sensitive_grammar_19():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    k, m, n = symbols("k m n")

    cfl = LanguageFormula(expression=[a ** n, b ** n, a ** n],
                          conditions=[n > 0])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "RX")

    buffer.run_rule("R", "aRBC", times=3)
    buffer.run_rule("R", "aB")

    buffer.run_rule_until("CB", "BC")
    buffer.run_rule_until("CX", "XC")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bX", "ba")

    buffer.run_rule_until("aC", "aa")

    assert cfl.check_grammar(grammar, length=10)


def test_language_turing_machine_1():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k >= 0, m >= 0, n >= 0])

    cfl.info()

    lang_machine = TuringParser(language=cfl)
    lang_machine.info()

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)

        assert read_status


def test_language_turing_machine_2():
    print("[+] FD ")
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k >= 0, m >= 0, Eq(n, m)])

    cfl.info()
    lang_machine = TuringParser(language=cfl)

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_3():
    a, e, b, c, aa, ccc, bb = symbols("a e b c aa ccc bb")
    m, k, n, p, q = symbols("m k n p q")

    cfl_with_ones = LanguageFormula(expression=[aa, aa ** k, aa ** q, e ** n, b, b ** k,
                                                bb ** p, c ** k, ccc ** m, c ** n, b ** m],
                                    conditions=[k >= 0, m >= 0, Eq(n, m + 1), Eq(p + 1, k), Eq(q, m)])

    cfl = LanguageFormula(expression=[aa, aa ** k, aa ** q, e ** n, b, b ** k,
                                      bb ** p, c ** k, ccc ** m, c ** n, b ** m],
                          conditions=[k >= 0, m >= 0, Eq(n, m), Eq(p, k), Eq(q, m)])

    cfl.info()
    difference = set(cfl_with_ones.enumerate_strings(length=25)).difference(cfl.enumerate_strings(length=25))

    lang_machine = TuringParser(language=cfl)
    lang_machine.info()

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    for data in difference:
        print("[+] testing string from ones language", data)

        read_status = lang_machine.turing.read(data)

        if read_status:
            lang_machine.turing.debug(data)

        assert not read_status


def test_language_turing_machine_4():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl_with_null = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                                    conditions=[k >= 0, m >= 0, n >= 0])

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k > 0, m > 0, n > 0])
    cfl.info()

    difference = set(cfl_with_null.enumerate_strings(length=25)).difference(cfl.enumerate_strings(length=25))

    lang_machine = TuringParser(language=cfl)
    lang_machine.info()

    for data in cfl.enumerate_strings(length=25):
        print("[+] testing string", data)

        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)

        assert read_status

    for data in difference:
        print("[+] testing string from nulled language", data)
        read_status = lang_machine.turing.read(data)

        if read_status:
            lang_machine.turing.debug(data)

        assert not read_status


def test_language_turing_machine_5():
    a, e, b, c, aa, ccc, ab, d, f = symbols("a e b c aa ccc ab d f")
    m, k, n, q = symbols("m k n q")

    cfl = LanguageFormula(expression=[e ** q, a ** k, e ** n, ab ** k, b ** k, c ** m, b ** m, ccc ** n, f ** q],
                          conditions=[k > 0, q > 0, Eq(k + q, m + n)])
    cfl.info()

    print("\n[+] language machine")

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_6():
    a, e, b, c, aa, ccc, ab, d, f = symbols("a e b c aa ccc ab d f")
    m, k, n, q, r, s = symbols("m k n q r s")

    cfl = LanguageFormula(expression=[e ** q, a ** k, e ** n, ab ** k, b ** r,
                                      c ** m, b ** m, ccc ** n, f ** q, aa ** s],
                          conditions=[k > 0, q > 0, Eq(k, m + n), Eq(r + s, q)])
    cfl.info()

    print("\n[+] language machine")

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_7():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n < m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n <= m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n > m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    cfl = LanguageFormula(expression=[a ** k, e ** n, c ** m],
                          conditions=[k >= 0, m > 0, n >= m])

    cfl.info()

    print(cfl.enumerate_strings(length=10))

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=10):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_8():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** (2 * k + 1), e ** n, b ** (k + 2), c ** (k + 3 * m)],
                          conditions=[k >= 0, m > 0, n < m])

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    for data in cfl.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_9():
    n, j = symbols("n j")
    ac, b, c, d = symbols("ac b c d")

    lang = LanguageFormula(expression=[ac ** (n + 1), b ** j, c ** n, d ** j], conditions=[n >= 0, j > n])

    print(lang.enumerate_strings(length=15))

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_10():
    n, k, d = symbols("n k d")
    a, d, b, c, f = symbols("a d b c f")

    lang = LanguageFormula(expression=[a ** (k + n), d ** n, b ** (k + 1), c ** k, f ** d],
                           conditions=[n >= 0, k >= 0, d < n, d >= 0])

    print(lang.enumerate_strings(length=15))

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_turing_machine_11():
    s, n, k = symbols("s n k")
    a, d, b, e = symbols("a d b e")

    lang = LanguageFormula(expression=[a ** (2 * n), d ** (s + 1), b ** k, e ** n],
                           conditions=[s >= 0, n > 0, k > 0, ~Eq(n, k)])

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=20):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_union_turing_machine_12():
    n, j, i = symbols("n j i")
    aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i], conditions=[n > 0, j > n, i > 0])
    lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** (2 * n + 1), e ** i], conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=15):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_union_turing_machine_13():
    n, j, i, r = symbols("n j i r")
    aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])
    lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** (2 * n + 1), e ** i, c ** n, c ** i],
                             conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=25):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status


def test_language_union_turing_machine_14():
    n, j, i, r = symbols("n j i r")
    aab, aba, ca, daa, eaa, a, b, c, d, e = symbols("aab aba ca daa eaa a b c d e")

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])
    lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** (2 * n), e ** i, c ** n, c ** i],
                             conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=25):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
                             conditions=[n > 0, j > n, i > 0, r >= 0])

    lang_b = LanguageFormula(expression=[b ** n, a ** j, d ** (2 * n), e ** i, c ** n, c ** i],
                             conditions=[n > 0, j > n, i > 0])

    lang = lang_a + lang_b

    lang.info()

    lang_machine = TuringParser(language=lang)

    lang_machine.info()

    for data in lang.enumerate_strings(length=25):
        print("[+] testing string", data)
        read_status = lang_machine.turing.read(data)

        if not read_status:
            lang_machine.turing.debug(data)
            lang_machine.info()

        assert read_status

    # lang_a = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j, a ** r, c ** r],
    #                          conditions=[n > 0, j > n, i > 0, r >= 0])
    #
    # lang_b = LanguageFormula(expression=[b ** n, a ** j, c ** n, d ** i, c ** j],
    #                          conditions=[n > 0, j > n, i > 0, r >= 0])
    #
    # lang = lang_a + lang_b
    #
    # lang.info()
    #
    # lang_machine = TuringParser(language=lang)
    #
    # lang_machine.info()
    #
    # for data in lang.enumerate_strings(length=25):
    #     print("[+] testing string", data)
    #     read_status = lang_machine.turing.read(data)
    #
    #     if not read_status:
    #         lang_machine.turing.debug(data)
    #         lang_machine.info()
    #
    #     assert read_status


def test_planner_turing_machine_15():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    print("[+] parse lone symbol")

    tester = PlanTester(plan=ParseLoneSymbol(block=0),
                        language=LanguageFormula(expression=[ccc], conditions=[]))

    assert tester.test({C(0): "ccc"}, word="ccc")
    assert not tester.test({C(0): "cdc"}, word="ccc")

    print("[+] parse accumulate test")

    tester = PlanTester(plan=ParseAccumulate(block=0, tape=1),
                        language=LanguageFormula(expression=[a**n], conditions=[n >= 0]))

    assert tester.test({C(0): "aaaaaaa"}, word="a", checker=lambda i: i.tapes[C(1)].data() == "ZZZZZZZB0")

    assert not tester.test({C(0): "aaaabaaa"}, word="a")

    print("[+] parse equal test")

    tester = PlanTester(plan=ParseEqual(block=0, tape=1),
                        language=LanguageFormula(expression=[a**n], conditions=[n >= 0]))

    assert tester.test({C(0): "aaa", C(1): "XZZZ"}, word="a")
    assert tester.test({C(0): "ababab", C(1): "XZZZ"}, word="ab")
    assert not tester.test({C(0): "ababab", C(1): "XZZ"}, word="ab")

    tester = PlanTester(plan=AddTapes(source_tape=0, target_tape=1),
                        language=LanguageFormula(expression=[a**n], conditions=[n >= 0]))

    assert tester.test({C(0): "XZZZ", C(1): "XZZZ"}, checker=lambda i: i.tapes[C(1)].data() == "XZZZZZZB0")


def test_planner_turing_machine_16():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** n],
                          conditions=[k >= 0, n >= 0])

    planner = TuringPlanner(language=cfl, tapes=2)

    planner.add_plan(ParseLoneSymbol(block=0))
    planner.add_plan(ParseAccumulate(block=1, tape=1))

    planner.add_plan(ParseAccumulate(block=2, tape=2))

    planner.add_plan(ParseLoneSymbol(block=3))
    planner.add_plan(ParseEqual(block=4, tape=1))
    planner.add_plan(ParseEqual(block=5, tape=1))

    planner.add_plan(ParseEqual(block=6, tape=2))

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("aaeebcccccc")


def testing_turing_language():
    test_language_turing_machine_1()
    test_language_turing_machine_2()
    test_language_turing_machine_3()
    test_language_turing_machine_4()
    test_language_turing_machine_5()
    test_language_turing_machine_6()
    test_language_turing_machine_7()
    test_language_turing_machine_8()
    test_language_turing_machine_9()
    test_language_turing_machine_10()
    test_language_turing_machine_11()

    test_language_union_turing_machine_12()
    test_language_union_turing_machine_13()
    test_language_union_turing_machine_14()

    test_planner_turing_machine_15()
    test_planner_turing_machine_16()

    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** (2 * k + 2), e ** n, b ** (k + 1), c ** (k + 3 * m)],
                          conditions=[k >= 0, m >= 0, n > m])

    lang_machine = TuringParser(language=cfl)

    lang_machine.info()

    assert not lang_machine.turing.read("aaaaaaeeeeebccccc")  # it shouldn't detect it


def test_grammar_tree_1():
    a, b, c, d, w, x, y, aa = symbols("a b c d w x y aa")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[aa, a ** n, d ** j, x ** i, b ** n, w, c ** j, w ** i],
                          conditions=[n > 0, i > 0, j > 0])

    tree = GrammarTree(language=cfl)

    print("[+] non terminals", tree.generate_non_terminal())
    print("[+] terminals", tree.generate_with_terminals())

    tree.info()

    assert cfl.check_grammar(tree.grammar, length=15)


def test_grammar_tree_2():
    a, b, c, d, w, x, y, aa = symbols("a b c d w x y aa")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[a, a ** n, d ** j, x ** i, b ** n, w, c ** j, w ** i, d ** n],
                          conditions=[n > 0, i > 0, j > 0])

    tree = GrammarTree(language=cfl)

    print("[+] non terminals", tree.generate_non_terminal())
    print("[+] terminals", tree.generate_with_terminals())

    tree.info()

    assert cfl.check_grammar(tree.grammar, length=10)


def test_grammar_tree_3():
    a, b, c, d, w, x, y, aa = symbols("a b c d w x y aa")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[a, a ** n, d ** n],
                          conditions=[n > 0])

    tree = GrammarTree(language=cfl)

    print("[+] non terminals", tree.generate_non_terminal())
    print("[+] terminals", tree.generate_with_terminals())

    tree.info()

    assert cfl.check_grammar(tree.grammar, length=10)


def test_grammar_tree_4():
    a, b, c, d, w, x, y, aa = symbols("a b c d w x y aa")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[a, a ** n, y ** j],
                          conditions=[n > 0, j > 0])

    tree = GrammarTree(language=cfl)

    print("[+] non terminals", tree.generate_non_terminal())
    print("[+] terminals", tree.generate_with_terminals())

    tree.info()

    assert cfl.check_grammar(tree.grammar, length=10)


def test_grammar_tree_5():
    a, b, c, d, w, x, y, aa = symbols("a b c d w x y aa")
    m, n, i, j = symbols("m n i j")

    cfl = LanguageFormula(expression=[a ** n, b, x ** i, b ** n, c ** n, w ** i, y ** i],
                          conditions=[n > 0, i > 0])

    tree = GrammarTree(language=cfl)

    print("[+] non terminals", tree.generate_non_terminal())
    print("[+] terminals", tree.generate_with_terminals())

    tree.info()

    print(tree.grammar.enumerate(length=15))
    assert cfl.check_grammar(tree.grammar, length=15)


def test_sentinel_grammar():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n, a ** n],
                          conditions=[n > 0])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "RX")

    buffer.run_rule("R", "aRBC", times=3)
    buffer.run_rule("R", "aB")

    buffer.run_rule_until("CB", "BC")
    buffer.run_rule_until("CX", "XC")

    buffer.run_rule_until("aB", "ab")

    buffer.run_rule_until("bB", "bb")
    buffer.run_rule_until("bX", "ba")

    buffer.run_rule_until("aC", "aa")

    assert cfl.check_grammar(grammar, length=10)


def test_genesis_leafs():
    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n, c ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "Z")
    buffer.run_rule("Z", "aZBC", times=4)
    buffer.run_rule("Z", "aBC")

    buffer.run_rules_until([("CB", "BC")])

    buffer.run_rule_until("aB", "ab")
    buffer.run_rule_until("bB", "bb")

    buffer.run_rule_until("bC", "bc")

    buffer.run_rule_until("cC", "cc")

    assert cfl.check_grammar(grammar, length=10)

    a, b, c, d = symbols("a b c d")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[a ** n, b ** n], conditions=[n > 0])

    grammar = OpenGrammar()

    buffer = grammar.get_string()

    buffer.run_rule("S", "Z")
    buffer.run_rule("Z", "aZB", times=4)
    buffer.run_rule("Z", "aB")

    buffer.run_rule_until("aB", "ab")
    buffer.run_rule_until("bB", "bb")

    assert cfl.check_grammar(grammar, length=10)


def grammar_case_1():
    a, b, c, d, w, x, y, aa, f = symbols("a b c d w x y aa f")
    m, n, i, j, k = symbols("m n i j k")

    cfl = LanguageFormula(expression=[a ** k, d ** n, b, b ** k, c ** k, f ** d],
                          conditions=[k >= 0, d < n, d >= 0])

    cfl.info()

    grammar = OpenGrammar()

    grammar.add("S", "RN")
    grammar.add("S", "J")
    grammar.add("S", "K")

    grammar.add("R", "aRPO")
    grammar.add("R", "aPO")

    grammar.add("OP", "PO")

    grammar.add("K", "dK")

    grammar.add("J", "dJL")
    grammar.add("J", "JM")
    grammar.add("J", "M")

    grammar.add("N", "MNL")
    grammar.add("N", "NM")
    grammar.add("N", "M")

    grammar.add("PM", "MP")
    grammar.add("OM", "MO")

    grammar.add("aM", "ad")

    grammar.add("dM", "dd")
    grammar.add("dP", "dbb")
    grammar.add("dL", "dbf")
    grammar.add("dK", "db")

    grammar.add("bP", "bb")
    grammar.add("bO", "bc")
    grammar.add("bL", "bf")

    grammar.add("cO", "cc")
    grammar.add("cL", "cf")

    grammar.add("fL", "ff")

    print(grammar)
    print(grammar.enumerate(length=10))

    assert cfl.check_grammar(grammar, length=10)


def grammar_case_2():
    a, b, c, d, w, x, y, aa, f = symbols("a b c d w x y aa f")
    m, n, i, j, k = symbols("m n i j k")

    lang_a = LanguageFormula(expression=[a ** k, d ** n, b, b ** k, c ** k, f ** d],
                             conditions=[n > 0, k >= 0, d < n, d >= 0])

    lang_b = LanguageFormula(expression=[a ** k, b, b ** k, c ** k],
                             conditions=[k >= 0])

    cfl = lang_a + lang_b

    cfl.info()

    grammar = OpenGrammar()

    grammar.add("S", "RN")
    grammar.add("S", "RT")
    grammar.add("S", "R")
    grammar.add("S", "J")
    grammar.add("S", "K")

    grammar.add("S", "b")

    grammar.add("R", "aRPO")
    grammar.add("R", "aPO")

    grammar.add("OP", "PO")

    grammar.add("K", "dK")

    grammar.add("J", "dJL")
    grammar.add("JL", "TL")

    grammar.add("N", "MNL")
    grammar.add("NL", "TL")

    grammar.add("T", "TM")
    grammar.add("T", "M")

    grammar.add("PM", "MP")
    grammar.add("OM", "MO")

    grammar.add("aM", "ad")
    grammar.add("aP", "abb")

    grammar.add("dM", "dd")
    grammar.add("dP", "dbb")
    grammar.add("dL", "dbf")
    grammar.add("dK", "db")

    grammar.add("bP", "bb")
    grammar.add("bO", "bc")
    grammar.add("bL", "bf")

    grammar.add("cO", "cc")
    grammar.add("cL", "cf")

    grammar.add("fL", "ff")

    print(grammar)
    print(grammar.enumerate(length=15))

    grammar.print_stack("aaadddbbbbcccff")
    grammar.print_stack("ddddddddbffffff")
    grammar.print_stack("dddddddddb")
    grammar.print_stack("aaaadbbbbbcccc")
    grammar.print_stack("aaaabbbbbcccc")

    assert cfl.check_grammar(grammar, length=15)


def grammar_case_3():
    a, b, c, d, w, x, y, aa, f, e, ccc = symbols("a b c d w x y aa f e ccc")
    m, n, i, j, k = symbols("m n i j k")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k >= 0, m >= 0, m < n])

    grammar = OpenGrammar()

    grammar.add("S", "aaRN")
    grammar.add("S", "aaRT")
    grammar.add("S", "aaN")
    grammar.add("S", "aaJ")

    grammar.add("R", "RQPO")
    grammar.add("R", "QPO")

    grammar.add("OQ", "QO")
    grammar.add("OP", "PO")
    grammar.add("PQ", "QP")
    grammar.add("PM", "MP")
    grammar.add("OM", "MO")

    grammar.add("N", "MNL")
    grammar.add("NL", "TL")

    grammar.add("J", "eJ")

    grammar.add("T", "TM")
    grammar.add("T", "M")

    grammar.add("aaQ", "aaaa")
    grammar.add("aaM", "aae")
    grammar.add("aaP", "aabb")

    grammar.add("eM", "ee")
    grammar.add("eP", "ebb")
    grammar.add("eL", "ebccc")

    grammar.add("eJ", "eb")

    grammar.add("bP", "bb")
    grammar.add("bO", "bc")

    grammar.add("cO", "cc")
    grammar.add("cL", "cccc")
    grammar.add("cccL", "cccccc")

    print(grammar)

    print(grammar.enumerate(length=15))

    grammar.print_stack("aaeeeeebcccccc")
    grammar.print_stack("aaeeeeeeeeeeeb")
    grammar.print_stack("aaaaaaeebbbcc")
    grammar.print_stack("aaaaaaeeebbbcccccccc")

    assert cfl.check_grammar(grammar, length=10)


def turing_case_1():
    a, e, b, c, aa, ccc = symbols("a e b c aa ccc")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[aa, aa ** k, e ** n, b, b ** k, c ** k, ccc ** m],
                          conditions=[k >= 0, m >= 0, m < n])

    planner = TuringPlanner(language=cfl, tapes=2)

    planner.add_plan(ParseLoneSymbol(block=0))
    planner.add_plan(ParseAccumulate(block=1, tape=1))

    planner.add_plan(ParseAccumulate(block=2, tape=2))

    planner.add_plan(ParseLoneSymbol(block=3))

    planner.add_plan(ParseEqual(block=4, tape=1))
    planner.add_plan(ParseEqual(block=5, tape=1))

    planner.add_plan(ParseStrictLessEqual(block=6, tape=2))

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("aaaaeeeeebbccccccc")

    for data in cfl.enumerate_strings(length=22):
        print("[+] testing string", data)
        read_status = machine.turing.read(data)

        if not read_status:
            machine.turing.debug(data)
            machine.info()

        assert read_status

    machine.info()


def turing_function_case_1():
    x, y, i1, i0 = symbols("x y 1 0")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[i1 ** n, i0, i1 ** m],
                          conditions=[n >= 0, m >= 0])

    planner = TuringPlanner(language=cfl, tapes=4)

    planner.machine_plan = [
        ParseAction(block=0, actions=[Copy(tapes=[1, 2])]),

        ParseLoneSymbol(block=1),

        ParseAction(block=2, actions=[Subtract(tapes=[1, 2], symbol="1")]),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1"),

        AddWithFactorTapes(target_tape=4, source_tape=3, multiplier=3, symbol="1"),

        WipeTapes(tapes=[1, 2], symbol="1"),

        AddTapes(target_tape=[1, 2], source_tape=0, symbol="1", stop="0"),

        MultiplyTapes(result=4, one_tape=1, other_tape=2, symbol="1")
    ]

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("111110111")

    machine.info()

    TuringPlotter.to_csv("function.csv", machine.turing)


def turing_function_case_2():
    x, y, i1, i0 = symbols("x y 1 0")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[i1 ** n, i0, i1 ** m],
                          conditions=[n >= 0, m >= 0])

    planner = TuringPlanner(language=cfl, tapes=3)

    planner.machine_plan = [
        ParseAction(block=0, actions=[Copy(tapes=[1, 2])]),

        ParseLoneSymbol(block=1),

        ParseAction(block=2, actions=[Subtract(tapes=[1, 2], symbol="1")]),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1"),
        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1"),
        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1"),

        WipeTapes(tapes=[1, 2], symbol="1"),

        AddTapes(target_tape=[1, 2], source_tape=0, symbol="1", stop="0"),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1")
    ]

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("111110111")

    machine.info()


def turing_function_case_3():
    x, y, i1, i0 = symbols("x y 1 0")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[i1 ** n, i0, i1 ** m],
                          conditions=[n >= 0, m > 0])

    planner = TuringPlanner(language=cfl, tapes=3)

    planner.machine_plan = [
        ParseAction(block=0, actions=[Copy(tapes=[1, 2])]),

        ParseLoneSymbol(block=1),

        ParseAction(block=2, actions=[Subtract(tapes=[1, 2], symbol="1")]),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1", multiplier=3),

        WipeTapes(tapes=[1, 2], symbol="1"),

        AddTapes(target_tape=[1, 2], source_tape=0, symbol="1", stop="0"),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1")
    ]

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("111110111")

    machine.info()

    TuringPlotter.to_csv("function.csv", machine.turing)


def turing_function_case_4():
    x, y, i1, i0 = symbols("x y 1 0")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[i1 ** n, i0, i1 ** m],
                          conditions=[n >= 0, m >= 0])

    planner = TuringPlanner(language=cfl, tapes=4)

    planner.machine_plan = [
        ParseAction(block=0, actions=[Copy(tapes=[1, 2])]),

        ParseLoneSymbol(block=1),

        ParseAction(block=2, actions=[Subtract(tapes=[1, 2], symbol="1")]),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1"),

        WipeTapes(tapes=[1, 2], symbol="1"),

        AddTapes(target_tape=[1, 2], source_tape=0, symbol="1", stop="0"),

        MultiplyTapes(result=3, one_tape=1, other_tape=2, symbol="1")
    ]

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("111101")


def run_cases():
    print("[+] running test case 1")
    assert test_case_1("00111011110", run_grammar=True)
    assert test_case_1("00")
    assert test_case_1("11")
    assert test_case_1("001110011010")
    assert test_case_1("010101010101010100")
    assert not test_case_1("0101010101010101")
    assert not test_case_1("")
    assert not test_case_1("1")
    assert not test_case_1("0")

    print("[+] running test case 2")
    assert test_case_2("", run_grammar=True)
    assert test_case_2("ababa")
    assert test_case_2("aa")
    assert test_case_2("bb")
    assert test_case_2("b")
    assert test_case_2("a")

    print("[+] running test case 3")
    assert test_case_3("10010101", run_grammar=True)
    assert not test_case_3("1110010101")
    assert test_case_3("10010101")
    assert test_case_3("1")
    assert test_case_3("10010101")
    assert not test_case_3("010101")

    print("[+] running test case 4")
    assert test_case_4("1", run_grammar=True)
    assert test_case_4("00")
    assert test_case_4("00001")
    assert test_case_4("0000001")
    assert not test_case_4("1111")

    print("[+] running test case 5")
    assert test_case_5("badddcbdb", run_grammar=True)
    assert test_case_5("ddb")
    assert test_case_5("ddddb")
    assert test_case_5("b")
    assert test_case_5("bdd")
    assert test_case_5("bbddb")
    assert test_case_5("bdbdb")
    assert not test_case_5("")
    assert not test_case_5("dd")
    assert not test_case_5("dddb")
    assert not test_case_5("ddbaddc")
    assert not test_case_5("baddc")
    assert not test_case_5("baddcdd")
    assert test_case_5("bddcdd")

    print("[+] running test case 9")
    assert test_case_9("1", run_grammar=True)
    assert test_case_9("00")
    assert test_case_9("00001")
    assert test_case_9("0000001")
    assert not test_case_9("1111")

    print("[+] running test case 11")
    assert test_case_11("00111011110", run_grammar=True)
    assert test_case_11("00")
    assert test_case_11("11")
    assert test_case_11("001110011010")
    assert test_case_11("010101010101010100")
    assert not test_case_11("0101010101010101")
    assert not test_case_11("")
    assert not test_case_11("1")
    assert not test_case_11("0")

    print("[+] running test case 12")
    assert test_case_12("00111011110", run_grammar=True)
    assert test_case_12("00")
    assert test_case_12("11")
    assert test_case_12("001110011010")
    assert test_case_12("010101010101010100")
    assert not test_case_12("0101010101010101")
    assert not test_case_12("")
    assert not test_case_12("1")
    assert not test_case_12("0")

    print("[+] running test case 13")
    assert test_case_13("c", run_grammar=True)
    assert test_case_13("d")
    assert test_case_13("ccd")
    assert test_case_13("aaccd")
    assert not test_case_13("aaaccd")
    assert not test_case_13("aaaccdd")
    assert not test_case_13("aaaccdc")
    assert not test_case_13("")

    print("[+] running test case 14")
    test_case_14()

    print("[+] running test case 15")
    test_case_15()

    print("[+] running test case 16")
    test_case_16()

    print("[+] running test case for constraints")
    test_constraints()

    print("[+] running test case PDA 1")
    test_pda_case_1()

    print("[+] testing input actions")
    test_input_actions()

    print("[+] testing buffer actions")
    test_buffer_actions()

    print("[+] testing stack actions")
    test_stack_actions()

    print("[+] testing transitions")
    test_transitions()

    print("[+] testing formula language 1")
    test_case_1_flang()

    print("[+] testing formula language 2")
    test_case_2_flang()

    print("[+] testing context sensitive grammars")
    context_sensitive_grammar_1()
    context_sensitive_grammar_2()
    context_sensitive_grammar_4()
    context_sensitive_grammar_5()
    context_sensitive_grammar_6()
    context_sensitive_grammar_7()
    context_sensitive_grammar_8()
    context_sensitive_grammar_9()
    context_sensitive_grammar_10()
    context_sensitive_grammar_11()
    context_sensitive_grammar_12()
    context_sensitive_grammar_13()
    context_sensitive_grammar_14()
    context_sensitive_grammar_15()
    context_sensitive_grammar_16()
    context_sensitive_grammar_17()
    context_sensitive_grammar_18()

    print("[+] running turing machine builder test")
    testing_turing_language()

    print("[+] testing CSG")
    test_grammar_tree_1()
    test_grammar_tree_2()
    test_grammar_tree_3()
    test_grammar_tree_4()
    test_grammar_tree_5()

    print("[+] testing turing machine function")
    turing_function_case_1()
    turing_function_case_2()
    turing_function_case_3()
    turing_function_case_4()


run_cases()
