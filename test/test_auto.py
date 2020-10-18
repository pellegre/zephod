from pyauto.language import *
from pyauto.automata.finite import *
from pyauto.automata.pushdown import *


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


run_cases()
