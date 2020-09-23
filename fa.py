from pyauto.finite_automata import *
from utils.builder import *
from pyauto.grammar import *
from pyauto.language import *
from pyauto.pushdown_automata import *


import copy


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

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plotter:
        AutomataPlotter.plot(nfsm)
        AutomataPlotter.plot(minimized)

    return rebased_nfsm_value


def test_case_2(inp, plot=False, run_grammar=False):
    transition = FADelta()
    transition.add("s0", "s0", {"a"})
    transition.add("s0", "s2", {NullTransition.SYMBOL})
    transition.add("s1", "s2", {"a", "b"})
    transition.add("s2", "s0", {"a"})
    transition.add("s2", "s1", {NullTransition.SYMBOL})

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

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(nfsm)
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(stripped)
        AutomataPlotter.plot(minimized)

    return nfsm_value


def test_case_3(inp, plot=False, run_grammar=False):
    expr = (Z("1") + Z("10")) | ~Z("01")
    assert expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_4(inp, plot=False, run_grammar=False):
    expr = Z("00") + (~Z("0") | Z("1"))
    assert expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_5(inp, plot=False, run_grammar=False):
    expr = FiniteAutomataBuilder.get_finite_automata_from_csv("./csv/zuto.csv")
    assert not expr.has_null_transitions()

    expr_value = expr.read(inp)

    dfsm = expr.get_deterministic_automata()
    assert not dfsm.is_non_deterministic()

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == expr_value

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(minimized)

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_9(inp, plot=False, run_grammar=False):
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

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_11(inp, plot=False, run_grammar=False):
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

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert nfsm.read(grammar_from_fda(length=j))

    assert not nfsm.has_null_transitions()
    dfsm = nfsm.get_deterministic_automata()

    expr_value = dfsm.read(inp)

    assert not dfsm.is_non_deterministic()

    pda = dfsm.get_pushdown_automata()
    pda_value = pda.read(inp)
    assert pda_value == expr_value

    if plot:
        AutomataPlotter.plot(dfsm)

    return expr_value


def test_case_12(inp, plot=False, run_grammar=False):
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

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_13(inp, plot=False, run_grammar=False):
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

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(expr)
        AutomataPlotter.plot(stripped)
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_14(plot=False, run_grammar=False):
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

    for i in range(500):
        for j in [1, 5, 10, 20, 30]:
            assert fda.read(g(length=j))

    grammar_from_fda = Grammar.build_from_finite_automata(fda)

    for i in range(500):
        for j in [1, 5, 10, 20, 30]:
            assert fda.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(fda)


def test_case_15(plot=False):
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
    for i in range(500):
        for j in [1, 5, 10, 20, 30]:
            w = grammar_from_fda(length=j)
            assert minimized.read(w)
            if not len(w):
                got_null_string = True

    assert got_null_string

    if plot:
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

    transition = FADelta()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    final = [State("e3")]

    expr = (~(Z("aa") + Z("b")) | (Z("c") + Z("d")) | ~Z("cd"))
    transition = expr.transition
    final = {State("z18")}
    initial = expr.initial

    buffers, accepted = [Buffer(data="aaccd", initial=initial)], None
    while not all(map(lambda b: b.consumed and b.done, buffers)):
        parsed = set()
        for buffer in filter(lambda b: not b.done, buffers):
            parsed.update(transition(buffer))

        buffers = [buffer for buffer in parsed]

        consumed_in_final = list(map(lambda b: b.consumed and b.done and b.state() in final, buffers))
        if any(consumed_in_final):
            accepted = buffers[consumed_in_final.index(True)]

    assert accepted


def test_case_16():
    expr = (~Z("aaa") | ~Z("bb") | ~Z("cd")).minimal()

    transition = FADelta()
    transition.add("z0", "z1", {~Z("aaa")})
    transition.add("z1", "z2", {~Z("bb")})
    transition.add("z2", "z3", {~Z("cd")})

    fda = FiniteAutomata(transition, initial="z0", final={"z0", "z1", "z2", "z3"})

    grammar_from_fda = Grammar.build_from_finite_automata(expr)

    for i in range(500):
        for j in [1, 5, 10, 20, 30]:
            s = grammar_from_fda(length=j)
            assert fda.read(s)


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

    for i in range(500):
        for j in [1, 5, 10, 20, 30]:
            s = grammar(length=j)
            assert pda.read(s)


def run_cases():
    assert test_case_1("00111011110", run_grammar=True)
    assert test_case_1("00")
    assert test_case_1("11")
    assert test_case_1("001110011010")
    assert test_case_1("010101010101010100")
    assert not test_case_1("0101010101010101")
    assert not test_case_1("")
    assert not test_case_1("1")
    assert not test_case_1("0")

    assert test_case_2("", run_grammar=True)
    assert test_case_2("ababa")
    assert test_case_2("aa")
    assert test_case_2("bb")
    assert test_case_2("b")
    assert test_case_2("a")

    assert test_case_3("10010101", run_grammar=True)
    assert not test_case_3("1110010101")
    assert test_case_3("10010101")
    assert test_case_3("1")
    assert test_case_3("10010101")
    assert not test_case_3("010101")

    assert test_case_4("1", run_grammar=True)
    assert test_case_4("00")
    assert test_case_4("00001")
    assert test_case_4("0000001")
    assert not test_case_4("1111")

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

    assert test_case_9("1", run_grammar=True)
    assert test_case_9("00")
    assert test_case_9("00001")
    assert test_case_9("0000001")
    assert not test_case_9("1111")

    assert test_case_11("00111011110", run_grammar=True)
    assert test_case_11("00")
    assert test_case_11("11")
    assert test_case_11("001110011010")
    assert test_case_11("010101010101010100")
    assert not test_case_11("0101010101010101")
    assert not test_case_11("")
    assert not test_case_11("1")
    assert not test_case_11("0")

    assert test_case_12("00111011110", run_grammar=True)
    assert test_case_12("00")
    assert test_case_12("11")
    assert test_case_12("001110011010")
    assert test_case_12("010101010101010100")
    assert not test_case_12("0101010101010101")
    assert not test_case_12("")
    assert not test_case_12("1")
    assert not test_case_12("0")

    assert test_case_13("c", run_grammar=True)
    assert test_case_13("d")
    assert test_case_13("ccd")
    assert test_case_13("aaccd")
    assert not test_case_13("aaaccd")
    assert not test_case_13("aaaccdd")
    assert not test_case_13("aaaccdc")
    assert not test_case_13("")

    test_case_14()
    test_case_15()
    test_case_16()

    test_constraints()

    test_pda_case_1()


if __name__ == '__main__':
    print("[+] FD ")

    # run_cases()
    # print(test_case_13("c", True))
