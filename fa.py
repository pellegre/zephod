from pyauto.fsm import *
from utils.builder import *
from pyauto.grammar import *

import copy


def test_case_1(inp, plot=False, run_grammar=False):
    transition = Transition()
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

    transition = Transition()
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

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(nfsm)

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(nfsm)
        AutomataPlotter.plot(minimized)

    return rebased_nfsm_value


def test_case_2(inp, plot=False, run_grammar=False):
    transition = Transition()
    transition.add("s0", "s0", {"a"})
    transition.add("s0", "s2", {"$"})
    transition.add("s1", "s2", {"a", "b"})
    transition.add("s2", "s0", {"a"})
    transition.add("s2", "s1", {"$"})

    nfsm = FiniteAutomata(transition, "s0", {"s1"})
    assert nfsm.has_null_transitions()

    nfsm_value = nfsm.read(inp)
    stripped = nfsm.remove_null_transitions()
    dfsm = stripped.get_deterministic_automata()

    assert not dfsm.is_non_deterministic()
    assert dfsm.read(inp) == nfsm_value

    minimized = dfsm.minimize_automata()
    assert minimized.read(inp) == nfsm_value

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


def test_case_6(inp, plot=False, run_grammar=False):
    unit = Unit(alphabet={"a", "b", "c", "d"},
                constraints=[Odd(pattern="b"), Even(pattern="d"), NotContained("addc")],
                name="w")

    expr = FiniteAutomataBuilder.get_finite_automata_from_unit(unit)
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


def test_case_7(inp, plot=False, run_grammar=False):
    unit = Unit(alphabet={"a", "b", "c", "d"},
                constraints=[Odd(pattern="b"), NotContained("adc")],
                name="w")

    # print(unit.get_frame(total=True))
    expr = FiniteAutomataBuilder.get_finite_automata_from_unit(unit)
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


def test_case_8(inp, plot=False, run_grammar=False):
    unit_1 = Unit(alphabet={"a", "b"},
                  constraints=[Even(pattern="b")],
                  name="x")
    expr_1 = FiniteAutomataBuilder.get_finite_automata_from_unit(unit_1)

    unit_2 = Unit(alphabet={"a", "c"},
                  constraints=[Odd(pattern="c"), NotContained("aac")],
                  name="w")
    expr_2 = FiniteAutomataBuilder.get_finite_automata_from_unit(unit_2)

    unit_3 = unit_1 | unit_2
    expr_3 = FiniteAutomataBuilder.get_finite_automata_from_unit(unit_3)

    print(unit_1.get_frame(total=True))
    print(unit_2.get_frame(total=True))
    print(unit_3.get_frame(total=True))

    print(expr_3.read("bbc"))

    expr = (expr_1 | expr_2).remove_null_transitions()
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

    if run_grammar:
        grammar_from_fda = Grammar.build_from_finite_automata(minimized)

        for i in range(500):
            for j in [1, 5, 10, 20, 30]:
                assert minimized.read(grammar_from_fda(length=j))

    if plot:
        AutomataPlotter.plot(dfsm)
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_10(inp, plot=False, run_grammar=False):
    unit = Unit(alphabet={"0", "1"},
                  constraints=[NotContained(pattern="11"), NotEnd("00")],
                  name="x")
    expr = FiniteAutomataBuilder.get_finite_automata_from_unit(unit)

    expr = ~Z("aa") | expr | ~Z("c")
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
        AutomataPlotter.plot(minimized)

    return expr_value


def test_case_11(inp, plot=False, run_grammar=False):
    transition = Transition()
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

    if plot:
        AutomataPlotter.plot(dfsm)

    return expr_value


def test_case_12(inp, plot=False, run_grammar=False):
    transition = Transition()
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

    assert test_case_6("badddcbdb", run_grammar=True)
    assert test_case_6("ddb")
    assert test_case_6("ddddb")
    assert test_case_6("b")
    assert test_case_6("bdd")
    assert test_case_6("bbddb")
    assert test_case_6("bdbdb")
    assert not test_case_6("")
    assert not test_case_6("dd")
    assert not test_case_6("dddb")
    assert not test_case_6("ddbaddc")
    assert not test_case_6("baddc")
    assert not test_case_6("baddcdd")
    assert test_case_6("bddcdd")

    assert test_case_7("abbbad", run_grammar=True)
    assert test_case_7("b")
    assert test_case_7("abbb")
    assert test_case_7("abbbbbcdcd")
    assert test_case_7("baddd")
    assert not test_case_7("badc")
    assert not test_case_7("")
    assert not test_case_7("bbbadc")
    assert not test_case_7("bbbbbadc")

    assert test_case_9("1", run_grammar=True)
    assert test_case_9("00")
    assert test_case_9("00001")
    assert test_case_9("0000001")
    assert not test_case_9("1111")

    assert test_case_10("aa01000101010101c", run_grammar=True)
    assert test_case_10("aa010001010101010c")
    assert test_case_10("aa010001010101010cccccc")
    assert test_case_10("aaaa0100010cccc")
    assert not test_case_10("aaaa01000100cccc")
    assert not test_case_10("aaaaa010001cccc")
    assert not test_case_10("aaaa0110001cccc")

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


if __name__ == '__main__':
    print("[+] FD ")

    run_cases()
    # print(test_case_13("c", True))
