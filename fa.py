from pyauto.fsm import *
from utils.builder import *


def test_case_1(inp, plot=False):
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

    rebased_nfsm = nfsm.rebase(17)
    rebased_nfsm_value = rebased_nfsm.read(inp)

    if plot:
        AutomataPlotter.plot(dfsm)

    return dfsm_value and nfsm_value and rebased_nfsm_value


def test_case_2(inp, plot=False):
    transition = Transition()
    transition.add("s0", "s0", {"a"})
    transition.add("s0", "s2", {"$"})
    transition.add("s1", "s2", {"a", "b"})
    transition.add("s2", "s0", {"a"})
    transition.add("s2", "s1", {"$"})

    nfsm = FiniteAutomata(transition, "s0", {"s1"})
    assert nfsm.has_null_transitions()

    nfsm_value = nfsm.read(inp)

    if plot:
        AutomataPlotter.plot(nfsm)

    return nfsm_value


def test_case_3(inp, plot=False):
    expr = (Z("1") + Z("10")) | ~Z("01")
    assert expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_4(inp, plot=False):
    expr = Z("00") + (~Z("0") | Z("1"))
    assert expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_5(inp, plot=False):
    expr = FiniteAutomataBuilder.get_finite_automata_from_csv("./csv/zuto.csv")
    assert not expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_6(inp, plot=False):
    unit = Unit(alphabet={"a", "b", "c", "d"},
                constraints=[Odd(pattern="b"), Even(pattern="d"), NotContained("addc")],
                name="w")

    expr = FiniteAutomataBuilder.get_finite_automata_from_unit(unit)
    assert not expr.has_null_transitions()

    expr_value = expr.read(inp)

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_7(inp, plot=False):
    unit = Unit(alphabet={"a", "b", "c", "d"},
                constraints=[Odd(pattern="b"), NotContained("adc")],
                name="w")

    # print(unit.get_frame(total=True))
    expr = FiniteAutomataBuilder.get_finite_automata_from_unit(unit)
    assert not expr.has_null_transitions()

    expr_value = expr.read(inp)
    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_8(inp, plot=False):
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

    if plot:
        AutomataPlotter.plot(expr)

    return expr_value


def test_case_9(inp, plot=False):
    expr = Z("00") + (~Z("0") | Z("1"))
    assert expr.has_null_transitions()

    stripped = expr.remove_null_transitions()
    expr_value = stripped.read(inp)

    if plot:
        AutomataPlotter.plot(stripped)

    return expr_value


def test_case_10(inp, plot=False):
    unit = Unit(alphabet={"0", "1"},
                  constraints=[NotContained(pattern="11"), NotEnd("00")],
                  name="x")
    expr = FiniteAutomataBuilder.get_finite_automata_from_unit(unit)

    expr = ~Z("aa") | expr | ~Z("c")
    assert expr.has_null_transitions()

    stripped = expr.remove_null_transitions()
    expr_value = stripped.read(inp)

    if plot:
        AutomataPlotter.plot(expr)
        AutomataPlotter.plot(stripped)

    return expr_value


def test_case_11(inp, plot=False):
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

    if plot:
        AutomataPlotter.plot(dfsm)

    return expr_value


def run_cases():
    assert test_case_1("00111011110")
    assert test_case_1("00")
    assert test_case_1("11")
    assert test_case_1("001110011010")
    assert test_case_1("010101010101010100")
    assert not test_case_1("0101010101010101")
    assert not test_case_1("")
    assert not test_case_1("1")
    assert not test_case_1("0")

    assert test_case_2("")
    assert test_case_2("ababa")
    assert test_case_2("aa")
    assert test_case_2("bb")
    assert test_case_2("b")
    assert test_case_2("a")

    assert test_case_3("10010101")
    assert not test_case_3("1110010101")
    assert test_case_3("10010101")
    assert test_case_3("1")
    assert test_case_3("10010101")
    assert not test_case_3("010101")

    assert test_case_4("1")
    assert test_case_4("00")
    assert test_case_4("00001")
    assert test_case_4("0000001")
    assert not test_case_4("1111")

    assert test_case_5("badddcbdb")
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

    assert test_case_6("badddcbdb")
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

    assert test_case_7("abbbad")
    assert test_case_7("b")
    assert test_case_7("abbb")
    assert test_case_7("abbbbbcdcd")
    assert test_case_7("baddd")
    assert not test_case_7("badc")
    assert not test_case_7("")
    assert not test_case_7("bbbadc")
    assert not test_case_7("bbbbbadc")

    assert test_case_9("1")
    assert test_case_9("00")
    assert test_case_9("00001")
    assert test_case_9("0000001")
    assert not test_case_9("1111")

    assert test_case_10("aa01000101010101c")
    assert test_case_10("aa010001010101010c")
    assert test_case_10("aa010001010101010cccccc")
    assert test_case_10("aaaa0100010cccc")
    assert not test_case_10("aaaa01000100cccc")
    assert not test_case_10("aaaaa010001cccc")
    assert not test_case_10("aaaa0110001cccc")

    assert test_case_11("00111011110")
    assert test_case_11("00")
    assert test_case_11("11")
    assert test_case_11("001110011010")
    assert test_case_11("010101010101010100")
    assert not test_case_11("0101010101010101")
    assert not test_case_11("")
    assert not test_case_11("1")
    assert not test_case_11("0")


if __name__ == '__main__':
    print("[+] FD ")
    run_cases()
    # print(test_case_11("01", True))
