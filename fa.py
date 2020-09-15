from pyauto.fsm import *
from utils.builder import *

import random
import string

import pandas


def get_tmp_filename():
    return "/tmp/" + "".join(random.choice(string.ascii_letters) for _ in range(12))


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
    nfsm_value = nfsm.read(inp)

    rebased_nfsm = nfsm.rebase(17)
    rebased_nfsm_value = rebased_nfsm.read(inp)

    if plot:
        dot = rebased_nfsm.build_dot()
        dot.view(filename=get_tmp_filename())

    return dfsm_value and nfsm_value and rebased_nfsm_value


def test_case_2(inp, plot=False):
    transition = Transition()
    transition.add("s0", "s0", {"a"})
    transition.add("s0", "s2", {"$"})
    transition.add("s1", "s2", {"a", "b"})
    transition.add("s2", "s0", {"a"})
    transition.add("s2", "s1", {"$"})

    nfsm = FiniteAutomata(transition, "s0", {"s1"})
    nfsm_value = nfsm.read(inp)

    if plot:
        dot = nfsm.build_dot()
        dot.view(filename=get_tmp_filename())

    return nfsm_value


def test_case_3(inp, plot=False):
    expr = (Z("1") + Z("10")) | ~Z("01")

    expr_value = expr.read(inp)

    if plot:
        dot = expr.build_dot()
        dot.view(filename=get_tmp_filename())

    return expr_value


def test_case_4(inp, plot=False):
    expr = Z("00") + (~Z("0") | Z("1"))

    expr_value = expr.read(inp)

    if plot:
        dot = expr.build_dot()
        dot.view(filename=get_tmp_filename())

    return expr_value


def test_case_5(inp, plot=False):
    expr = FiniteAutomataBuilder.get_finite_automata_from_csv("./csv/zuto.csv")

    expr_value = expr.read(inp)
    print(expr)
    print("expr =", expr_value)

    if plot:
        dot = expr.build_dot()
        dot.view(filename=get_tmp_filename())

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


if __name__ == '__main__':
    print("[+] FD ")
    # run_cases()
    test_case_5("badddcbdb", True)