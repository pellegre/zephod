from pyauto.fsm import *
import random
import string


def get_tmp_filename():
    return "/tmp/" + "".join(random.choice(string.ascii_letters) for _ in range(12)) + ".pdf"


def test_case_1(plot=False):
    inp = "101010101000"

    transition = Transition()
    transition.add("e0", "e1", {"0"})
    transition.add("e0", "e2", {"1"})
    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e1", {"0"})
    transition.add("e1", "e3", {"0"})
    transition.add("e2", "e3", {"1"})
    transition.add("e3", "e3", {"0", "1"})

    dfsm = FiniteAutomata(transition, "e0", {"e3"})
    print("dfsm = ", dfsm.read(inp))

    transition = Transition()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    nfsm = FiniteAutomata(transition, "e0", {"e2", "e4"})
    print("nfsm = ", nfsm.read(inp))

    if plot:
        dot = nfsm.build_dot()
        dot.view(filename=get_tmp_filename())


def test_case_2(plot=False):
    inp = "abbbb"

    transition = Transition()
    transition.add("s0", "s0", {"a"})
    transition.add("s0", "s2", {"$"})
    transition.add("s1", "s2", {"a", "b"})
    transition.add("s2", "s0", {"a"})
    transition.add("s2", "s1", {"$"})

    nfsm = FiniteAutomata(transition, "s0", {"s1"})
    print("nfsm = ", nfsm.read(inp))

    if plot:
        dot = nfsm.build_dot()
        dot.view(filename=get_tmp_filename())


if __name__ == '__main__':
    print("[+] FD ")
    test_case_1()
    test_case_2()
