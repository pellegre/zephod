from pyauto.finite_automata import *
from utils.builder import *
from pyauto.grammar import *
from pyauto.language import *
from pyauto.pushdown_automata import *
from pyauto.tape import *


def nfsm_example():
    transition = FADelta()
    transition.add("e0", "e0", {"0", "1"})
    transition.add("e0", "e1", {"1"})
    transition.add("e0", "e3", {"0"})

    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e2", {"0", "1"})

    transition.add("e3", "e4", {"0"})
    transition.add("e4", "e4", {"0", "1"})

    nfsm = FiniteAutomata(transition, "e0", {"e2", "e4"})

    g = Grammar.build_from_finite_automata(nfsm.minimal())
    data = g(length=16)
    nfsm.debug(data)

    AutomataPlotter.plot(nfsm)
    AutomataPlotter.plot(nfsm.minimal())


def pda_example():
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

    pda.debug(grammar(length=15))

    AutomataPlotter.plot(pda)


def regex_example():
    expr = (~Z("aaa") | ~Z("bb") | (~Z("cd") + Z("ab"))).minimal()

    transition = FADelta()
    transition.add("z0", "z1", {~Z("aaa")})
    transition.add("z1", "z2", {~Z("bb")})
    transition.add("z2", "y1", {Transition.NULL})
    transition.add("y1", "z3", {(~Z("cd") + Z("ab"))})

    fda = FiniteAutomata(transition, initial="z0", final={"z0", "z1", "z2", "z3"})

    g = Grammar.build_from_finite_automata(expr)
    data = g(length=16)
    fda.debug(data)

    AutomataPlotter.plot(fda)


def main():
    print("[+] FD ")

    regex_example()

    # grammar = OpenGrammar()
    #
    # grammar.add("S", "A")
    #
    # grammar.add("A", "aABC")
    # grammar.add("A", "aBC")
    #
    # grammar.add("CB", "BC")
    #
    # grammar.add("aB", "ab")
    # grammar.add("bB", "bb")
    # grammar.add("bC", "bc")
    # grammar.add("cC", "cc")
    #
    # print(grammar(length=10))
    #
    # exit()


main()
