from zephod.finite import *

from utils.plotter import *
from utils.language.grammar import *


def regex_example():
    expr = (~Z("aaa") | ~Z("bb") | (~Z("cd") + Z("ab"))).minimal()

    g = Grammar.build_from_finite_automata(expr)
    strings = g.enumerate(length=8)

    for each in strings:
        assert expr.read(each)

    expr.debug(strings.pop())
    AutomataPlotter.plot(expr)

    print(expr)


regex_example()
