import networkx
import matplotlib.pyplot as plt

from utils.automaton.builder import *
from utils.function import *
from networkx import DiGraph


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

    print(nfsm)


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

    pda.debug(grammar.enumerate(length=15).pop())

    AutomataPlotter.plot(pda)
    AutomataPlotter.tikz(pda, filename="to_text", output=".")

    print(pda)


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

    print(fda)


def turing_machine_example():
    transition = TuringDelta()

    transition.add("e0", "e1", {
        C(0): A("a", move=Stay()),
        C(1): A(Tape.BLANK, new="X", move=Right())
    })

    transition.add("e0", "e1", {
        C(0): A("b", move=Stay()),
        C(1): A(Tape.BLANK, new="X", move=Right())
    })

    transition.add("e0", "e3", {
        C(0): A("c", move=Right()),
        C(1): A(Tape.BLANK, move=Right())
    })

    # ---

    transition.add("e1", "e1", {
        C(0): A("a", move=Right()),
        C(1): A(Tape.BLANK, new="a", move=Right())
    })

    transition.add("e1", "e1", {
        C(0): A("b", move=Right()),
        C(1): A(Tape.BLANK, new="b", move=Right())
    })

    transition.add("e1", "e2", {
        C(0): A("c", move=Stay()),
        C(1): A(Tape.BLANK, move=Left())
    })

    # ---

    transition.add("e2", "e2", {
        C(0): A("c", move=Stay()),
        C(1): A("a", move=Left())
    })

    transition.add("e2", "e2", {
        C(0): A("c", move=Stay()),
        C(1): A("b", move=Left())
    })

    transition.add("e2", "e3", {
        C(0): A("c", move=Right()),
        C(1): A("X", move=Right())
    })

    # ---

    transition.add("e3", "e3", {
        C(0): A("b", move=Right()),
        C(1): A("b", move=Right())
    })

    transition.add("e3", "e3", {
        C(0): A("a", move=Right()),
        C(1): A("a", move=Right())
    })

    transition.add("e3", "e4", {
        C(0): A(Tape.BLANK, move=Stay()),
        C(1): A(Tape.BLANK, move=Stay())
    })

    # ---
    transition.add_tape()
    transition.add_tape()
    transition.add_tape()

    turing = TuringMachine(initial="e0", final={"e4"}, transition=transition)

    turing.debug("abbabbacabbabba")

    print(turing)


def turing_function_example():
    x, y, z, w = symbols("x y z w")

    function = FunctionMachine(expression=3 * x - 4 * z - 7 * w + 9 * y, domain=[x >= 0, y > x, z >= 0, w >= 0])

    function.info()

    assert function.run_machine({x: 3, y: 5, z: 6, w: 2})
    assert function.run_machine({x: 3, y: 5, z: 6, w: 0})

    function = FunctionMachine(expression=9 * x - 3 * y, domain=[x >= 0, y >= 0])

    function.info()

    assert function.run_machine({x: 7, y: 3})
    assert function.run_machine({x: 8, y: 0})
    assert function.run_machine({x: 8, y: 6})


def ll1_grammar():
    grammar = OpenGrammar()

    grammar.add("S", "L")
    grammar.add("S", "aB")

    grammar.add("B", "$")
    grammar.add("B", "aL")
    grammar.add("B", "ea")

    grammar.add("L", "$")
    grammar.add("L", "d")
    grammar.add("L", "aL")

    grammar.add("P", "d")
    grammar.add("P", "ed")

    print(grammar)

    print("-----")

    lang1 = set(sorted(grammar.enumerate(length=15), key=lambda w: len(w)))

    grammar = OpenGrammar()
    grammar.add("S", "$")

    grammar.add("S", "Z")
    grammar.add("S", "Y")

    grammar.add("Z", "M")

    grammar.add("M", "aN")

    grammar.add("N", "aN")
    grammar.add("N", "$")

    grammar.add("N", "ea")
    grammar.add("N", "Y")

    grammar.add("Y", "d")

    print(grammar)
    lang2 = set(sorted(grammar.enumerate(length=15), key=lambda w: len(w)))

    print(lang2)

    print(lang1.difference(lang2))


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

    tester = PlanTester(plan=Accumulate(source_tape=0, target_tape=1),
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


class GrammarLeaf:
    def __init__(self, language, initial, grammar):
        if len(language.symbols) > 1:
            language.info()
            raise RuntimeError("invalid language for leaf")

        self.language = language
        self.initial = initial

        self.grammar = grammar

    @staticmethod
    def build(language, initial, grammar):
        if len(language.expression) == 1:
            expr = language.expression[0]
            if isinstance(expr, Symbol):
                return LoneLeaf(language=language, initial=initial, grammar=grammar)

            else:
                return SingleLeaf(language=language, initial=initial, grammar=grammar)

        elif len(language.expression) == 2:
            return TupleLeaf(language=language, initial=initial, grammar=grammar)

        elif len(language.expression) == 3:
            return TripletLeaf(language=language, initial=initial, grammar=grammar)

    @staticmethod
    def build_genesis(language, initial, grammar):
        if len(language.expression) == 1:
            expr = language.expression[0]
            if isinstance(expr, Symbol):
                return LoneGenesisLeaf(language=language, initial=initial, grammar=grammar)

            else:
                return SingleLeaf(language=language, initial=initial, grammar=grammar)

        elif len(language.expression) == 2:
            return TupleGenesisLeaf(language=language, initial=initial, grammar=grammar)

        elif len(language.expression) == 3:
            return TripletGenesisLeaf(language=language, initial=initial, grammar=grammar)


class LoneLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.language.expression) == 1

        expr = self.language.expression[0]
        assert isinstance(expr, Symbol)
        self.non_terminals = {expr: self.initial}


class LoneGenesisLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.language.expression) == 1

        expr = self.language.expression[0]
        assert isinstance(expr, Symbol)
        self.non_terminals = {expr: self.initial}

        # trigger domino train wave blob
        self.grammar.add(self.initial, str(expr))


class SingleLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 1

        expr = self.language.expression[0]

        assert isinstance(expr, Pow)

        non_terminal = self.grammar.get_non_terminal()
        self.non_terminals = {expr: non_terminal}

        self.grammar.add(self.initial, self.initial + non_terminal)
        self.grammar.add(self.initial, non_terminal)


class SingleGenesisLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 1

        expr = self.language.expression[0]

        assert isinstance(expr, Pow)

        non_terminal = self.grammar.get_non_terminal()
        self.non_terminals = {expr: non_terminal}

        self.grammar.add(self.initial, self.initial + non_terminal)
        self.grammar.add(self.initial, non_terminal)


class TupleLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 2

        expr_left = self.language.expression[0]
        expr_right = self.language.expression[1]

        assert isinstance(expr_left, Pow) and isinstance(expr_right, Pow)

        non_terminal_left = self.grammar.get_non_terminal()
        non_terminal_right = self.grammar.get_non_terminal()

        self.non_terminals = {expr_left: non_terminal_left, expr_right: non_terminal_right}

        self.grammar.add(self.initial, non_terminal_left + self.initial + non_terminal_right)
        self.grammar.add(self.initial, non_terminal_left + non_terminal_right)


class TupleGenesisLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 2

        expr_left = self.language.expression[0]
        expr_right = self.language.expression[1]

        assert isinstance(expr_left, Pow) and isinstance(expr_right, Pow)

        non_terminal_right = self.grammar.get_non_terminal()

        self.non_terminals = {expr_left: self.initial, expr_right: non_terminal_right}

        self.grammar.add(self.initial, str(expr_left.base) + self.initial + non_terminal_right)
        self.grammar.add(self.initial, str(expr_left.base) + non_terminal_right)

        # buffer.run_rule("S", "Z")
        # buffer.run_rule("Z", "aZB", times=4)
        # buffer.run_rule("Z", "aB")
        #
        # buffer.run_rule_until("aB", "ab")
        # buffer.run_rule_until("bB", "bb")
        #
        # assert cfl.check_grammar(grammar, length=10)


class TripletLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 3

        expr_a, ta = self.language.expression[0], self.grammar.get_non_terminal()
        expr_b, tb = self.language.expression[1], self.grammar.get_non_terminal()
        expr_c, tc = self.language.expression[2], self.grammar.get_non_terminal()

        self.non_terminals = {expr_a: ta, expr_b: tb, expr_c: tc}

        self.grammar.add(self.initial, self.initial + ta + tb + tc)
        self.grammar.add(self.initial, ta + tb + tc)

        self.grammar.add(tc + ta, ta + tc)
        self.grammar.add(tc + tb, tb + tc)
        self.grammar.add(tb + ta, ta + tb)


class TripletGenesisLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) == 3

        expr_a = self.language.expression[0].base
        expr_b, tb = self.language.expression[1], self.grammar.get_non_terminal()
        expr_c, tc = self.language.expression[2], self.grammar.get_non_terminal()

        self.non_terminals = {self.language.expression[0]: self.initial, expr_b: tb, expr_c: tc}

        self.grammar.add(self.initial, str(expr_a) + self.initial + tb + tc)
        self.grammar.add(self.initial, str(expr_a) + tb + tc)

        self.grammar.add(tc + tb, tb + tc)

        # buffer.run_rule("S", "Z")
        # buffer.run_rule("Z", "aZBC", times=4)
        # buffer.run_rule("Z", "aBC")
        #
        # buffer.run_rules_until([("CB", "BC")])
        #
        # buffer.run_rule_until("aB", "ab")
        # buffer.run_rule_until("bB", "bb")
        #
        # buffer.run_rule_until("bC", "bc")
        #
        # buffer.run_rule_until("cC", "cc")


class MultipleLeaf(GrammarLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert len(self.language.expression) > 2

        self.non_terminals = {expr: self.grammar.get_non_terminal() for expr in self.language.expression}


class GrammarTree:
    ONE = "one"

    @staticmethod
    def generate_rule_swapper(buffer, non_terminal_blocks, grammar):
        non_terminal_arrangement = reduce(lambda l, xx: l.append(xx) or l if xx not in l else l, buffer, [])

        for i in range(0, len(non_terminal_blocks)):
            symbol = non_terminal_blocks[i]
            for j in filter(lambda m: non_terminal_arrangement[m] == symbol,
                            range(0, len(non_terminal_arrangement))):

                if j > i:
                    for k in range(i, j):
                        neighbor = non_terminal_arrangement[k]

                        grammar.add(neighbor + symbol, symbol + neighbor)

                    after_reordering = list()
                    for k in range(0, i):
                        after_reordering.append(non_terminal_arrangement[k])

                    after_reordering.append(symbol)

                    for k in filter(lambda m: non_terminal_arrangement[m] != symbol,
                                    range(i, len(non_terminal_arrangement))):
                        after_reordering.append(non_terminal_arrangement[k])

                    non_terminal_arrangement = after_reordering

        return non_terminal_arrangement

    def __init__(self, language, initial="S"):
        self.language = LanguageFormula.normalize(language)
        self.tree = networkx.DiGraph()

        self.initial = initial
        self.non_terminal_counter = ord(initial)

        self.grammar = OpenGrammar()

        self.tree.add_node(self.initial, language=self.language)

        self._build_tree(self.initial)

        leafs = networkx.get_node_attributes(self.tree, name="leaf")
        self.non_terminals, self.expression_non_terminals = {}, {}

        for node in leafs:
            for expr in leafs[node].non_terminals:
                self.non_terminals[expr] = leafs[node].non_terminals[expr]
                self.expression_non_terminals[leafs[node].non_terminals[expr]] = expr

        self.non_terminal_blocks = [self.non_terminals[e] for e in self.language.expression if e in self.non_terminals]

        self._swap_symbols()

        self._add_terminal_rules()

        genesis_block = self.non_terminal_blocks[0]
        expr = self.language.expression[0]

        if isinstance(expr, Symbol):
            terminal = str(expr)
            self.grammar.rules[self.initial] = [terminal + each for each in self.grammar.rules[self.initial]]
            self.grammar.add(terminal + genesis_block, terminal + str(self.language.expression[1].base))
        else:
            terminal = str(expr.base)

            grammar = OpenGrammar()
            for each in self.grammar.rules:
                if genesis_block not in each:
                    for right in self.grammar.rules[each]:
                        if genesis_block in right:
                            if right[0] == genesis_block:
                                grammar.add(each, right.replace(genesis_block, terminal))
                            else:
                                grammar.add(each, right.replace(each + genesis_block, terminal + each))
                        else:
                            grammar.add(each, right)

            self.grammar = grammar

        minimal = self._get_minimum_indices()
        for expr in self.non_terminals:
            if any([m[expr.exp] == 0 for m in minimal]):
                self.grammar.add(self.non_terminals[expr], Transition.NULL)

    def _get_minimum_indices(self, constraints=None):
        space = ExponentSpace(sym=self.language.symbols, conditions=self.language.conditions,
                              length=self.language.total_length)

        return space.get_minimal(constraints)

    def _add_terminal_rules(self):
        non_terminal_arrangement = self.non_terminal_blocks

        block = 0
        for i in range(0, len(non_terminal_arrangement) - 1):
            symbol, next_symbol = non_terminal_arrangement[i], non_terminal_arrangement[i + 1]

            expr, next_expr = self.expression_non_terminals[symbol], self.expression_non_terminals[next_symbol]

            while self.language.expression[block] != expr and block < len(self.language.expression):
                block += 1

            next_block_expr = self.language.expression[block + 1]

            self.grammar.add(str(expr.base) + symbol, 2*str(expr.base))

            if isinstance(next_block_expr, Symbol):
                self.grammar.add(str(expr.base) + next_symbol,
                                 str(expr.base) + str(next_block_expr) + str(next_expr.base))
            else:
                self.grammar.add(str(expr.base) + next_symbol, str(expr.base) + str(next_expr.base))

        # omega
        last_symbol = non_terminal_arrangement[-1]
        last_expr = self.language.expression[-1]

        if isinstance(last_expr, Pow):
            self.grammar.add(str(last_expr.base) + last_symbol, 2 * str(last_expr.base))

    def _swap_symbols(self):
        for each in self.generate_non_terminal():
            self.generate_rule_swapper(each, self.non_terminal_blocks, self.grammar)

    def _is_root(self, node):
        return self.initial == node

    def _build_tree(self, node):
        language = networkx.get_node_attributes(self.tree, "language")

        blocks = []
        stack, block = set(), 0
        for each in language[node].expression:

            if isinstance(each, Pow):
                if len(language[node].expression_partition) > 1:
                    part = language[node].symbols_partition[each.exp]
                    expr = [e for e in language[node].expression if isinstance(e, Pow) and e.exp in part]
                    cond = [c for c in language[node].conditions if c.free_symbols.issubset(part)]

                else:
                    part = each.exp
                    expr = [e for e in language[node].expression if isinstance(e, Pow) and e.exp == part]
                    cond = [each.exp >= 0]

                if part not in stack:
                    stack.add(part)

                    non_terminal = self.grammar.get_non_terminal()

                    blocks.append(non_terminal)

                    lang = LanguageFormula(expression=expr, conditions=cond)

                    if len(lang.symbols) == 1:
                        self.tree.add_node(non_terminal, language=lang,
                                           leaf=GrammarLeaf.build(lang, non_terminal, self.grammar))

                    else:
                        self.tree.add_node(non_terminal, language=lang)
                        self._build_tree(non_terminal)

                    self.tree.add_edge(node, non_terminal, block=block)

                    block += 1

        if node == self.initial:
            self.grammar.add(node, ''.join(blocks))
        else:
            self.grammar.add(node, node + ''.join(blocks))
            self.grammar.add(node, ''.join(blocks))

    def _print_tree(self, node, space=""):
        blocks = networkx.get_edge_attributes(self.tree, "block")
        edges = sorted({e: blocks[e] for e in blocks if e[0] == node}, key=lambda e: blocks[e])

        attr = networkx.get_node_attributes(self.tree, name="language")

        print(space, node, "->", attr[node].expression)

        space += "   "
        if len(edges):
            for each in edges:
                self._print_tree(each[1], space)
        else:
            attr = networkx.get_node_attributes(self.tree, name="leaf")
            non_terminals = attr[node].non_terminals

            for each in non_terminals:
                print(space, non_terminals[each], "->", each)

    def _run_non_terminal_rules(self, node, buffer):
        blocks = networkx.get_edge_attributes(self.tree, "block")

        edges = sorted({e: blocks[e] for e in blocks if e[0] == node}, key=lambda e: blocks[e])

        if node in self.grammar.rules:
            recursive_rules = [r for r in self.grammar.rules[node] if node in r]
            terminal_rules = [r for r in self.grammar.rules[node] if node not in r]

            for rule in recursive_rules:
                if set([c for c in rule]).issubset(self.grammar.non_terminal):
                    buffer.run_rule(node, rule, times=3)

            for rule in terminal_rules:
                if set([c for c in rule]).issubset(self.grammar.non_terminal):
                    buffer.run_rule_until(node, rule)

            if len(edges):
                for each in edges:
                    self._run_non_terminal_rules(each[1], buffer)

    def info(self):
        print("\n[+] language")
        self.language.info()

        self._print_tree(self.initial)

        print("\n[+] grammar")
        print(self.grammar)

        print("\n[+] non terminal blocks", self.non_terminal_blocks)

    def generate_non_terminal(self):
        buffers = list()

        for initial_rule in self.grammar.rules[self.initial]:
            buffer = self.grammar.get_string()
            buffer.run_rule(self.initial, initial_rule)

            self._run_non_terminal_rules(self.initial, buffer)

            run_rules = True
            while run_rules:
                run_rules = False

                for each in self.grammar.rules:
                    if each in buffer.current and len(each) == 2 and \
                            set([c for c in each]).issubset(self.grammar.non_terminal):

                        for right in self.grammar.rules[each]:
                            if set([c for c in right]).issubset(self.grammar.non_terminal):
                                buffer.run_rule_until(each, right)
                                run_rules = True

            buffers.append(buffer.current)

        return buffers

    def generate_with_terminals(self):
        buffers = list()

        for each in self.generate_non_terminal():
            run_rules = True

            buffer = GrammarString(grammar=self.grammar, current=each)

            while run_rules:
                run_rules = False
                for left in self.grammar.rules:
                    if left in each:
                        for right in self.grammar.rules[left]:
                            each = buffer.run_rule(left, right)
                        run_rules = True

            buffers.append(each)

        return buffers

    def plot(self):
        attr = networkx.get_node_attributes(self.tree, name="language")
        expr = {s: s + " " + str(attr[s].expression) for s in attr}

        networkx.draw_networkx(self.tree, labels=expr)
        plt.show()


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
                          conditions=[n >= 0, k > 0, d < n, d >= 0])

    cfl.info()

    grammar = OpenGrammar()

    grammar.add("S", "RN")
    grammar.add("S", "R")
    grammar.add("S", "RT")

    grammar.add("R", "aRPO")
    grammar.add("R", "aPO")

    grammar.add("OP", "PO")

    grammar.add("N", "MNL")
    grammar.add("NL", "TL")

    grammar.add("T", "TM")
    grammar.add("T", "M")

    grammar.add("PM", "MP")
    grammar.add("OM", "MO")

    grammar.add("aM", "ad")
    grammar.add("dM", "dd")
    grammar.add("dP", "dbb")
    grammar.add("bP", "bb")
    grammar.add("bO", "bc")
    grammar.add("cO", "cc")
    grammar.add("cL", "cf")
    grammar.add("fL", "ff")

    print(grammar)
    print(grammar.enumerate(length=10))
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

    planner.add_plan(ParseEqualLeft(block=4, tape=1))
    planner.add_plan(ParseEqualRight(block=5, tape=1))

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


def main():
    print("[+] FD ")

    # grammar_case_1()

    # test_grammar_tree_1()
    # test_grammar_tree_2()
    # test_grammar_tree_3()
    # test_grammar_tree_4()
    # test_grammar_tree_5()

    # a, b, c, d, w, x, y, aa, f = symbols("a b c d w x y aa f")
    # m, n, i, j, k = symbols("m n i j k")
    #
    # cfl = LanguageFormula(expression=[a ** k, d ** n, b, b ** k, c ** k, f ** n],
    #                       conditions=[n > 0, k > 0])
    #
    # tree = GrammarTree(language=cfl)
    #
    # print("[+] terminals", tree.generate_with_terminals())
    #
    # tree.info()
    #
    # print(tree.grammar.enumerate(length=15))
    # assert cfl.check_grammar(tree.grammar, length=15)

    # tree.plot()

    testing_turing_language()
    
    x, y, i1, i0 = symbols("x y 1 0")
    m, k, n = symbols("m k n")

    cfl = LanguageFormula(expression=[i1 ** n, i0, i1 ** m],
                          conditions=[n >= 0, m >= 0])

    planner = TuringPlanner(language=cfl, tapes=2)

    planner.add_plan(ParseAccumulate(block=0, tape=1))

    planner.add_plan(ParseLoneSymbol(block=1))

    planner.add_plan(ParseAccumulate(block=2, tape=2))

    machine = MachineBuilder(planner=planner)

    assert machine.turing.debug("111101111")

    # for data in cfl.enumerate_strings(length=22):
    #     print("[+] testing string", data)
    #     read_status = machine.turing.read(data)
    #
    #     if not read_status:
    #         machine.turing.debug(data)
    #         machine.info()
    #
    #     assert read_status
    #
    # machine.info()

    # TuringPlotter.to_csv("machine.csv", machine.turing)


main()
