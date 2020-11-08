from zephod.turing import *


def turing_machine_example():
    transition = TuringDelta(tapes=2)

    transition.add("e0", "e1", {
        T(0): A("a", move=Stay()),
        T(1): A(Tape.BLANK, new="X", move=Right())
    })

    transition.add("e0", "e1", {
        T(0): A("b", move=Stay()),
        T(1): A(Tape.BLANK, new="X", move=Right())
    })

    transition.add("e0", "e3", {
        T(0): A("c", move=Right()),
        T(1): A(Tape.BLANK, move=Right())
    })

    # ---

    transition.add("e1", "e1", {
        T(0): A("a", move=Right()),
        T(1): A(Tape.BLANK, new="a", move=Right())
    })

    transition.add("e1", "e1", {
        T(0): A("b", move=Right()),
        T(1): A(Tape.BLANK, new="b", move=Right())
    })

    transition.add("e1", "e2", {
        T(0): A("c", move=Stay()),
        T(1): A(Tape.BLANK, move=Left())
    })

    # ---

    transition.add("e2", "e2", {
        T(0): A("c", move=Stay()),
        T(1): A("a", move=Left())
    })

    transition.add("e2", "e2", {
        T(0): A("c", move=Stay()),
        T(1): A("b", move=Left())
    })

    transition.add("e2", "e3", {
        T(0): A("c", move=Right()),
        T(1): A("X", move=Right())
    })

    # ---

    transition.add("e3", "e3", {
        T(0): A("b", move=Right()),
        T(1): A("b", move=Right())
    })

    transition.add("e3", "e3", {
        T(0): A("a", move=Right()),
        T(1): A("a", move=Right())
    })

    transition.add("e3", "e4", {
        T(0): A(Tape.BLANK, move=Stay()),
        T(1): A(Tape.BLANK, move=Stay())
    })

    turing = TuringMachine(initial="e0", final={"e4"}, transition=transition)

    turing.debug("abbabbacabbabba")

    print(turing)


turing_machine_example()
