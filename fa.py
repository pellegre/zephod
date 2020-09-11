from pyauto.fsm import *
import random
import string


def get_tmp_filename():
    return "/tmp/" + "".join(random.choice(string.ascii_letters) for _ in range(12)) + ".pdf"


if __name__ == '__main__':
    print("[+] FD ")

    transition = Transition()
    transition.add("e0", "e1", {"0"})
    transition.add("e0", "e2", {"1"})
    transition.add("e1", "e2", {"1"})
    transition.add("e2", "e1", {"0"})
    transition.add("e1", "e3", {"0"})
    transition.add("e2", "e3", {"1"})
    transition.add("e3", "e3", {"0", "1"})

    fsm = FiniteAutomata(transition, "e0", {"e3"})
    print(fsm)

    dot = fsm.build_dot()
    dot.view(filename=get_tmp_filename())
