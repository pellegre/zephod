# zephod

**Q**:	How many hardware engineers does it take to change a light bulb?

**A**:	None.  We'll fix it in software.

------------

**Q**:	How many system programmers does it take to change a light bulb?

**A**:	None.  The application can work around it.

------------

**Q**:	How many software engineers does it take to change a light bulb?

**A**:	None.  We'll document it in the manual.

------------

**Q**:	How many tech writers does it take to change a light bulb?

**A**:	None.  The user can figure it out.

------------

**Q**:	How many automatas does it take to change a light bulb?

**A**:	Neo. And now the CPU radiator is broken.

------------

*zephod* is a python framework to study and model abstract computing machines. It supports out of the box 
[Finite Automatas](./zephod/finite.py), [Pushdown Automatas](./zephod/pushdown.py) and 
[Turing Machines](./zephod/turing.py) and can be easily extended with other models of computations.

It also comes with some [utils](./utils/) and [automation](./utils/automaton) tools to work with 
[grammars](./utils/language/grammar.py) and [languages](./utils/language/formula.py).

To run it locally, you should configure your python path to point where you cloned the repo, install the dependencies
and test it with some [examples](./examples/). 
