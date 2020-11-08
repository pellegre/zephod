from utils.language.regular import *


def regular_language_definition_fda_example():
    lang = RegularLanguage(alphabet={"a", "b", "c"},
                           definition={
                               "rules": [
                                   Even(pattern="baa"), Odd(pattern="c"),
                                   NotContained(pattern="aac"), Order(before={"a", "b"}, after={"a", "c"})
                               ],
                               "closure": {"a", "b", "c"}
                           })

    assert lang.fda.read("c")
    assert lang.fda.read("baabaabc")
    assert not lang.fda.read("baabaac")
    assert lang.fda.read("baabaabccc")
    assert lang.fda.read("baaabaaabaabaaabcacac")
    assert not lang.fda.read("baaabaaabaabaaabcacacb")

    AutomataPlotter.plot(lang.fda.minimal())

    lang = RegularLanguage(alphabet={"a", "b", "c", "d"},
                           definition={
                               "rules": [
                                   Even(pattern="baa"),
                                   Order(before={"a", "b"}, after={"a", "c"}),
                                   Odd(pattern="c"), NotContained(pattern="aac"),
                                   Order(before={"a", "b", "c"}, after={"a", "d"}),
                                   Odd(pattern="dd"), NotContained(pattern="ad")
                               ],
                               "closure": {"a", "b", "c", "d"}
                           })

    assert lang.fda.read("cdd")
    assert lang.fda.read("baabaabbbbbcdd")
    assert lang.fda.read("cccdddddd")
    assert lang.fda.read("cacacdddddd")
    assert not lang.fda.read("cacacadddddd")
    assert not lang.fda.read("cacacaddddddb")
    assert not lang.fda.read("cacacaddddcdd")
    assert not lang.fda.read("bacabaabbbbbcdd")
    assert not lang.fda.read("badabaabbbbbcdd")

    AutomataPlotter.plot(lang.fda.minimal())


regular_language_definition_fda_example()