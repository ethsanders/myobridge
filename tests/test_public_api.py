"""The names `import myobridge` promises, and the example in its docstring."""

import doctest

import myobridge


def test_everything_in_all_is_importable():
    for name in myobridge.__all__:
        assert getattr(myobridge, name) is not None, name


def test_both_ways_of_saying_not_sure_are_exported():
    # gesture sets with a coverage promise, and answers with an accuracy promise
    assert {"ConformalGestures", "SelectiveGestures"} <= set(myobridge.__all__)
    assert myobridge.SelectiveGestures.__module__ == "myobridge.selective"


def test_version_is_a_release_number():
    major, minor, patch = myobridge.__version__.split(".")
    assert all(part.isdigit() for part in (major, minor, patch))


def test_package_docstring_example_runs():
    example = myobridge.__doc__.split("::", 1)[1].split("\n\nSee ")[0]
    code = "\n".join(line[4:] for line in example.splitlines() if line.strip())
    namespace: dict[str, object] = {"print": lambda *a: None}
    exec(compile(code, "myobridge.__doc__", "exec"), namespace)
    assert namespace["pipeline"].samples_seen == 1000


def test_no_stray_doctests_fail():
    # docstrings that look like interactive sessions must actually run
    assert doctest.testmod(myobridge, verbose=False).failed == 0
