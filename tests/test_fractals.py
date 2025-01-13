import numpy
import pytest

from fracted import fractals

transfs = [
    lambda t: t,
    lambda t: (t[0] + 2, t[1] * 10),
    lambda t: (
        t[1],
        t[0],
    ),
    lambda t: (t[0] * t[1], numpy.sin(t[0])),
]


@pytest.mark.parametrize(
    ("nonzero_probs", "start_point"),
    [({1}, (0, 0)), ({0, 3, 2}, (-3, 8.4)), ({1, 2}, (0.0001, 0))],
)
def test_transfs(nonzero_probs, start_point):
    length = len(transfs)
    prob = 1 / len(nonzero_probs)
    probs = []
    for i in range(length):
        probs.append(prob * (i in nonzero_probs))
    frac = fractals.IFS(transfs, probs, start_point)
    last_point = start_point
    for i in range(20):
        frac.step()
        test = False
        for i in range(length):
            if transfs[i](last_point) == frac.point and probs[i]:
                test = True
                break
        assert test
        last_point = frac.point
