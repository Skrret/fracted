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
    ("probs", "start_point"),
    [
        ([0, 0, 3, 9.7], (0, 0)),
        ([0.01, 0.005, 0, 0], (-3, 8.4)),
        ([0, 120, 0, 0], (0.0001, 0)),
        (None, (-390.987, 23)),
    ],
)
def test_transfs(probs, start_point):
    """Test if IFS.step() applies transformation with non-zero probability"""
    frac = fractals.IFS(transfs, probs, start_point)
    last_point = start_point
    for i in range(20):
        frac.step()
        test = False
        for i in range(len(transfs)):
            if transfs[i](last_point) == frac.point and (probs is None or probs[i]):
                test = True
                break
        assert test
        last_point = frac.point


def test_bad_probs_error():
    """Test if IFS.__init__() raises an error when 'transfs' and 'probs' have different length"""
    with pytest.raises(ValueError):
        fractals.IFS(transfs, [0, 3, 2])
