import numpy
import pytest

from fracted import fractals
from fracted.transformations import Transformation

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
    """Test if IFS.step() applies transformation with non-zero probability.

    Test both ways to assign probabilities - list and attribute. Test if they
    behaves same when same seeds given..
    """
    frac1 = fractals.IFS(transfs, probs, start_point=start_point, seed=123)
    probs2 = probs if probs else [1] * len(transfs)
    frac2 = fractals.IFS(
        [Transformation(t, p) for t, p in zip(transfs, probs2)],
        start_point=start_point,
        seed=123,
    )
    last_point = start_point
    for i in range(20):
        frac1.step()
        frac2.step()
        possible = [t(last_point) for t, p in zip(transfs, probs2) if p]
        assert frac1.point in possible
        assert frac2.point in possible
        assert frac1.point == frac2.point
        last_point = frac1.point


def test_bad_probs_error():
    """Test if IFS.__init__() raises an error when 'transfs' and 'probs' have different length"""
    with pytest.raises(ValueError):
        fractals.IFS(transfs, [0, 3, 2])


@pytest.mark.parametrize(
    ("min_x", "min_y", "max_x", "max_y", "resolution", "n_iter", "transfs"),
    [
        (
            -50,
            -50,
            50,
            50,
            1.1,
            50,
            [
                (lambda point, sx=sx, sy=sy: (point[0] / 3 + sx, point[1] / 3 + sy))
                for sx, sy in [(0, 0), (3, 7), (50, -50), (-50, 50)]
            ],
        ),
        (
            -1,
            -5.2,
            1,
            -2.5,
            30,
            20,
            [
                lambda point: (point[0] / 3 + 2, point[1] / 2),
                lambda point: (point[1] / 3 + 0.7, point[0] ** 2),
            ],
        ),
    ],
)
def test_fractal(min_x, min_y, max_x, max_y, resolution, n_iter, transfs):
    """Tests if the fractal is drawed wit no errors and with all points."""
    frac = fractals.IFS(
        transfs,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        resolution=resolution,
    )
    points_off = 0
    for _ in range(n_iter):
        frac.step(draw=True)
        if not ((min_x < frac.point[0] < max_x) and (min_y < frac.point[1] < max_y)):
            points_off += 1
    assert frac.array.sum() + points_off == n_iter
