import pytest

from fracted.transformations import Transformation


@pytest.mark.parametrize(
    ("f1", "f2", "input"),
    [
        (
            lambda point: (point[1] / 2, point[0] * 3),
            lambda point: (point[0] + 3, point[1] - 2.7),
            (-5, 3),
        ),
        (lambda point: (1, 2), lambda point: (point[1], 8.6), (2, 0)),
    ],
)
def test_transf_composion(f1, f2, input):
    output = f2(f1(input))
    t1 = Transformation(f1)
    t2 = Transformation(f2)
    t3 = Transformation(f1)
    assert (t2 @ t1)(input) == output
    t3 @= t2
    assert t3(input) == output
    t2.append_before(t1)
    assert t2(input) == output
