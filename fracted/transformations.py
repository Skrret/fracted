from fracted.types import Point, Transformation


class Transf:

    func: Transformation

    def __init__(self, func: Transformation) -> None:
        self.func = func

    def __call__(self, point: Point) -> Point:
        return self.func(point)
