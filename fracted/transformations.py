from fracted.types import Point, TransformationLike


class Transformation:

    func: TransformationLike

    def __init__(self, func: TransformationLike) -> None:
        self.func = func

    def __call__(self, point: Point) -> Point:
        return self.func(point)
