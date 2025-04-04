from typing import Callable, List, Tuple

Transformation = Callable[[Tuple[float, float]], Tuple[float, float]]


class Transf:

    func: Transformation

    def __init__(self, func: Transformation) -> None:
        self.func = func

    def __call__(self, point: Tuple[float, float]) -> Tuple[float, float]:
        return self.func(point)
