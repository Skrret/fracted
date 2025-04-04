from typing import List

import numpy
from numpy.typing import NDArray

from fracted.types import Point, Transformation


class IFS:

    transfs: List[Transformation]
    probs: List[float] | None
    rng: numpy.random.Generator = numpy.random.default_rng()
    point: Point
    resolution: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    array: NDArray[numpy.uint32]

    def __init__(
        self,
        transfs: List[Transformation],
        probs: List[float] | None = None,
        resolution: float = 1,
        min_x: float = -100,
        max_x: float = 100,
        min_y: float = -100,
        max_y: float = 100,
        start_point: Point = (0, 0),
    ) -> None:
        """Parameters:

        transfs - list of transformations,

        probs - list of transformation probabilities.
        If not given, the sample assumes a uniform distribution over all transformations.
        Must have same size as transfs.
        sum(probs) don't have to be 1

        resolution - pixels of the final image per unit,

        start_point - initial position of the drawing point
        """
        if not (probs is None):
            if len(probs) != len(transfs):
                raise ValueError("Tranfs and probs must have the same length.")
            s = sum(probs)
            if s != 1:
                for i in range(len(probs)):
                    probs[i] /= s
        self.transfs = transfs
        self.probs = probs
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.array = numpy.zeros(
            (int((max_x - min_x) * resolution), int((max_y - min_y) * resolution)),
            dtype=numpy.uint32,
        )
        self.point = start_point
        self.resolution = resolution

    def step(self, draw: bool = False) -> None:
        """Applies random transformation to the point."""
        i: int = self.rng.choice(len(self.transfs), p=self.probs)
        t: Transformation = self.transfs[i]
        self.point = t(self.point)
        if draw:
            self.draw_point(self.point)

    def draw_point(self, point: Point | None = None) -> None:
        if point is None:
            point = self.point
        x, y = point
        if self.min_x < x < self.max_x and self.min_y < y < self.max_y:
            self.array[
                int(self.resolution * (x + self.min_x)),
                int(self.resolution * (y + self.min_y)),
            ] += 1

    def draw(self, start_iter: int, n_iter: int) -> NDArray[numpy.uint32]:
        for _ in range(start_iter):
            self.step(draw=False)
        for _ in range(n_iter):
            self.step(draw=True)
        return self.array
