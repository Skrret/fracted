from typing import Callable, List, Tuple

import numpy

Transformation = Callable[[Tuple[float, float]], Tuple[float, float]]


class IFS:

    transfs: List[Transformation]
    probs: List[float] | None
    rng: numpy.random.Generator = numpy.random.default_rng()
    point: Tuple[float, float]

    def __init__(
        self,
        transfs: List[Transformation],
        probs: List[float] | None = None,
        start_point: Tuple[float, float] = (0, 0),
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
        self.point = start_point

    def step(self) -> None:
        """Applies random transformation to the point."""
        i: int = self.rng.choice(len(self.transfs), p=self.probs)
        t: Transformation = self.transfs[i]
        self.point = t(self.point)
