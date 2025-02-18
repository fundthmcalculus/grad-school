import abc

import numpy as np
from numpy.typing import NDArray


class MembershipFunction(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        pass

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.mu(x)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.domain()}"

    def __repr__(self) -> str:
        return self.__str__()

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("mu() must be implemented in subclass")

    def domain(self) -> NDArray[np.float64]:
        raise NotImplementedError("domain() must be implemented in subclass")

    def in_domain(self, x: NDArray[np.float64] | float) -> bool:
        if isinstance(x, float):
            x = np.array([x])
        return np.all(np.logical_and(self.domain()[0] <= x, x <= self.domain()[1]))

    def centroid(self) -> float:
        raise NotImplementedError("centroid() must be implemented in subclass")


class TriangularMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        assert a <= b <= c
        self.a = a
        self.b = b
        self.c = c

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(
            np.minimum(
                (x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)
            ),
            0,
        )

    def domain(self) -> NDArray[np.float64]:
        return np.array([self.a, self.c])


class TrapezoidalMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        super().__init__(name)
        assert a <= b <= c <= d
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(
            np.minimum(
                (x - self.a) / (self.b - self.a), 1, (self.d - x) / (self.d - self.c)
            ),
            0,
        )

    def domain(self) -> NDArray[np.float64]:
        return np.array([self.a, self.d])


class LinearZMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a >= b
        self.a = a
        self.b = b

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(np.minimum((self.b - x) / (self.b - self.a), 1), 0)

    def domain(self) -> NDArray[np.float64]:
        return np.array([-np.inf, self.b])


class LinearSMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(np.minimum((x - self.a) / (self.b - self.a), 1), 0)

    def domain(self) -> NDArray[np.float64]:
        return np.array([self.a, np.inf])


class PiMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        self.a = a
        self.b = b

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + ((x - self.a) / self.b) ** 2.0)

    def domain(self) -> NDArray[np.float64]:
        # TODO - Handle the long-tail of the distribution! :)
        n_sigma = 4.0
        return np.array([self.a - n_sigma * self.b, self.a + n_sigma * self.b])
