import abc

import numpy as np
from numpy.typing import NDArray


class FuzzyVariable:
    def __init__(self, var_name: str, value: NDArray[np.float64]):
        self.var_name = var_name
        self.value = value

    def __str__(self) -> str:
        return f"FuzzyVariable:{self.var_name}: {self.value}"


class MembershipFunction(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        pass

    def __call__(
        self, x: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        return self.mu(x)

    def __str__(self):
        return f"{self.__class__.__name__}:{self.name}:{self.domain()}"

    def __repr__(self) -> str:
        return self.__str__()

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("mu() must be implemented in subclass")

    def inverse_mu(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("inverse_mu() must be implemented in subclass")

    def domain(self) -> NDArray[np.float64]:
        raise NotImplementedError("domain() must be implemented in subclass")

    def in_domain(self, x: NDArray[np.float64] | float) -> bool:
        if isinstance(x, float):
            x = np.array([x])
        return np.all(np.logical_and(self.domain()[0] <= x, x <= self.domain()[1]))

    def centroid(self) -> float:
        raise NotImplementedError("centroid() must be implemented in subclass")

    def d_dx(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("derivative() must be implemented in subclass")

    def gradiant(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("gradient() must be implemented in subclass")

    def hessian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("hessian() must be implemented in subclass")


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

    def centroid(self) -> float:
        C_ab = 2 * self.b / 3 + self.a / 3
        A_ab = 0.5 * (self.b - self.a)
        C_bc = 2 * self.b / 3 + self.c / 3
        A_bc = 0.5 * (self.c - self.b)
        return (C_ab * A_ab + C_bc * A_bc) / (A_ab + A_bc)


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


class LeftShoulderMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(np.minimum((self.b - x) / (self.b - self.a), 1), 0)

    def domain(self) -> NDArray[np.float64]:
        # TODO - Handle technically infinite domain?
        return np.array([self.a, self.b])

    def centroid(self) -> float:
        return 2 * self.a / 3 + self.b / 3


class RightShoulderMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(np.minimum((x - self.a) / (self.b - self.a), 1), 0)

    def domain(self) -> NDArray[np.float64]:
        # TODO - Handle technically infinite domain?
        return np.array([self.a, self.b])

    def centroid(self) -> float:
        return 2 * self.b / 3 + self.a / 3


def create_uniform_triangle_memberships(
    name: str | list[str], x0: float, x1: float, n_fcns: int
) -> list[LeftShoulderMF | TriangularMF | RightShoulderMF]:
    n_fcns = int(n_fcns)
    if isinstance(name, str):
        name = [f"{name}-{i}" for i in range(n_fcns)]
    spacing = (x1 - x0) / (n_fcns - 1)
    all_mus: list[LeftShoulderMF | TriangularMF | RightShoulderMF] = [LeftShoulderMF(name[0], x0, x0 + spacing)]
    for ij in range(1, n_fcns - 1):
        all_mus.append(
            TriangularMF(
                name[ij],
                x0 + (ij - 1) * spacing,
                x0 + ij * spacing,
                x0 + (ij + 1) * spacing,
            )
        )
    all_mus.append(RightShoulderMF(name[-1], x0 + (n_fcns - 2) * spacing, x1))
    return all_mus


def create_triangle_memberships(
    triangle_data: dict[str, float],
) -> list[LeftShoulderMF | TriangularMF | RightShoulderMF]:
    all_mus: list[LeftShoulderMF | TriangularMF | RightShoulderMF] = []
    items = list(triangle_data.items())
    for idx in range(len(items)):
        name = items[idx][0]
        if idx == 0:
            all_mus.append(LeftShoulderMF(name, items[idx][1], items[idx + 1][1]))
        elif idx == len(items) - 1:
            all_mus.append(RightShoulderMF(name, items[idx - 1][1], items[idx][1]))
        else:
            a = items[idx - 1][1]
            b = items[idx][1]
            c = items[idx + 1][1]
            all_mus.append(TriangularMF(name, a, b, c))
    return all_mus


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

    def centroid(self) -> float:
        return self.a

    def inverse_mu(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.a + self.b * np.sqrt(1 / y - 1)


class GuassianMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float):
        super().__init__(name)
        assert a <= b
        self.a = a
        self.b = b

    def mu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(-((x-self.a) / self.b)**2)

    def domain(self) -> NDArray[np.float64]:
        # TODO - Handle the long-tail of the distribution, since the domain is technically [-inf, inf]
        n_sigma = 4.0
        return np.array([self.a - n_sigma * self.b, self.a + n_sigma * self.b])

    def centroid(self) -> float:
        return self.a

    def inverse_mu(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        # Y = exp(- (x-a)^2 /b^2)
        # TODO - Handle the other side option.
        return np.sqrt(-self.b**2 * np.log(y))+self.a

    def d_dx(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.mu(x) * -2.0*(x-self.a)/self.b

    # TODO - Gradient and Hessian!
