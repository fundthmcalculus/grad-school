import abc

import numpy as np
from numpy.typing import NDArray

from AEEM6097.membership_functions import MembershipFunction


# TODO - This is a terrible inversion of control, but without resorting to Python metaclass magic, it's the easiest way to do this.


class FuzzySet(abc.ABC):
    def __init__(self, var_name: str, mf: list[MembershipFunction]):
        self.mf = mf
        self.var_name = var_name

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fuzzify(x)

    def __str__(self) -> str:
        return f"FuzzySet:{self.var_name}: {self.mf}"

    def __and__(self, other: "FuzzySet") -> "FuzzyOperator":
        return FuzzyAnd(self, other)

    def __or__(self, other: "FuzzySet") -> "FuzzyOperator":
        return FuzzyOr(self, other)

    def __neg__(self) -> "FuzzyOperator":
        return FuzzyNot(self)

    def __getitem__(self, item: str):
        for mf in self.mf:
            if mf.name == item:
                return mf
        raise KeyError(f"Membership function {item} not found in {self}")

    def __contains__(
        self, item: float | NDArray[np.float64]
    ) -> bool | NDArray[np.float64]:
        if isinstance(item, float):
            return self.domain[0] <= item <= self.domain[1]
        elif isinstance(item, NDArray[np.float64]):
            return np.logical_and(self.domain[0] <= item, item <= self.domain[1])
        else:
            raise TypeError("item must be float or NDArray[np.float64]")

    @property
    def domain(self) -> NDArray[np.float64]:
        all_domains = [mf.domain for mf in self.mf]
        # TODO - Handle discontinuous domains!!
        return np.array(
            [min([d[0] for d in all_domains]), max([d[1] for d in all_domains])]
        )

    def fuzzify(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([mf(x) * mf.in_domain(x) for mf in self.mf])

    def defuzzify(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(
            "defuzzify() must be implemented in subclass by type of model!"
        )


class FuzzySystem(abc.ABC):
    def __init__(self, fs: FuzzySet):
        self.fs = fs

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.inference(x)

    def inference(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("inference() must be implemented in subclass")


class TKSRule:
    def __init__(self, name: str, linear_scaling: NDArray[np.float64] | None):
        # TODO - Other options!
        self.name = name
        self.linear_scaling = linear_scaling

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.name}, {self.linear_scaling}"

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.evaluate(x)

    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # TODO - Handle other methods!
        return np.dot(x, self.linear_scaling)


class TKSFuzzySystem(FuzzySystem):
    def __init__(self, fs: FuzzySet, rules: list[TKSRule]):
        super().__init__(fs)
        self.rules = rules

    def inference(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        mu_x = self.fs(x)
        rule_x = np.array([rule(mu_x) for rule in self.rules])
        return np.dot(mu_x, rule_x) / np.sum(mu_x)


class FuzzyOperator(abc.ABC):
    def __init__(self, a: FuzzySet, b: FuzzySet):
        self.a = a
        self.b = b
        pass

    def __call__(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.evaluate(x, y)

    def __str__(self) -> str:
        return f"{self.a} {self.__class__.__name__} {self.b}"

    def evaluate(self, *varargs: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("evaluate() must be implemented in subclass")


# These are the defaults, but there are alternative implementations of this!
class FuzzyAnd(FuzzyOperator):
    def evaluate(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.minimum(x, y)


class FuzzyOr(FuzzyOperator):
    def evaluate(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.maximum(x, y)


class FuzzyNot(FuzzyOperator):
    # TODO - Type hint this better?
    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 - x

class FuzzyEquals(FuzzyOperator):
    def evaluate(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.isclose(x, y)


class FuzzyRule:
    def __init__(self, antecedent: FuzzyOperator, consequent: FuzzySet):
        self.antecedent = antecedent
        self.consequent = consequent

    def __str__(self) -> str:
        return f"IF {self.antecedent} THEN {self.consequent}"

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.evaluate(x)

    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("evaluate() must be implemented in subclass")
