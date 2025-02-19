import abc

import numpy as np
from numpy.typing import NDArray

from AEEM6097.membership_functions import MembershipFunction, FuzzyVariable


# TODO - This is a terrible inversion of control, but without resorting to Python metaclass magic, it's the easiest way to do this.
# TODO - Set the internal workings of different mathematical methods.


class FuzzySet(abc.ABC):
    def __init__(self, var_name: str, mf: list[MembershipFunction]):
        self.membership_functions = mf
        self.var_name = var_name

    def __call__(
        self,
        x: NDArray[np.float64] | FuzzyVariable | list[FuzzyVariable],
        tgt_mf_name: str = "",
    ) -> list[FuzzyVariable]:
        # TODO - Better return type?
        return self.fuzzify(x, tgt_mf_name)

    def __str__(self) -> str:
        return f"FuzzySet:{self.var_name}: {self.membership_functions}"

    def __getitem__(self, item: str) -> MembershipFunction:
        for mf in self.membership_functions:
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

    def __eq__(self, other: str | list[str]) -> "FuzzyEquals":
        if isinstance(other, str):
            return FuzzyEquals(self, other)
        elif isinstance(other, list):
            y = FuzzyEquals(self, other[0])
            for i in range(1, len(other)):
                y = y | FuzzyEquals(self, other[i])
            return y
        else:
            raise TypeError("other must be str or list[str]")

    def __ne__(self, other: str | list[str]) -> "FuzzyNot":
        return FuzzyNot(self == other)

    @property
    def domain(self) -> NDArray[np.float64]:
        all_domains = [mf.domain() for mf in self.membership_functions]
        # TODO - Handle discontinuous domains!!
        return np.array(
            [min([d[0] for d in all_domains]), max([d[1] for d in all_domains])]
        )

    def fuzzify(
        self,
        x: NDArray[np.float64] | FuzzyVariable | list[FuzzyVariable],
        tgt_mf_name: str = "",
    ) -> list[FuzzyVariable]:
        if isinstance(x, list):
            # Ensure these are valid variables
            # TODO - Handle more than one!
            x = [v for v in x if v.var_name == self.var_name][0]
        if isinstance(x, FuzzyVariable):
            if tgt_mf_name:
                if tgt_mf_name not in [mf.name for mf in self.membership_functions]:
                    raise KeyError(
                        f"Membership function {tgt_mf_name} not found in {self}"
                    )
                # TODO - Should this be other than a list?
                return [
                    FuzzyVariable(mf.name, mf.mu(x.value))
                    for mf in self.membership_functions
                    if mf.name == tgt_mf_name
                ]
            return [
                FuzzyVariable(mf.name, mf.mu(x.value))
                for mf in self.membership_functions
            ]
        else:
            raise NotImplementedError("fuzzify() not implemented for NDArrays, yet!")

    def defuzzify(self, mu_x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(
            "defuzzify() must be implemented in subclass by type of model!"
        )


class FuzzyInference(FuzzyVariable):
    def __init__(self, output_set: FuzzySet, var_name: str, mu_value: np.float64):
        super().__init__(var_name, mu_value)
        self.output_set: FuzzySet = output_set

    def __str__(self) -> str:
        return f"FuzzyInference:{self.var_name}:{self.value}"


class FuzzyOperator(abc.ABC):
    def __call__(
        self, *varargs: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        return self.evaluate(*varargs)

    def __str__(self) -> str:
        raise NotImplementedError("__str__() must be implemented in subclass")

    def __and__(self, other: "FuzzyOperator") -> "FuzzyOperator":
        return FuzzyAnd(self, other)

    def __or__(self, other: "FuzzyOperator") -> "FuzzyOperator":
        return FuzzyOr(self, other)

    def __neg__(self) -> "FuzzyOperator":
        return FuzzyNot(self)

    def evaluate(
        self, *varargs: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        raise NotImplementedError("evaluate() must be implemented in subclass")


# These are the defaults, but there are alternative implementations of this!
class FuzzyAnd(FuzzyOperator):
    def __init__(self, a: FuzzyOperator, b: FuzzyOperator):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"({self.a} AND {self.b})"

    def evaluate(
        self, x: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        # TODO - Alternative implementations!
        a1 = self.a(x)
        b1 = self.b(x)
        return np.minimum(a1, b1)


class FuzzyOr(FuzzyOperator):
    def __init__(self, a: FuzzyOperator, b: FuzzyOperator):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"({self.a} OR {self.b})"

    def evaluate(
        self, x: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        # TODO - Alternative implementations!
        a1 = self.a(x)
        b1 = self.b(x)
        return np.maximum(a1, b1)


class FuzzyNot(FuzzyOperator):
    def __init__(self, a: FuzzyOperator):
        self.a = a

    def __str__(self) -> str:
        return f"NOT {self.a}"

    def evaluate(
        self, x: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        a1 = self.a(x)
        return 1 - a1


class FuzzyEquals(FuzzyOperator):
    def __init__(self, a: FuzzySet, b: str):
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"{self.a.var_name} == {self.b}"

    def evaluate(
        self, x: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        # Only look at the variable that matches this set
        if isinstance(x, list):
            req_var: FuzzyVariable = [v for v in x if v.var_name == self.a.var_name][0]
            y = self.a(req_var, self.b)
            # Convert to NDArray, since we know it is filtered to the right variable!
            return np.array([vy.value for vy in y])
        else:
            raise NotImplementedError("evaluate() not implemented for NDArrays, yet!")


class FuzzyRule:
    def __init__(self, rule_name: str, antecedent: FuzzyOperator, consequent: FuzzySet):
        self.rule_name = rule_name
        self.antecedent = antecedent
        self.consequent = consequent

    def __str__(self) -> str:
        return f"{self.rule_name}: IF {self.antecedent} THEN {self.consequent}"

    def __call__(
        self, x: NDArray[np.float64] | list[FuzzyVariable]
    ) -> NDArray[np.float64]:
        return self.evaluate(x)

    def evaluate(self, x: NDArray[np.float64] | list[FuzzyVariable]) -> FuzzyInference:
        raise NotImplementedError("evaluate() must be implemented in subclass")


class MamdaniRule(FuzzyRule):
    def __init__(
        self,
        rule_name: str,
        antecedent: FuzzyOperator,
        consequent: FuzzySet,
        consequent_target: str,
    ):
        super().__init__(rule_name, antecedent, consequent)
        self.consequent_target = consequent_target

    def __str__(self) -> str:
        return f"{self.rule_name}: IF ({self.antecedent}) THEN {self.consequent.var_name} = {self.consequent_target}"

    def __call__(self, x: NDArray[np.float64] | list[FuzzyVariable]) -> FuzzyInference:
        return self.evaluate(x)

    def evaluate(self, x: NDArray[np.float64] | list[FuzzyVariable]) -> FuzzyInference:
        mu_x = self.antecedent(x)
        return FuzzyInference(self.consequent, self.consequent_target, mu_x)


class FuzzySystem(abc.ABC):
    def __init__(self, fs: FuzzySet):
        self.fs = fs

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.inference(x)

    def inference(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("inference() must be implemented in subclass")


# TODO - Implement the Mamdani Fuzzy System class for clarity!
