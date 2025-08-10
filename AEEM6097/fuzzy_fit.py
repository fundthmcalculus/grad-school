from dataclasses import dataclass
from typing import Any, Generator

from scipy.signal import find_peaks, peak_prominences, peak_widths

from AEEM6097.aco_solver import AcoContinuousVariable
from AEEM6097.fuzzy_types import *


class VariablesInfo:
    def __init__(self):
        self.variables: list[AcoContinuousVariable] = []
        self.mu_indexes: list[list[int]] = []
        self.rule_op_indexes: list[int] = []
        self.rule_coeff_indexes: list[list[int,]] = []
        self.rule_args: ai64 | None = None

    def find_variable(self, name_prefix: str) -> tuple[int, AcoContinuousVariable | None]:
        for idx, variable in enumerate(self.variables):
            if variable.name.startswith(name_prefix):
                return idx, variable
        return -1, None

    def append_variables(self, x: AcoContinuousVariable | list[AcoContinuousVariable]) -> None:
        # Check the name for type information.
        if isinstance(x, AcoContinuousVariable):
            var_idx = len(self.variables)
            self.variables.append(x)
            var_name = x.name
        else:
            var_idx = list(range(len(self.variables),len(self.variables)+len(x)))
            self.variables.extend(x)
            var_name = x[0].name
        if "and/or-op" in var_name:
            if isinstance(x, AcoContinuousVariable):
                self.rule_op_indexes.append(var_idx)
            else:
                raise Exception("lists of Operator variables are not supported")
        elif "mu_" in var_name:
            if isinstance(x, list):
                self.mu_indexes.append(var_idx)
            else:
                raise Exception("single element membership functions are not supported")
        elif "rule-coeff" in var_name:
            if isinstance(x, list):
                self.rule_coeff_indexes.append(var_idx)
            else:
                # TODO - Handle 0th-order and 1st-order
                raise Exception("single element TSK rules are not supported")
        else:
            raise Exception("variable name is not supported")


    @property
    def n_membership_fcns(self) -> int:
        return len(self.mu_indexes)

    @property
    def n_rules(self) -> int:
        return len(self.rule_args)


@dataclass
class DataScaling:
    data_min: af64
    data_max: af64


@dataclass
class FuzzyDataSet:
    train_data: af64
    test_data: af64
    scale_info: DataScaling
    labels: list[str]

    @staticmethod
    def create_from_data(data: af64, test_percent: f64, labels: list[str]) -> "FuzzyDataSet":
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - data_min) / (data_max - data_min)
        test_size = int(len(data) * test_percent)
        return FuzzyDataSet(train_data=data[:-test_size],
                            test_data=data[-test_size:],
                            scale_info=DataScaling(data_min=data_min, data_max=data_max),
                            labels=labels)

    def scale_data(self, data: af64) -> af64:
        return (data - self.scale_info.data_min) / (self.scale_info.data_max - self.scale_info.data_min)

    def unscale_data(self, data: af64, index: int | None = None) -> af64:
        if index is None:
            return data * (self.scale_info.data_max - self.scale_info.data_min) + self.scale_info.data_min
        else:
            return data * (self.scale_info.data_max[index] - self.scale_info.data_min[index]) + self.scale_info.data_min[index]


@dataclass
class PeakInfo:
    x: i64
    # left_base: i64
    # right_base: i64
    # prominence: f64
    y: f64
    half_width: f64


class PeakCollection(list[PeakInfo]):
    def __init__(self, x: af64, data_pdf: af64, ):
        super().__init__()
        self.extend(PeakCollection.get_peak_data(x, data_pdf))

    @staticmethod
    def get_peak_data(x: af64, data_pdf: af64) -> Generator[PeakInfo, Any, None]:
        mod_pdf = np.zeros(data_pdf.shape[0] + 2)
        mod_pdf[1:-1] = data_pdf
        peak_indexes, peak_info = find_peaks(mod_pdf)
        prominences = peak_prominences(mod_pdf, peak_indexes)
        results_half = peak_widths(mod_pdf, peak_indexes, prominence_data=prominences)
        for ij, peak_idx in enumerate(peak_indexes):
            # Order is left-half-idx, half-y, width in samples, right-half-idx
            scaled_width = results_half[2][ij] / len(x)
            yield PeakInfo(x=x[min(peak_idx,len(x)-1)],
                           y=mod_pdf[peak_idx],
                           # left_base=x[max(0, peak_info['left_bases'][ij])],
                           # right_base=x[min(peak_info['right_bases'][ij], len(x) - 1)],
                           # prominence=peak_info['prominences'][ij],
                           half_width=scaled_width)
