import math
import numpy as np
from model.tree.builder.builder import DecisionTreeBuilder
from model.tree.node import Node
from model.tree.pivot import Pivot
from utilities import require


class PivotCandidate:

    def __init__(self, feature: int, gain: float, probe: float):
        self.__feature = feature
        self.__gain = gain
        self.__probe = probe

    # @property
    def feature(self) -> int:
        return require(self.__feature, 'feature')

    # @property
    def gain(self) -> float:
        return require(self.__gain, 'gain')

    # @property
    def probe(self) -> float:
        return require(self.__probe, 'probe')

    def update(self, feature: int, gain: float, probe: float) -> bool:
        if gain < self.gain():
            return False
        self.__feature = feature
        self.__gain = gain
        self.__probe = probe
        return True

    @staticmethod
    def initial() -> 'PivotCandidate':
        return PivotCandidate(0, -math.inf, 0.5)


class ID3DecisionTreeBuilder(DecisionTreeBuilder):

    def build(self, x, y) -> Node:
        classes = np.unique(y)
        choices = len(classes)

        if choices <= 0:  # no choices
            default_value = '<default-value>'  # TODO: get default value
            return Node.terminate(default_value)

        if choices == 1:  # one clear choice
            return Node.terminate(classes[0])

        candidate: PivotCandidate = PivotCandidate.initial()
        attributes = x.shape[1]
        for index in range(attributes):
            # TODO: handle continuous and categorical attributes
            attribute = x[:, index]
            probes = ID3DecisionTreeBuilder.create_probe_values(attribute.min(), attribute.max())
            for probe in probes:
                # gain = self.purity(attribute, probe, x, y)  # TODO: compute information gain
                gain = self.measure_progress(y, attribute, probe)
                candidate.update(feature=index, gain=gain, probe=probe)

        # TODO: sanity check candidate or build pivot from candidate
        pivot: Pivot = Pivot.build(candidate.feature(), candidate.probe())
        # TODO: use or apply pivot data-structure and make more efficient...
        idx_lower = x[:, candidate.feature()] <= candidate.probe()
        idx_upper = x[:, candidate.feature()] > candidate.probe()

        return Node.branch(pivot, self.build(x[idx_lower], y[idx_lower]), self.build(x[idx_upper], y[idx_upper]))

    def purity(self, attribute, x, y) -> float:
        return np.random.random()
        # raise NotImplementedError('ID3DecisionTreeBuilder#purity')

    def measure_progress(self, y, attribute, threshold) -> float:
        return 0.0

    @staticmethod
    def compute_entropy(y, eps=1e-9) -> float:
        classes, counts = np.unique(y, return_counts=True)
        frequency = counts / len(y)
        return -(frequency * np.log2(frequency + eps)).sum()

    @staticmethod
    def create_probe_values(minima, maxima):
        return [v * minima + (1.0 - v) * maxima for v in [0.75, 0.5, 0.25]]  # TODO: expand values

