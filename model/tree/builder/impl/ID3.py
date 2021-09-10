import math
import numpy as np
from model.tree.builder.builder import DecisionTreeBuilder
from model.tree.node import Node


class ID3DecisionTreeBuilder(DecisionTreeBuilder):

    def build(self, x, y) -> Node:
        classes = np.unique(y)
        choices = len(classes)

        if choices <= 0:  # no choices
            default_value = 0  # TODO: get default value
            return Node.terminate(default_value)

        if choices == 1:  # one clear choice
            return Node.terminate(classes[0])

        attributes = x.shape[1]
        for index in range(attributes):
            attribute = x[:, index]
            probes = ID3DecisionTreeBuilder.create_probe_values(attribute.min(), attribute.max())
            for probe in probes:
                pass
                # TODO: measure gain ...
            pass
        # maxima = -math.inf
        # TODO: assume attributes are continuous (account for categorical later)

    def purity(self, attribute, x, y) -> float:
        raise NotImplementedError('ID3DecisionTreeBuilder#purity')

    @staticmethod
    def create_probe_values(minima, maxima):
        return [v * minima + (1.0 - v) * maxima for v in [0.75, 0.5, 0.25]]  # TODO: expand values

