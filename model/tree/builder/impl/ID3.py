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

        maxima = -math.inf

    def purity(self, attribute, x, y) -> float:
        raise NotImplementedError('ID3DecisionTreeBuilder#purity')
