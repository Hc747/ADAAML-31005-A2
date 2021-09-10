import numpy as np
from typing import Optional
from model.model import Model
from model.tree.builder.builder import DecisionTreeBuilder
from model.tree.node import Node
from utilities import require


class DecisionTree(Model):

    # TODO: default value (most common class..?)
    __builder: DecisionTreeBuilder = DecisionTreeBuilder.default()
    __root: Optional[Node] = None

    def compile(self, *args, **kwargs):
        try:
            builder = DecisionTreeBuilder.factory(kwargs['implementation'], **kwargs)
        except (KeyError, ValueError):
            builder = DecisionTreeBuilder.default()
        self.__builder = builder

    def fit(self, x, y, *args, **kwargs):
        self.__root = self.builder.build(x, y)

    def predict(self, x, *args, **kwargs):
        tree = self.root
        samples = x.shape[0]
        predictions = np.zeros(samples)  # TODO: default prediction not ZEROs
        for index in range(samples):
            sample = x[index]
            predictions[index] = tree.eval(sample)
        return predictions

    @property
    def builder(self) -> DecisionTreeBuilder:
        return require(self.__builder, 'builder')

    @property
    def root(self) -> Node:
        return require(self.__root, 'root')
