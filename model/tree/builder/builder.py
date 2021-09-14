import abc
import numpy as np
from model.tree.node import Node
from utilities import require, value_counts


class ImpurityFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute(self, frequencies) -> float:
        pass


class Entropy(ImpurityFunction):
    def compute(self, frequencies, eps=1e-9) -> float:
        return -(frequencies * np.log2(frequencies + eps)).sum()


class GiniIndex(ImpurityFunction):
    def compute(self, frequencies) -> float:
        return 1 - np.sum(np.square(frequencies))


class DecisionTreeBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self, x, y) -> Node:
        raise NotImplementedError('DecisionTreeBuilder#build')

    @staticmethod
    def compute_impurity(samples, criterion: str = 'entropy') -> float:
        _, probabilities = value_counts(samples, normalise=True)
        fns = {'entropy': Entropy, 'gini': GiniIndex}
        fn: ImpurityFunction = require(fns.get(criterion, None), criterion)()
        return fn.compute(probabilities)

    @staticmethod
    def factory(implementation: str, **kwargs) -> 'DecisionTreeBuilder':
        from model.tree.builder.impl.ID3 import ID3DecisionTreeBuilder  # TODO: imports...
        factories = {
            'ID3': ID3DecisionTreeBuilder
        }
        constructor = require(factories.get(implementation, None), implementation)
        return constructor(**kwargs)

    @staticmethod
    def default() -> 'DecisionTreeBuilder':
        return DecisionTreeBuilder.factory('ID3')
