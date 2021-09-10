import abc
from model.tree.builder.impl.ID3 import ID3DecisionTreeBuilder
from model.tree.node import Node
from utilities import require


class DecisionTreeBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self, x, y) -> Node:
        raise NotImplementedError('DecisionTreeBuilder#build')

    @abc.abstractmethod
    def purity(self, attribute, x, y) -> float:
        raise NotImplementedError('DecisionTreeBuilder#purity')

    @staticmethod
    def factory(implementation: str, **kwargs) -> 'DecisionTreeBuilder':
        factories = {
            'ID3': ID3DecisionTreeBuilder
        }
        constructor = require(factories.get(implementation, default=None), implementation)
        return constructor(**kwargs)

    @staticmethod
    def default() -> 'DecisionTreeBuilder':
        return DecisionTreeBuilder.factory('ID3')
