import abc
from typing import Optional
from model.tree.pivot import Pivot
from utilities import require


class Node(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def eval(self, value):
        raise NotImplementedError('Node#eval')

    @staticmethod
    def branch(pivot: Pivot, lower: Optional['Node'] = None, upper: Optional['Node'] = None) -> 'Node':
        return BranchNode(pivot=pivot, lower=lower, upper=upper)

    @staticmethod
    def terminate(value) -> 'Node':
        return TerminalNode(value=value)


class BranchNode(Node):

    __lower: Optional[Node]
    __upper: Optional[Node]

    def __init__(self, pivot: Pivot, lower: Optional[Node] = None, upper: Optional[Node] = None):
        self.__pivot = require(pivot, 'pivot')
        self.__lower = lower
        self.__upper = upper

    def eval(self, value):
        branch: Node = self.lower if self.pivot.split(value) else self.upper
        return branch.eval(value)

    @property
    def pivot(self) -> Pivot:
        return self.__pivot

    @property
    def lower(self) -> Optional[Node]:
        return require(self.__lower, 'lower')

    @lower.setter
    def lower(self, value: Optional[Node]):
        self.__lower = value

    @property
    def upper(self) -> Optional[Node]:
        return require(self.__upper, 'upper')

    @upper.setter
    def upper(self, value: Optional[Node]):
        self.__upper = value


class TerminalNode(Node):

    def __init__(self, value):
        self.__value = require(value, 'value')

    def eval(self, value):
        return self.value

    @property
    def value(self):
        return require(self.__value, 'value')

