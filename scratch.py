import abc
import numpy as np

from typing import Optional


# Utility functions.
def require(value: Optional[any], field: str):
    if value is None:
        raise ValueError(f'Missing required value: "{field}".')
    return value


def default(value: Optional[any], otherwise: any) -> any:
    return otherwise if value is None else value


def is_numeric(x):
    return x.dtype == int or x.dtype == float or x.dtype == bool


def value_counts(y, normalise: bool = True):
    classes, counts = np.unique(y, return_counts=True)
    if normalise:
        return classes, counts / (y if np.isscalar(y) else len(y))
    return classes, counts


# Interface definitions.
class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compile(self, *args, **kwargs):
        raise NotImplementedError('Model#compile')

    @abc.abstractmethod
    def fit(self, x, y, *args, **kwargs):
        raise NotImplementedError('Model#fit')

    @abc.abstractmethod
    def predict(self, x, *args, **kwargs):
        raise NotImplementedError('Model#predict')


class SplitCriterionFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute(self, frequencies) -> float:
        pass


class Entropy(SplitCriterionFunction):
    def compute(self, frequencies, eps=1e-9) -> float:
        return -(frequencies * np.log2(frequencies + eps)).sum()


class GiniIndex(SplitCriterionFunction):
    def compute(self, frequencies) -> float:
        return 1 - np.sum(np.square(frequencies))


class MappingFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def translate(self, value: any) -> any:
        pass


class IdentityMappingFunction(MappingFunction):
    def translate(self, value: any) -> any:
        return value


class LookupMappingFunction(MappingFunction):
    def __init__(self, lookup: lambda value: any):
        self.__lookup = require(lookup, 'lookup')

    def translate(self, value: any) -> any:
        return self.__lookup(value)


class Pivot:

    def __init__(self, feature: str, predicate: lambda x: bool):
        self.__feature = require(feature, 'feature')
        self.__predicate = require(predicate, 'predicate')

    @property
    def feature(self):
        return self.__feature

    @property
    def predicate(self) -> lambda x: bool:  # TODO: lambda type returning boolean (predicate)
        return self.__predicate

    def split(self, value: any) -> bool:
        return self.predicate(value)

    @staticmethod
    def build(attribute, probe) -> 'Pivot':
        feature = f'x[{attribute}] <= {probe}'

        def predicate(x):
            return x[attribute] <= probe
        return Pivot(feature=feature, predicate=predicate)

    def __str__(self) -> str:
        return self.feature


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
        print(f'before: {self}')
        self.__feature = feature
        self.__gain = gain
        self.__probe = probe
        print(f'after:  {self}')
        return True

    @staticmethod
    def initial() -> 'PivotCandidate':
        return PivotCandidate(0, 0, 0.5)
        # return PivotCandidate(0, -math.inf, 0.5)

    def __str__(self):
        return f'feature: {self.__feature}, gain: {self.__gain}, probe: {self.__probe}'


class Node(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def eval(self, value):
        raise NotImplementedError('Node#eval')

    @abc.abstractmethod
    def render(self, out, depth: int = 0):
        pass

    @staticmethod
    def branch(pivot: 'Pivot', lower: Optional['Node'] = None, upper: Optional['Node'] = None) -> 'Node':
        return BranchNode(pivot=pivot, lower=lower, upper=upper)

    @staticmethod
    def lookup(mapping: dict[any, 'Node'], translator: 'MappingFunction') -> 'Node':
        return LookupNode(mapping=mapping, translator=translator)

    @staticmethod
    def terminate(value) -> 'Node':
        return TerminalNode(value=value)


class BranchNode(Node):

    __lower: Optional[Node]
    __upper: Optional[Node]

    def __init__(self, pivot: Pivot, lower: Node, upper: Node):
        self.__pivot = require(pivot, 'pivot')
        self.__lower = require(lower, 'lower')
        self.__upper = require(upper, 'upper')

    def eval(self, value):
        branch: Node = self.lower if self.pivot.split(value) else self.upper
        return branch.eval(value)

    @property
    def pivot(self) -> Pivot:
        return self.__pivot

    @property
    def lower(self) -> Optional[Node]:
        return self.__lower

    @property
    def upper(self) -> Optional[Node]:
        return self.__upper

    def render(self, out, depth: int = 0):
        padding = ('-' * depth) + '>'
        out(f'({depth}) {padding} Branch')
        out(f'({depth}) {padding} Pivot: {self.pivot}')
        out(f'({depth}) {padding} Lower:')
        if self.lower is not None:
            self.lower.render(out, depth=depth + 1)
        out(f'({depth}) {padding} Upper:')
        if self.upper is not None:
            self.upper.render(out, depth=depth + 1)


class LookupNode(Node):

    __mapping: dict[any, Node]
    __translator: MappingFunction

    def __init__(self, mapping: dict[any, Node], translator: Optional[MappingFunction]):
        self.__mapping = require(mapping, 'mapping')
        self.__translator = default(translator, IdentityMappingFunction())

    def eval(self, value):
        lookup = self.translator.translate(value)
        return require(self.mapping[lookup], 'eval')

    def render(self, out, depth: int = 0):
        padding = ('-' * depth) + '>'
        for (key, value) in self.mapping.items():
            out(f'({depth}) {padding} {key}')
            value.render(out, depth=depth + 1)

    @property
    def mapping(self) -> dict[any, Node]:
        return self.__mapping

    @property
    def translator(self) -> MappingFunction:
        return self.__translator


class TerminalNode(Node):

    def __init__(self, value):
        self.__value = require(value, 'value')

    def eval(self, value):
        return self.value

    @property
    def value(self):
        return require(self.__value, 'value')

    def render(self, out, depth: int = 0):
        padding = ('-' * depth) + '>'
        out(f'({depth}) {padding} Terminal: {self.value}')


class DecisionTreeBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self, x, y) -> Node:
        raise NotImplementedError('DecisionTreeBuilder#build')

    @staticmethod
    def compute_impurity(samples, criterion: str = 'entropy') -> float:
        _, probabilities = value_counts(samples, normalise=True)
        fns = {'entropy': Entropy, 'gini': GiniIndex}
        fn: SplitCriterionFunction = require(fns.get(criterion, None), criterion)()
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
        # cached to prevent expensive access or concurrent modification (... if python has concurrent modification)
        tree = self.root
        return np.asarray([tree.eval(sample) for sample in x])

    @property
    def builder(self) -> DecisionTreeBuilder:
        return require(self.__builder, 'builder')

    @property
    def root(self) -> Node:
        return require(self.__root, 'root')
