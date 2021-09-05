import abc
from typing import Optional, TypeVar

import numpy as np
from tqdm.auto import tqdm

ITERATION_LIMIT: int = 1_000

CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'
SAMPLE     = 'azdfhvskfejgsdjhsdfjhsdfhj'
VOWEL: int = 1
CONSONANT: int = -1

is_vowel = lambda char: char in 'aeiou'
assign_class = lambda char: VOWEL if is_vowel(char) else CONSONANT

training_x = np.array([ord(char) for char in CHARACTERS])
training_y = np.array([assign_class(char) for char in CHARACTERS])
testing_x = np.array([ord(char) for char in SAMPLE])
testing_y = np.array([assign_class(char) for char in SAMPLE])


def require(x, field: str):
    if x is None:
        raise ValueError(f'Missing required value: "{field}".')
    return x


# TODO: representation of internal state for externalisation
class State:
    __attributes: dict[str, any] = {}

    def set(self, key: str, value: any):
        self.__attributes[key] = value

    def get(self, key: str, default: any = None):
        return self.__attributes.get(key, default=default)

    def has(self, attr: str) -> bool:
        return attr in self.__attributes


class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compile(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, x, y, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, y, *args, **kwargs):
        pass


class Pivot:

    def __init__(self, description, predicate: lambda x: bool):
        self.__description = require(description, 'description')
        self.__predicate = require(predicate, 'predicate')

    @property
    def description(self):
        return self.__description

    @property
    def predicate(self):
        return self.__predicate

    def split(self, value) -> bool:
        return self.predicate(value)

    def gini_impurity(self, x, y) -> float:
        vector = np.vectorize(lambda v: self.split(v))(x)
        lower = vector[vector == True]
        upper = vector[vector == False]
        return 3.14
        # p_true =


class Node(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def eval(self, value):
        pass


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


class DecisionTree(Model):

    # __tree: Optional[DecisionTreeBuilder]

    def compile(self, *args, **kwargs):
        pass

    def fit(self, x, y, *args, **kwargs):
        pass

    def predict(self, y, *args, **kwargs):
        pass


# TODO: represent a condition as a complex data type (rather than just x <= y)
# TODO: struct (description, lambda(x))


class DecisionTreeBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self, x, y) -> DecisionTree:
        pass

    @abc.abstractmethod
    def purity(self, attribute, x, y):
        pass

    @staticmethod
    def factory(implementation: str, **kwargs):
        factories = {
            'CART': CARTDecisionTreeBuilder,
            'ID3': ID3DecisionTreeBuilder,
            'C4.5': C45DecisionTreeBuilder
        }
        factory = require(factories.get(implementation, default=None), implementation)
        return factory.__init__(**kwargs)


class CARTDecisionTreeBuilder(DecisionTreeBuilder):

    def build(self, x, y) -> DecisionTree:
        pass

    def purity(self, attribute, x, y):
        pass


class ID3DecisionTreeBuilder(DecisionTreeBuilder):

    def purity(self, attribute, x, y):
        # p_true = 105
        # p_false = 0.25
        pass

    def build(self, x, y) -> DecisionTree:
        raise NotImplementedError('TODO: implement ID3DecisionTreeBuilder#build')


class C45DecisionTreeBuilder(DecisionTreeBuilder):

    def purity(self, attribute, x, y):
        pass

    def build(self, x, y) -> DecisionTree:
        pass


class Perceptron(Model):

    def __init_state(self, length):
        mu, sigma = 0, 0.5
        state = self.state = np.random.normal(mu, sigma, length + 1)
        return state

    def compile(self, *args, **kwargs):
        pass
        # self.__weights = kwargs.get('weights', self.__weights)
        # self.__bias = kwargs.get('bias', self.__bias)

    def fit(self, x, y, *args, **kwargs):
        self.__init_state(len(x))
        limit: int = kwargs.get('limit', ITERATION_LIMIT)

        print(f'Limit: {limit}, Initial State: {self.state}')

        for iteration in tqdm(range(limit)):
            predictions = self.predict(x)
            error_indexes = np.nonzero(predictions != y)[0]
            errors = len(error_indexes)

            if errors <= 0:
                tqdm.write('Training complete.')
                break

            index = error_indexes[np.random.randint(errors)]
            self.state[:-1] += x[index] * float(y[index])  # update weights
            self.state[-1] += float(y[index])  # update bias

        print(f'Final State: {self.state}')

    def predict(self, x, *args, **kwargs):
        state = self.state
        hypothesis, bias = state[:-1], state[-1]
        return np.sign(Perceptron.compute_linear_score(x, hypothesis, bias)).astype(int)
        # return (Perceptron.compute_linear_score(x, hypothesis, bias) > 0).astype(int)

    @staticmethod
    def compute_linear_score(x, hypothesis, bias):
        # return (x * hypothesis).sum(axis=1) + bias
        return (x * hypothesis).sum() + bias

print(f'Train X: {training_x}')
print(f'Train Y: {training_y}')

print(f'Test X: {testing_x}')
print(f'Test Y: {testing_y}')

model: Model = Perceptron()
model.fit(training_x, training_y)
predictions = model.predict(np.array([ord('a')]))

print(f'Predicted: {predictions}, Actual: {testing_y}')
