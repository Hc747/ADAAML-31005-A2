import math
import numpy as np
from model.tree.builder.builder import DecisionTreeBuilder
from model.tree.node import Node
from model.tree.pivot import Pivot
from utilities import require, value_counts
from sklearn.datasets import load_iris as dataset


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


class ID3DecisionTreeBuilder(DecisionTreeBuilder):

    def build(self, x, y) -> Node:
        classes = np.unique(y)
        choices = len(classes)

        if choices <= 0:  # edge-case: no choices
            default_value = '<todo:default-value>'  # TODO: get default value
            return Node.terminate(default_value)

        if choices == 1:  # edge-case: one clear choice
            return Node.terminate(classes[0])

        candidate: PivotCandidate = PivotCandidate.initial()
        attributes = x.shape[1]
        for index in range(attributes):
            # TODO: handle continuous and categorical attributes
            attribute = x[:, index]  # array of all values at that index

            print(f'attribute => {attribute}')
            # print(f'build')
            # print(f'index: {index}, attribute: {attribute}')
            probes = ID3DecisionTreeBuilder.create_probe_values(attribute.min(), attribute.max())
            # print(f'probes: {probes}')
            for probe in probes:
                gain = ID3DecisionTreeBuilder.compute_information_gain(y, attribute, probe)
                # gain = np.random.random()

                # gain = 0.0  # compute_gain(samples, attribute, target)
                # gain = self.purity(attribute, probe, x, y)  # TODO: compute information gain
                # gain = self.measure_progress(y, attribute, probe)
                # gain = self.purity(attribute, )
                candidate.update(feature=index, gain=gain, probe=probe)

        # TODO: sanity check candidate or build pivot from candidate
        pivot: Pivot = Pivot.build(candidate.feature(), candidate.probe())
        # TODO: use or apply pivot data-structure and make more efficient...
        idx_lower = x[:, candidate.feature()] <= candidate.probe()
        idx_upper = x[:, candidate.feature()] > candidate.probe()

        def build_index(indices) -> Node:
            return self.build(x[indices], y[indices])

        return Node.branch(pivot, build_index(idx_lower), build_index(idx_upper))

    @staticmethod
    def measure_progress(y, attribute, target, criterion: str = 'entropy'):
        size = len(y)
        lte, gt = attribute <= target, attribute > target
        total_e = DecisionTreeBuilder.compute_impurity(y, criterion=criterion)
        lower_e = DecisionTreeBuilder.compute_impurity(y[lte], criterion=criterion)
        upper_e = DecisionTreeBuilder.compute_impurity(y[gt], criterion=criterion)
        lower_w = np.count_nonzero(lte) / size
        upper_w = np.count_nonzero(gt) / size

        return total_e - (lower_w * lower_e + upper_w * upper_e)

    @staticmethod
    def compute_information_gain(samples, attribute, target) -> float:
        return ID3DecisionTreeBuilder.measure_progress(samples, attribute, target)
        # classes, frequencies = value_counts(samples, normalise=True)
        # total: float = DecisionTreeBuilder.compute_impurity(samples=target)
        # cumulative: float = 0
        # print(f'compute_information_gain')
        # print(f'total: {total}')
        # print(f'classes: {classes}')
        # print(f'frequencies: {frequencies}')
        # print(f'samples: {samples}')
        # print(f'attribute: {attribute}')
        # print(f'target: {target}')
        # print()
        # print()
        # for (value, frequency) in zip(classes, frequencies):
        #     print(f'class: {value}, frequency: {frequency}')
        #     indices = attribute[attribute <= target]
        #     # indices = [0]
        #     # indices = attributes[]
        #     # indices = samples[attribute] == value
        #     # indices = samples[attribute == value]
        #     print(f'indices {indices}')
        #     # print(f'indices: {indices}')
        #     contribution = DecisionTreeBuilder.compute_impurity(target[indices])
        #     cumulative += frequency * contribution
        # return total - cumulative

    # def purity(self, attribute, x, y) -> float:
    #     return 0.0
        # return np.random.random()
        # raise NotImplementedError('ID3DecisionTreeBuilder#purity')

    # def measure_progress(self, y, attribute, threshold) -> float:
    #     return 0.0

    # def compute_information_gain(self, samples, attribute, target):
    #     total = DecisionTreeBuilder.compute_impurity(samples[attribute], criterion='entropy')
        # values = value_counts(samples[attribute], normalise=True)

    @staticmethod
    def create_probe_values(minima, maxima):
        return [v * minima + (1.0 - v) * maxima for v in [0.75, 0.5, 0.25]]  # TODO: expand values


if __name__ == '__main__':
    from model.tree.tree import DecisionTree
    [x, y] = dataset(return_X_y=True)
    # x = np.array([
    #     np.array([123, 'hello']),
    #     np.array([324, 'hola']),
    #     np.array([453, 'bonjour']),
    #     np.array([345345, 'goodbye'])
    # ])
    # y = np.array(['greeting', 'greeting', 'greeting', 'farewell'])
    print('x', x, x.shape, x.dtype)
    print('y', y, y.shape, y.dtype)
    # builder: ID3DecisionTreeBuilder = ID3DecisionTreeBuilder()
    # print('builder', builder)
    # node: Node = builder.build(x, y)
    # node.render(out=print, depth=0)

    tree = DecisionTree()
    tree.fit(x, y)
    predictions = tree.predict(x)
    print(f'predictions: {predictions}\n\n\n{y}')