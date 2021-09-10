import abc


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
