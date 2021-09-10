from utilities import require


class Pivot:

    def __init__(self, feature: str, predicate: lambda x: bool):
        self.__feature = require(feature, 'feature')
        self.__predicate = require(predicate, 'predicate')

    @property
    def feature(self) -> str:
        return self.__feature

    @property
    def predicate(self):  # TODO: lambda type returning boolean (predicate)
        return self.__predicate

    def split(self, value: any) -> bool:
        return self.predicate(value)
