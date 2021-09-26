# from utilities import require
#
#
# class Pivot:
#
#     def __init__(self, feature: str, predicate: lambda x: bool):
#         self.__feature = require(feature, 'feature')
#         self.__predicate = require(predicate, 'predicate')
#
#     @property
#     def feature(self):
#         return self.__feature
#
#     @property
#     def predicate(self) -> lambda x: bool:  # TODO: lambda type returning boolean (predicate)
#         return self.__predicate
#
#     def split(self, value: any) -> bool:
#         return self.predicate(value)
#
#     @staticmethod
#     def build(attribute, probe) -> 'Pivot':
#         feature = f'x[{attribute}] <= {probe}'
#
#         def predicate(x):
#             return x[attribute] <= probe
#         return Pivot(feature=feature, predicate=predicate)
#
#     def __str__(self) -> str:
#         return self.feature
