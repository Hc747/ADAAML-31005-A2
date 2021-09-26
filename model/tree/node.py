# import abc
# from typing import Optional
# from model.tree.pivot import Pivot
# from utilities import require, default
#
#
# class TranslationFunction(metaclass=abc.ABCMeta):
#     @abc.abstractmethod
#     def translate(self, x: any) -> any:
#         pass
#
#
# class IdentityTranslationFunction(TranslationFunction):
#     def translate(self, x: any) -> any:
#         return x
#
#
# class LookupTranslationFunction(TranslationFunction):
#
#     def __init__(self, lookup: lambda x: any):
#         self.__lookup = require(lookup, 'lookup')
#
#     def translate(self, x: any) -> any:
#         return self.__lookup(x)
#
#
# class Node(metaclass=abc.ABCMeta):
#
#     @abc.abstractmethod
#     def eval(self, value):
#         raise NotImplementedError('Node#eval')
#
#     @abc.abstractmethod
#     def render(self, out, depth: int = 0):
#         pass
#
#     @staticmethod
#     def branch(pivot: Pivot, lower: Optional['Node'] = None, upper: Optional['Node'] = None) -> 'Node':
#         return BranchNode(pivot=pivot, lower=lower, upper=upper)
#
#     @staticmethod
#     def lookup(mapping: dict[any, 'Node'], translator: TranslationFunction) -> 'Node':
#         return LookupNode(mapping=mapping, translator=translator)
#
#     @staticmethod
#     def terminate(value) -> 'Node':
#         return TerminalNode(value=value)
#
#
# class BranchNode(Node):
#
#     __lower: Optional[Node]
#     __upper: Optional[Node]
#
#     def __init__(self, pivot: Pivot, lower: Node, upper: Node):
#         self.__pivot = require(pivot, 'pivot')
#         self.__lower = require(lower, 'lower')
#         self.__upper = require(upper, 'upper')
#
#     def eval(self, value):
#         branch: Node = self.lower if self.pivot.split(value) else self.upper
#         return branch.eval(value)
#
#     @property
#     def pivot(self) -> Pivot:
#         return self.__pivot
#
#     @property
#     def lower(self) -> Optional[Node]:
#         return self.__lower
#
#     @property
#     def upper(self) -> Optional[Node]:
#         return self.__upper
#
#     def render(self, out, depth: int = 0):
#         padding = ('-' * depth) + '>'
#         out(f'({depth}) {padding} Branch')
#         out(f'({depth}) {padding} Pivot: {self.pivot}')
#         out(f'({depth}) {padding} Lower:')
#         if self.lower is not None:
#             self.lower.render(out, depth=depth + 1)
#         out(f'({depth}) {padding} Upper:')
#         if self.upper is not None:
#             self.upper.render(out, depth=depth + 1)
#
#
# class LookupNode(Node):
#
#     __mapping: dict[any, Node]
#     __translator: TranslationFunction
#
#     def __init__(self, mapping: dict[any, Node], translator: Optional[TranslationFunction]):
#         self.__mapping = require(mapping, 'mapping')
#         self.__translator = default(translator, IdentityTranslationFunction())
#
#     def eval(self, value):
#         lookup = self.translator.translate(value)
#         return require(self.mapping[lookup], 'eval')
#
#     def render(self, out, depth: int = 0):
#         padding = ('-' * depth) + '>'
#         for (key, value) in self.mapping.items():
#             out(f'({depth}) {padding} {key}')
#             value.render(out, depth=depth + 1)
#
#     @property
#     def mapping(self) -> dict[any, Node]:
#         return self.__mapping
#
#     @property
#     def translator(self) -> TranslationFunction:
#         return self.__translator
#
#
# class TerminalNode(Node):
#
#     def __init__(self, value):
#         self.__value = require(value, 'value')
#
#     def eval(self, value):
#         return self.value
#
#     @property
#     def value(self):
#         return require(self.__value, 'value')
#
#     def render(self, out, depth: int = 0):
#         padding = ('-' * depth) + '>'
#         out(f'({depth}) {padding} Terminal: {self.value}')
