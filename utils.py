from collections.abc import Hashable
from types import UnionType
from typing import Any, Dict, NoReturn, Tuple, TypeVar, Type

T = TypeVar("T")
TK = TypeVar("TK", bound=Hashable)
TV = TypeVar("TV")

def raise_type_err(obj: Any, t: Type[T] | UnionType) -> NoReturn:
    raise TypeError(f"Expected {t} but got {type(obj).__name__} with value: {obj}")

def assert_type(obj: Any, t: Type[T]) -> T:
    if not isinstance(obj, t):
        raise_type_err(obj, t)
    return obj

def get_dict_first_item(d: Dict[TK, TV]) -> Tuple[TK, TV]:
    return next(iter(d.items()))

