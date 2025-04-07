from collections.abc import Hashable
from types import UnionType
from typing import Any, Dict, Generic, NoReturn, Optional, Tuple, TypeVar, Type, Self

T = TypeVar("T")
TK = TypeVar("TK", bound=Hashable)
TV = TypeVar("TV")

class Model:
    pass

class Lora:
    pass

class Clip:
    pass

def raise_type_err(obj: Any, t: Type[T] | UnionType) -> NoReturn:
    raise TypeError(f"Expected {t} but got {type(obj).__name__} with value: {obj}")

def assert_type(obj: Any, t: Type[T]) -> T:
    if not isinstance(obj, t):
        raise_type_err(obj, t)
    return obj

def get_dict_first_item(d: Dict[TK, TV]) -> Tuple[TK, TV]:
    return next(iter(d.items()))

class Cache(Generic[TK, TV]):
    _cache: Dict[TK, TV] = dict()

    def __init__(self) -> None:
        pass

    def get(self, key: TK, fallback: Optional[TV] = None) -> Optional[TV]:
        return self._cache.get(key, fallback)
    
    def set(self, key: TK, value: TV) -> Self:
        self._cache[key] = value
        return self

    def remove(self, key: TK) -> Self:
        if key in self._cache.keys():
            del(self._cache[key])
        return self
    
    def contains(self, key: TK) -> bool:
        return key in self._cache.keys()


