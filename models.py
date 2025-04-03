from abc import abstractmethod
from types import NotImplementedType
from pydantic import BaseModel
from typing import List, Dict, Optional, Self, TypeVar, Union, Self
from utils import assert_type, get_dict_first_item, raise_type_err
from constants import UNCATEGORIZED_CATEGORY_NAME, TEMPLATE_STYLE_NAME

class Model(BaseModel):
    @classmethod
    @abstractmethod
    def from_yaml(cls, raw: List | Dict | str) -> Self:
        raise NotImplementedType

    @classmethod
    @abstractmethod
    def collection_from_yaml(cls, raw: List | Dict) -> List[Self]:
        raise NotImplementedType

TModel = TypeVar("TModel", bound=Model)

class Category(Model):
    name: str

    @classmethod
    def from_yaml(cls, raw: List | Dict | str) -> Self:
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, Dict):
            return cls(**raw)
        raise_type_err(raw, Union[Dict, str])
         
    
    @classmethod
    def collection_from_yaml(cls, raw: List | Dict | str) -> List[Self]:
        raise NotImplementedType

    @classmethod
    def uncategorized(cls) -> Self:
        return cls(name=UNCATEGORIZED_CATEGORY_NAME)

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Category) and self.name == other.name

class CategoryList(Model):
    categories: List[Category]

    @classmethod
    def from_yaml(cls, raw: List | Dict | str) -> Self:
        categories = []
        for item in assert_type(raw, List):
            if isinstance(item, str):
                categories.append(Category.from_yaml(item))
            elif isinstance(item, dict):
                _, v = get_dict_first_item(item)
                categories.extend([c for c in cls.from_yaml(v).categories])
            else:
                raise TypeError(f"Invalid type {type(item).__name__} with value {item}")
        return cls(categories=categories)

    @classmethod
    def collection_from_yaml(cls, raw: List | Dict | str) -> List[Self]:
        raise NotImplementedType

class Prompt(Model):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    category: Category = Category.uncategorized()

    @classmethod
    def from_yaml(cls, raw: List | Dict | str) -> Self:
        raw = assert_type(raw, Dict)
        raw["category"] = Category.from_yaml(raw.get("category", UNCATEGORIZED_CATEGORY_NAME))
        return cls(**assert_type(raw, Dict))

    @classmethod
    def collection_from_yaml(cls, raw: List | Dict | str) -> List[Self]:
        if not isinstance(raw, List):
            raw = [raw]
        prompts = []
        for i in raw:
            i = assert_type(i, Dict)
            i["category"] = Category(**i.get("category", UNCATEGORIZED_CATEGORY_NAME))
            prompts.append(cls(**assert_type(i, Dict)))
        return prompts
    
    @classmethod
    def empty(cls, category: Category) -> Self:
        return cls(
            prompt="",
            negative_prompt="",
            category=category
        )

class StyleList(Model):
    styles: Dict[str, Prompt]
    
    @classmethod
    def from_yaml(cls, raw: List | Dict | str) -> Self:
        return cls(styles={k: Prompt.from_yaml(v) for k, v in assert_type(raw, Dict).items() if k != TEMPLATE_STYLE_NAME})

    @classmethod
    def collection_from_yaml(cls, raw: List | Dict | str) -> List[Self]:
        raise NotImplementedType

