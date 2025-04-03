from abc import abstractmethod
from collections import defaultdict
from decimal import InvalidOperation
from functools import reduce
import os
from pathlib import Path
from pprint import pprint
import re
from types import FunctionType, NotImplementedType, UnionType
from weakref import proxy
from pydantic import BaseModel, NegativeFloat
from typing import Any, List, Dict, NoReturn, Optional, Self, TextIO, Tuple, TypeVar, Type, Generator, Union

from pydantic_core.core_schema import is_instance_schema
import yaml

ENCODED_PROMPT_TYPE = "ENCODED_PROMPT"
UNCATEGORIZED_CATEGORY_NAME = "uncategorized"

# UTILS
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")

def raise_type_err(obj: Any, t: Type[T] | UnionType) -> NoReturn:
    raise TypeError(f"Expected {t} but got {type(obj).__name__} with value: {obj}")

def assert_type(obj: Any, t: Type[T]) -> T:
    if not isinstance(obj, t):
        raise_type_err(obj, t)
    return obj

def get_dict_first_item(d: Dict[TK, TV]) -> Tuple[TK, TV]:
    return next(iter(d.items()))

# DATACLASSES
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
        return cls(styles={k: Prompt.from_yaml(v) for k, v in assert_type(raw, Dict).items()})

    @classmethod
    def collection_from_yaml(cls, raw: List | Dict | str) -> List[Self]:
        raise NotImplementedType

# CONTROLLERS

class YamlLoader:
    def __init__(self) -> None:
        self.search_path: Path = Path(os.path.dirname(os.path.realpath(__file__))) # dir of this script
        self.file_filter: str = "*.yaml"

    def _load(self, model: Type[TModel], recursive: bool = False) -> List[TModel]: 
        models = []
        for file in self._get_files(recursive):
            with open(file) as f:
                if (m := self._load_yaml(model, f)) is not None:
                    models.append(m)
        return models

    def _load_yaml(self, model: Type[TModel], yml: str | TextIO) -> Optional[TModel]:
        raw = yaml.safe_load(yml)
        return model.from_yaml(raw) if raw is not None else None

    def _load_yaml_collection(self, model: Type[TModel], yml: str | TextIO) -> Optional[List[TModel]]:
        raw = yaml.safe_load(yml)
        return model.collection_from_yaml(raw) if raw is not None else None

    def _get_files(self, recursive: bool = False) -> Generator[Path, None, None]:
        pattern = f"**/{self.file_filter}" if recursive else self.file_filter
        return self.search_path.glob(pattern)
        

class CategoryController(YamlLoader):
    def __init__(self) -> None:
        super().__init__()
        self.file_filter = "categories.yaml"
        self.category_list: CategoryList = self._load(CategoryList)[0] # TODO: maybe fix?
        self.category_list.categories.append(Category.uncategorized())

    def get_categories(self) -> List[Category]:
        return self.category_list.categories

    def find_category(self, name: str) -> Optional[Category]:
        for category in self.get_categories():
            if category.name == name:
                return category
        return None

                
class PromptController(YamlLoader):
    def __init__(self, category_controller: CategoryController) -> None:
        super().__init__()
        self.file_filter = "*style*.yaml"
        self.search_path = self.search_path.joinpath("styles")

        self.category_controller = category_controller
        self.style_list: StyleList = reduce(lambda acc, s: StyleList(styles= acc.styles | s.styles), self._load(StyleList), StyleList(styles={}))

        self.prompts: Dict[Category, Prompt] = {cat: Prompt.empty(cat) for cat in self.category_controller.get_categories()}

    def _get_actual_prompt_category(self, prompt: Prompt, categories: List[Category]) -> Category:
        if prompt.category in categories:
            return prompt.category
        return Category.uncategorized()
    
    def _prompt_str_join(self, s1: Optional[str], s2: Optional[str]) -> str:
        return ", ".join(s.strip(" ,") for s in [s1, s2] if s is not None and s != "")

    def merge_prompts(self, p1: Prompt, p2: Prompt) -> Prompt:
        categories = self.category_controller.get_categories()
        c1, c2 = self._get_actual_prompt_category(p1, categories), self._get_actual_prompt_category(p2, categories)
        if c1 != c2:
            raise InvalidOperation(f"Cannot merge prompts of category {c1.name} and {c2.name}")
        
        return Prompt(
            prompt=self._prompt_str_join(p1.prompt, p2.prompt),
            negative_prompt=self._prompt_str_join(p1.negative_prompt, p2.negative_prompt),
            category = c1)

    def load_encoded_prompt(self, prompt: str):
        decoded = self._load_yaml_collection(Prompt, prompt)
        for d in decoded if decoded is not None else []:
            if d is not None:
                self.load_prompt(d)
    
    def load_prompt(self, prompt: Prompt):
        self.prompts[prompt.category] = self.merge_prompts(self.prompts[prompt.category], prompt)

    def export_encoded(self) -> str:
        return yaml.dump([p.model_dump() for p in self.prompts.values()])
    
    def export_decoded(self) -> Tuple[str, str]:
        final_prompt = Prompt.empty(Category.uncategorized())
        for category in self.category_controller.category_list.categories:
            prompt = self.prompts[category] 
            final_prompt.prompt = self._prompt_str_join(final_prompt.prompt, prompt.prompt)
            final_prompt.negative_prompt = self._prompt_str_join(final_prompt.negative_prompt, prompt.negative_prompt)
        return (final_prompt.prompt if not final_prompt.prompt is None else "",
        final_prompt.negative_prompt  if not final_prompt.negative_prompt is None else "")
    
    def apply_styles(self, styles: List[str]):
        for style in styles:
            style_prompt = self.style_list.styles[style]
            if style_prompt is None:
                raise InvalidOperation(f"{style} is not a valid style")
            
            self.prompts[style_prompt.category] = self.merge_prompts(self.prompts[style_prompt.category], style_prompt)

        
# NODE RUNNER

class NodeRunner:
    def __init__(self) -> None:
        self.category_controller: CategoryController = CategoryController()
        self.prompt_controller: PromptController = PromptController(self.category_controller)

    def _get_category(self, category: str) -> Category:
        cat = self.category_controller.find_category(category)
        if cat is None:
            print(f"{category} was not found, defaulting to {UNCATEGORIZED_CATEGORY_NAME}")
            cat = self.category_controller.find_category(UNCATEGORIZED_CATEGORY_NAME)
            if cat is None:
                raise Exception(f"{UNCATEGORIZED_CATEGORY_NAME} category does not exist????")
        return cat

    def process_encoded_prompt_input(self, prompt: str) -> Self:
        self.prompt_controller.load_encoded_prompt(prompt)
        return self

    def process_positive_prompt_input(self, prompt: str, category: str) -> Self:
        self.prompt_controller.load_prompt(Prompt(
            prompt=prompt,
            negative_prompt="",
            category=self._get_category(category)
        ))
        return self;

    def process_negative_prompt_input(self, prompt: str) -> Self:
        self.prompt_controller.load_prompt(Prompt(
            prompt="",
            negative_prompt=prompt,
            category=Category.uncategorized()
        ))
        return self

    def apply_styles(self, styles: Optional[List[str]]) -> Self:
        if styles is not None:
            self.prompt_controller.apply_styles(styles)
        return self

    def get_categories(self) -> List[str]:
        return [c.name for c in self.category_controller.get_categories()]

    def get_encoded_prompt(self) -> str:
        return self.prompt_controller.export_encoded()
    
    def get_decoded_prompt(self) -> Tuple[str, str]:
        return self.prompt_controller.export_decoded()
    
    def get_styles(self, category: Optional[Category]) -> List[str]:
        print(category)
        if category is None:
            return [k for k in self.prompt_controller.style_list.styles.keys()]
        for k, v in self.prompt_controller.style_list.styles.items():
            print(f"{k}: {v.category.name}({type(v.category.name)}) == {category}({type(category)}) => {v.category.name == category.name}")
        return [k for k, v in self.prompt_controller.style_list.styles.items() if v.category.name == category.name]


# NODES

class StiffyPrompterNode:
    @classmethod
    def INPUT_TYPES(cls):
        runner = NodeRunner()
        return {
            "optional":{
                "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
                "category": (runner.get_categories(), {"default": ""}),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    RETURN_NAMES = ("prompt", )
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"
    def get_stiffy(self, prompt: str="", category: str="", positive_prompt: str="") -> Tuple[str]:
        return NodeRunner() \
            .process_encoded_prompt_input(prompt) \
            .process_positive_prompt_input(positive_prompt, category) \
            .get_encoded_prompt(),
            

class StiffyStylerBase:
    STYLE_CATEGORY: Category = Category.uncategorized()
    @classmethod
    def INPUT_TYPES(cls):
        runner = NodeRunner()
        return {"optional":{
            "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
            "style": ("COMBO", {
                "options": runner.get_styles(cls.STYLE_CATEGORY),
                "multi_select": True,
            })
        }}
    
    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    RETURN_NAMES = ("prompt", )
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"
    def get_stiffy(self, prompt: str="", styles: Optional[List[str]] = None) -> Tuple[str]:
        return NodeRunner() \
            .process_encoded_prompt_input(prompt) \
            .apply_styles(styles) \
            .get_encoded_prompt(),


class StiffyDecoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "encoded_prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = (
        "prompt",
        "negative_prompt"
    )
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"

    def get_stiffy(self, encoded_prompt: str, negative_prompt: str="") -> Tuple[str, str]:
        return NodeRunner() \
                .process_encoded_prompt_input(encoded_prompt) \
                .process_negative_prompt_input(negative_prompt) \
                .get_decoded_prompt()

class StiffyDebuggerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "encoded_prompt": (ENCODED_PROMPT_TYPE, {"multiline": True}),
                "plain_encoded_prompt": ("STRING", {"multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING", ENCODED_PROMPT_TYPE)
    RETURN_NAMES = ("plain_encoded_prompt", "encoded_prompt")
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"
    def get_stiffy(self, encoded_prompt: str="", plain_encoded_prompt: str="") -> Tuple[str, str]:
        return encoded_prompt, plain_encoded_prompt

# NODE FACTORY

class NodeFactory:
    def __init__(self) -> None:
        pass

    def register_node(self, node: Type[T]) -> Self:
        name = node.__name__
        display_name = re.sub(r"(?<!^)(?=[A-Z])|Node", " ", name)
        #NODE_CLASS_MAPPINGS[name] = node
        #NODE_DISPLAY_NAME_MAPPINGS[name] = display_name
        return self

    def register_prompter_node(self) -> Self:
        return self.register_node(StiffyPrompterNode)

    def register_decoder_node(self) -> Self:
        return self.register_node(StiffyDecoderNode)

    def register_styler_nodes(self) -> Tuple[Self, Any]:
        runner = NodeRunner()
        for category in [None, *runner.get_categories()]:
            cat_name = "All" if category is None else category.capitalize()
            node_name = f"StiffyStyler{cat_name}Node"
            node = type(node_name, (StiffyStylerBase, ), {"STYLE_CATEGORY": category})
            print(category)
            pprint(node.INPUT_TYPES())
            self.register_node(node)
        return self, n

if __name__ == "__main__":
    n = StiffyPrompterNode()
    n_out = n.get_stiffy(category="subject", positive_prompt="yoooo")[0]
    n2 = StiffyPrompterNode()
    n2_out = n2.get_stiffy(n_out, "clothing", "fucky")[0]

    _, n3 = NodeFactory().register_styler_nodes()

    #n3 = StiffyDecoderNode()
    #n3_out = n3.get_stiffy(n2_out)


