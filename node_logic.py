from dataclasses import dataclass
from datetime import datetime
from decimal import InvalidOperation
from functools import reduce
import os
from pathlib import Path
from typing import Generic, List, Dict, Optional, Self, TextIO, Tuple, Type, Generator

from .utils import TK, TV
from .constants import ALL_CATEGORIES, CUST_STYLES_PATH, ROOT_DIR
from .models import *
import yaml

@dataclass
class CachedFile: 
    time_modified: float
    contents: str

class FileCache:
    _items: Dict[Path, CachedFile] = dict()

    def __init__(self, path: Path) -> None:
        self.path = path
        pass

    def read(self) -> str:
        m_time = self.path.stat().st_mtime

        if self.path in self._items.keys() and \
            self._items[self.path].time_modified == m_time:
            #print(f"{self.path} cache hit")
            return self._items[self.path].contents
        
        print(f"{self.path} cache miss")
        with open(self.path) as f:
            cf = CachedFile(contents=f.read(), time_modified=m_time)
        self._items[self.path] = cf
        return cf.contents

    def write(self, contents: str) -> None:
        with open(self.path, "w") as f:
            f.write(contents)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @classmethod
    def open(cls, path: Path, create_if_does_not_exist: bool = False) -> Self:
        if not path.exists():
            if create_if_does_not_exist:
                path.touch()
            else:
                raise InvalidOperation(f"{path} does not exist!")

        if not path.is_file():
            raise InvalidOperation(f"{path} is not a file!")
        return cls(path)

    @classmethod
    def clear(cls):
        cls._items.clear()


class YamlLoader:
    def __init__(self) -> None:
        self.search_path: Path = ROOT_DIR
        self.file_filter: str = "*.yaml"

    def _load(self, model: Type[TModel], recursive: bool = False) -> List[TModel]: 
        models = []
        for file in self._get_yaml_files(recursive):
            m = self._load_yaml_file(model, file)
            if m is not None:
                models.append(m)
        return models

    def _load_yaml_file(self, model: Type[TModel], path: Path) -> Optional[TModel]:
        with FileCache.open(path) as f:
            if (m := self._dict_from_yaml(model, f.read())) is not None:
                return m

    def _dict_from_yaml(self, model: Type[TModel], yml: str) -> Optional[TModel]:
        raw = yaml.safe_load(yml)
        return model.from_yaml(raw) if raw is not None else None

    def _list_from_yaml(self, model: Type[TModel], yml: str | TextIO) -> Optional[List[TModel]]:
        raw = yaml.safe_load(yml)
        return model.collection_from_yaml(raw) if raw is not None else None

    def _get_yaml_files(self, recursive: bool = False) -> Generator[Path, None, None]:
        pattern = f"**/{self.file_filter}" if recursive else self.file_filter
        return self.search_path.glob(pattern)


class CategoryController(YamlLoader):
    def __init__(self) -> None:
        super().__init__()
        self.file_filter = "categories.yaml"
        self.category_list: CategoryList = self._load(CategoryList)[0] # TODO: maybe fix?
        self.category_list.categories.append(UNCATEGORIZED_CATEGORY_NAME)

    def get_categories(self) -> List[str]:
        return self.category_list.categories

    def find_category(self, name: str) -> Optional[str]:
        for category in self.get_categories():
            if category == name:
                return category
        return None

                
class PromptController(YamlLoader):
    def __init__(self, category_controller: CategoryController) -> None:
        super().__init__()
        self.file_filter = "*style*.yaml"
        self.search_path = self.search_path.joinpath("styles")

        self.category_controller = category_controller
        self.style_list: StyleList = reduce(lambda acc, s: StyleList(styles= acc.styles | s.styles), self._load(StyleList, True), StyleList(styles={}))

        self.prompts: Dict[str, Prompt] = {cat: Prompt.empty(cat) for cat in self.category_controller.get_categories()}

    def _get_actual_prompt_category(self, prompt: Prompt, categories: List[str]) -> str:
        if prompt.category in categories:
            return prompt.category
        return UNCATEGORIZED_CATEGORY_NAME
    
    def _prompt_str_join(self, s1: Optional[str], s2: Optional[str]) -> str:
        return ", ".join(s.strip(" ,") for s in [s1, s2] if s is not None and s != "")

    def merge_prompts(self, p1: Prompt, p2: Prompt) -> Prompt:
        categories = self.category_controller.get_categories()
        c1, c2 = self._get_actual_prompt_category(p1, categories), self._get_actual_prompt_category(p2, categories)
        if c1 != c2:
            raise InvalidOperation(f"Cannot merge prompts of category {c1} and {c2}")
        
        return Prompt(
            prompt=self._prompt_str_join(p1.prompt, p2.prompt),
            negative_prompt=self._prompt_str_join(p1.negative_prompt, p2.negative_prompt),
            category = c1)

    def load_encoded_prompt(self, prompt: str):
        decoded = self._list_from_yaml(Prompt, prompt)
        for d in decoded if decoded is not None else []:
            if d is not None:
                self.load_prompt(d)
    
    def load_prompt(self, prompt: Prompt):
        self.prompts[prompt.category] = self.merge_prompts(self.prompts[prompt.category], prompt)

    def export_encoded(self) -> str:
        return yaml.dump([p.model_dump() for p in self.prompts.values()])
    
    def export_decoded(self) -> Tuple[str, str]:
        final_prompt = Prompt.empty(UNCATEGORIZED_CATEGORY_NAME)
        for category in self.category_controller.category_list.categories:
            prompt = self.prompts[category] 
            final_prompt.prompt = self._prompt_str_join(final_prompt.prompt, prompt.prompt)
            final_prompt.negative_prompt = self._prompt_str_join(final_prompt.negative_prompt, prompt.negative_prompt)
        return (final_prompt.prompt if not final_prompt.prompt is None else "",
        final_prompt.negative_prompt  if not final_prompt.negative_prompt is None else "")
    
    def apply_styles(self, styles: List[str]):
        for style in styles:
            style_prompt = self.style_list.styles.get(style, None)
            if style_prompt is None:
                raise Exception(f"{style} is not a valid style, probably a custom style that has been removed,\n"+ \
                "in which case, unselect the style and refresh the page")
            
            self.prompts[style_prompt.category] = self.merge_prompts(self.prompts[style_prompt.category], style_prompt)

    def _load_custom_style_list(self) -> StyleList:
        sl: Optional[StyleList] = None
        if CUST_STYLES_PATH.exists():
            sl = self._load_yaml_file(StyleList, CUST_STYLES_PATH)
        if sl is None:
            sl = StyleList(styles=dict())
        return sl

    def _store_custom_style_list(self, sl: StyleList):
        with FileCache.open(CUST_STYLES_PATH, True) as f:
            f.write(yaml.dump({k: v.model_dump() for k, v in sl.styles.items()}))

    def save_custom_style(self, name: str, prompt: Prompt):
        sl = self._load_custom_style_list()
        if name in self.style_list.styles.keys() and name not in sl.styles.keys():
            raise Exception(f"style {name} is already defined somewhere else than {CUST_STYLES_PATH}!")

        sl.styles[name] = prompt
        self._store_custom_style_list(sl)

    def remove_custom_style(self, name: str):
        sl = self._load_custom_style_list()

        if name not in sl.styles.keys():
            return # nothing to do

        del(sl.styles[name])
        self._store_custom_style_list(sl)


class NodeRunner:
    def __init__(self) -> None:
        self.category_controller: CategoryController = CategoryController()
        self.prompt_controller: PromptController = PromptController(self.category_controller)

    def _get_category(self, category: str) -> str:
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
            category=UNCATEGORIZED_CATEGORY_NAME
        ))
        return self

    def apply_styles(self, styles: Optional[List[str]]) -> Self:
        if styles is not None:
            self.prompt_controller.apply_styles(styles)
        return self

    def save_style(self, name: str, category: str, positive_prompt: str, negative_prompt: str) -> Self:
        if name != "":
            if positive_prompt == "" and negative_prompt == "":
                self.prompt_controller.remove_custom_style(name)
            else:
                self.prompt_controller.save_custom_style(name, Prompt(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    category=category
                ))
        return self

    def get_categories(self) -> List[str]:
        return [c for c in self.category_controller.get_categories()]

    def get_encoded_prompt(self) -> str:
        return self.prompt_controller.export_encoded()
    
    def get_decoded_prompt(self) -> Tuple[str, str]:
        return self.prompt_controller.export_decoded()
    
    def get_styles(self, category: str) -> List[str]:
        if category == ALL_CATEGORIES:
            return [k for k in self.prompt_controller.style_list.styles.keys()]
        return [k for k, v in self.prompt_controller.style_list.styles.items() if v.category == category]
