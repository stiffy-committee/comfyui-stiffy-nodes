from decimal import InvalidOperation
from functools import reduce
import os
from pathlib import Path
from typing import List, Dict, Optional, Self, TextIO, Tuple, Type, Generator
from constants import ALL_CATEGORIES
from models import *
import yaml

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
        self.style_list: StyleList = reduce(lambda acc, s: StyleList(styles= acc.styles | s.styles), self._load(StyleList, True), StyleList(styles={}))

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
    
    def get_styles(self, category: Category) -> List[str]:
        print(category)
        if category.name == ALL_CATEGORIES:
            return [k for k in self.prompt_controller.style_list.styles.keys()]
        for k, v in self.prompt_controller.style_list.styles.items():
            print(f"{k}: {v.category.name}({type(v.category.name)}) == {category}({type(category)}) => {v.category.name == category.name}")
        return [k for k, v in self.prompt_controller.style_list.styles.items() if v.category.name == category.name]
