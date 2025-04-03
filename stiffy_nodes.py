from dataclasses import dataclass
import os
from sys import exception
from typing import Any, ForwardRef, Self, List, Optional, Dict, Tuple, Generic, TypeVar, Union, Set
import yaml
from pydantic import BaseModel
import pprint
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from enum import Enum

ENCODED_PROMPT_TYPE = "ENCODED_PROMPT"
UNCATEGORIZED_CATEGORY_NAME = "uncategorized"
GENERATE_LABELS = False

class CategoryMode(Enum):
    SHALLOW = 1
    DEEP = 2
    ALL = 3


class CategoryList(BaseModel):
    categories: List[str | Dict[str, Self]]

    @classmethod
    def from_yaml(cls, yaml_list: List[str | Dict[str, List[str | Dict]]]) -> Self:
        categories = []
        for item in yaml_list:
            if isinstance(item, str):
                categories.append(item)
            elif isinstance(item, dict):
                k, v = next(iter(item.items()))
                categories.append({k: cls.from_yaml(v)})
            else:
                raise Exception(f"invalid type {type(item)} with value {item}")

        return cls(categories=categories)

    def get_categories(self, mode: CategoryMode ) -> List[str]:
        cats = []
        for cat in self.categories:
            if isinstance(cat, str):
                cats.append(cat)
                continue
            elif not isinstance(cat, dict):
                continue # is not str nor dict, dunno what do
            if mode == CategoryMode.SHALLOW or mode == CategoryMode.ALL:
              cats.append(next(iter(cat.keys())))
            if mode == CategoryMode.DEEP or mode == CategoryMode.ALL:
                cats.extend(next(iter(cat.values())).get_categories(mode))
        return cats
    
    def contains(self, category: str) -> bool:
        trees = self.category_trees()
        for tree in trees:
            if category in tree:
                return True
        return False

    def category_trees(self) -> List[List[str]]:
        trees: List[List[str]] = []
        for cat in self.categories:
            if isinstance(cat, str):
                trees.append([cat])
            elif isinstance(cat, dict):
                k, v = next(iter(cat.items()))
                tree = [k]
                tree.extend([c for subcat in v.category_trees() for c in subcat])
                trees.append(tree)
        return trees
    
    def get_category_tree(self, category: str) -> List[str]:
        for subcat in self.category_trees():
            if category in subcat:
                return subcat
        return []


class Prompt(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    category: Optional[str | List[str]] = UNCATEGORIZED_CATEGORY_NAME

    @classmethod
    def positive(cls, prompt: str, category: str) -> Self:
        return cls(
            prompt=prompt,
            negative_prompt="",
            category=category
        )

    @classmethod
    def negative(cls, prompt: str) -> Self:
        return cls(
            prompt="",
            negative_prompt=prompt,
            category=""
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(
            prompt="",
            negative_prompt="",
            category=""
        )

    @classmethod
    def empty_category(cls, category: str) -> Self:
        return cls(
            prompt="",
            negative_prompt="",
            category=category
        )

    def append(self, other: Self, ignore_categories: bool = False) -> None:
        if not ignore_categories and other.category != self.category:
            raise Exception(f"Cannot merge {self.category} and {other.category}")
        self.prompt = self._merge_prompts(self.prompt, other.prompt)
        self.negative_prompt = self._merge_prompts(self.negative_prompt, other.negative_prompt)

    def _merge_prompts(self, p1: Optional[str], p2: Optional[str]) -> str:
        return ", ".join([p for p in [p1, p2] if p is not None and p != ""])

    def get_encoded(self) -> str:
        return yaml.dump([self.model_dump()])
    
    def get_category(self, available_categories: List[str]) -> str:
        if isinstance(self.category, str):
            return self.category
        elif isinstance(self.category, list):
            for cat in self.category:
                if cat in available_categories:
                    return cat
            return UNCATEGORIZED_CATEGORY_NAME
        else:
            raise Exception(f"{self.category} is not a string nor a list, it is {type(self.category)}")

    def is_in_category(self, category: str) -> bool:
        if isinstance(self.category, str):
            return self.category == category
        elif isinstance(self.category, list):
            for cat in self.category:
                if cat == category:
                    return True 
            return False
        else:
            raise Exception(f"{self.category} is not a string nor a list, it is {type(self.category)}")

    def is_in_category_tree(self, category_tree: List[str]) -> bool:
        if isinstance(self.category, str):
            return self.category in category_tree
        elif isinstance(self.category, list):
            for cat in self.category:
                if cat in category_tree:
                    return True 
            return False
        else:
            raise Exception(f"{self.category} is not a string nor a list, it is {type(self.category)}")


class PromptMgr:
    def __init__(self) -> None:
        self.categories: CategoryList = self.load_category_definitions()
        self.prompts: Dict[str, Prompt] = dict()
        self.negative_prompt: Prompt = Prompt.empty()

        self.styles: Dict[str, Prompt] = dict()

        self.load_category_definitions()

    def get_style_names_from_category(self, category: str, mode: CategoryMode) -> List[str]:
        if(mode == CategoryMode.SHALLOW):
            return self._get_style_names_from_category_tree(category)
        return self._get_style_names_from_category(category)

    def _get_style_names_from_category(self, category: str) -> List[str]:
        names = []
        for name, style in self.styles.items():
            if style.is_in_category(category):
                names.append(name)
                continue
            if category == UNCATEGORIZED_CATEGORY_NAME:
                style_cat = style.get_category(self.categories.get_categories(CategoryMode.DEEP))
                if style_cat is None or style_cat == "" or style_cat == UNCATEGORIZED_CATEGORY_NAME:
                    names.append(name)
                continue

        return names

    def _get_style_names_from_category_tree(self, category: str) -> List[str]:
        names = []
        tree = self.categories.get_category_tree(category)
        for name, style in self.styles.items():
            if style.is_in_category_tree(tree):
                names.append(name)
                continue
            if category == UNCATEGORIZED_CATEGORY_NAME:
                style_cat = style.get_category(self.categories.get_categories(CategoryMode.SHALLOW))
                if style_cat is None or style_cat == "" or style_cat == UNCATEGORIZED_CATEGORY_NAME:
                    names.append(name)
                continue

        return names

    def load_category_definitions(self) -> CategoryList:
        with open(self._get_category_definitions(os.path.dirname(os.path.realpath(__file__)))) as f:
            raw = yaml.safe_load(f)
            c = CategoryList.from_yaml(raw)
            c.categories.append(UNCATEGORIZED_CATEGORY_NAME)
            return c

    def load_styles(self, **kwargs: Dict[str, List[str]]):
        for cat, styles in kwargs.items():
            for s in styles: 
                if s not in self.styles:
                    print(f"{s} is a non-existent style!")
                    continue
                style = self.styles[s]
                self._get_prompt(style.get_category(self.categories.get_categories(CategoryMode.ALL))).append(style)
                self.negative_prompt.append(style, True)

    def load_style_definitions(self):
        for file in self._get_style_definitions(os.path.dirname(os.path.realpath(__file__)) + "/styles/"):
            for k, v in self._load_styles_from_yaml(file).items():
                self.styles[k] = v

    def _load_styles_from_yaml(self, path: str) -> Dict[str, Prompt]:
        with open(path) as f:
            raw = yaml.safe_load(f)

        return {key: Prompt(**value) for key, value in raw.items()}

    def _get_style_definitions(self, dir: str) -> List[str]:
        return self._get_yaml_files(dir, "styles")
    
    def _get_category_definitions(self, dir: str) -> str:
        return self._get_yaml_files(dir, "categories")[0]

    def _get_yaml_files(self, dir: str, file_filter: str) -> List[str]:
        paths = []
        for file in os.listdir(dir):
            if file_filter not in file:
                continue
            full_path = os.path.join(dir, file)
            if os.path.isfile(full_path) and (file.endswith(".yaml") or file.endswith(".yml")):
                paths.append(full_path)
        return paths

    def _get_prompt(self, category: Optional[str]):
        category = category if category is not None and category != "" else UNCATEGORIZED_CATEGORY_NAME
        if not self.categories.contains(category):
            raise Exception(f"{category} not in a list of known categories: {self.categories}")
        if category not in self.prompts:
            print(f"category {category} does not exist, creating")
            self.prompts[category] = Prompt.empty_category(category)
        return self.prompts[category]

    def process_categorized_input(self,mode: CategoryMode, **kwargs):
        for k in kwargs:
            p = Prompt.positive(kwargs[k], k)
            self._get_prompt(p.get_category(self.categories.get_categories(mode))).append(p)

    def process_node_input(self, text: str, mode: CategoryMode):
        y: List[Dict] = yaml.safe_load(text)
        for y_prompt in (y if y is not None else []):
            prompt = Prompt(**y_prompt)
            self._get_prompt(prompt.get_category(self.categories.get_categories(mode))).append(prompt)

    def process_negative_node_input(self, text: str):
        self.negative_prompt.append(Prompt.negative(text))

    def get_encoded_prompt(self) -> str:
        return yaml.dump([p.model_dump() for p in self.prompts.values()])

    def get_clean_prompt(self) -> str:
        parts = []
        for cat in self.categories.get_categories(CategoryMode.ALL):
            print(f"collecting category: {cat}")
            if self._get_prompt(cat).prompt is not None and self._get_prompt(cat).prompt != "":
                print(f"appending {self.prompts[cat]} to prompt")
                parts.append(self._get_prompt(cat).prompt)
        return ",".join(parts)

    def get_clean_negative_prompt(self) -> str:
        return self.negative_prompt.negative_prompt if self.negative_prompt.negative_prompt is not None else ""


def filter_out_labels_from_kwargs(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if not k.startswith("label_")}

### NODES ###


class StiffyPrompterLiteNode:
    CATEGORY_MODE = CategoryMode.SHALLOW
    @classmethod
    def INPUT_TYPES(cls):
        mgr = PromptMgr()
        opts = dict()

        for cat in mgr.categories.get_categories(cls.CATEGORY_MODE):
            if GENERATE_LABELS:
                opts[f"label_{cat}"] = ("STRING", {"default": cat})
            opts[cat] = ("STRING", {"multiline": True, "default": "", "tooltip": cat})

        return {"optional": {
                "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
                **opts,
        }}

    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    RETURN_NAMES = (
        "prompt",
    )
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"

    def get_stiffy(self, prompt: str="", **kwargs):
        mgr = PromptMgr()
        mgr.process_node_input(prompt, self.CATEGORY_MODE)
        mgr.process_categorized_input(self.CATEGORY_MODE, **filter_out_labels_from_kwargs(**kwargs))
        return mgr.get_encoded_prompt(), 

class StiffyPrompterSuperNode(StiffyPrompterLiteNode):
    CATEGORY_MODE = CategoryMode.DEEP

class StiffyPromptStylesLiteNode:
    CATEGORY_MODE = CategoryMode.SHALLOW
    @classmethod
    def INPUT_TYPES(cls):
        mgr = PromptMgr()
        mgr.load_style_definitions()

        opts = dict()
        for cat in mgr.categories.get_categories(cls.CATEGORY_MODE):
            if GENERATE_LABELS:
                opts[f"label_{cat}"] = ("STRING", {"default": cat})
            opts[cat] = ("COMBO", {"options": 
                          mgr.get_style_names_from_category(cat, cls.CATEGORY_MODE),
                          "multi_select": True,
                          "tooltip": cat,
                          "default": cat})

        return {"optional": {
                "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
            "negative_prompt": ("STRING", {"forceInput": True}),
            **opts
        }}

    RETURN_TYPES = (ENCODED_PROMPT_TYPE, "STRING", "STRING")
    RETURN_NAMES = (
        "prompt_encoded",
        "prompt",
        "negative_prompt",
    )
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"

    def get_stiffy(self, prompt="", negative_prompt="", **kwargs):
        mgr = PromptMgr()
        mgr.process_node_input(prompt, self.CATEGORY_MODE)
        mgr.process_negative_node_input(negative_prompt)
        mgr.load_style_definitions()
        mgr.load_styles(**filter_out_labels_from_kwargs(**kwargs))
        return(mgr.get_encoded_prompt(), mgr.get_clean_prompt(), mgr.get_clean_negative_prompt())

class StiffyPromptStylesSuperNode(StiffyPromptStylesLiteNode):
    CATEGORY_MODE = CategoryMode.DEEP

class StiffyPromptDecoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "encoded_prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = (
        "decoded_prompt",
    )
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"

    def get_stiffy(self, encoded_prompt: str):
        mgr = PromptMgr()
        mgr.process_node_input(encoded_prompt, CategoryMode.ALL)
        return (mgr.get_clean_prompt(), )

class StiffyPromptEncoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        mgr = PromptMgr()
        return {
            "required":{
                "prompt": ("STRING", {"forceInput": True}),
                "category": ([cat for cat in mgr.categories.get_categories(CategoryMode.ALL)], {"default": ""})
            }
        }

    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    RETURN_NAMES = (
        "encoded_prompt",
    )
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"

    def get_stiffy(self, prompt: str, category: str):
        return Prompt.positive(prompt, category).get_encoded(), 

class StiffyEncodedPromptToStringNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"

    def get_stiffy(self, prompt: str):
        return prompt,

class StiffyStringToEncodedPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"

    def get_stiffy(self, prompt: str):
        return prompt,

NODE_CLASS_MAPPINGS[StiffyPrompterLiteNode.__name__] = StiffyPrompterLiteNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyPrompterLiteNode.__name__] = "Stiffy Prompter Lite"

NODE_CLASS_MAPPINGS[StiffyPromptStylesLiteNode.__name__] = StiffyPromptStylesLiteNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyPromptStylesLiteNode.__name__] = "Stiffy Prompt Styles Lite"

NODE_CLASS_MAPPINGS[StiffyPrompterSuperNode.__name__] = StiffyPrompterSuperNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyPrompterSuperNode.__name__] = "Stiffy Prompter Super"

NODE_CLASS_MAPPINGS[StiffyPromptStylesSuperNode.__name__] = StiffyPromptStylesSuperNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyPromptStylesSuperNode.__name__] = "Stiffy Prompt Styles Super"

NODE_CLASS_MAPPINGS[StiffyPromptDecoderNode.__name__] = StiffyPromptDecoderNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyPromptDecoderNode.__name__] = "Stiffy Prompt Decoder"

NODE_CLASS_MAPPINGS[StiffyPromptEncoderNode.__name__] = StiffyPromptEncoderNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyPromptEncoderNode.__name__] = "Stiffy Prompt Encoder"

NODE_CLASS_MAPPINGS[StiffyEncodedPromptToStringNode.__name__] = StiffyEncodedPromptToStringNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyEncodedPromptToStringNode.__name__] = "Stiffy Encoded Prompt to String"

NODE_CLASS_MAPPINGS[StiffyStringToEncodedPromptNode.__name__] = StiffyStringToEncodedPromptNode
NODE_DISPLAY_NAME_MAPPINGS[StiffyStringToEncodedPromptNode.__name__] = "Stiffy String to Encoded Prompt"


def INPUT_TYPES(s):
    mgr = PromptMgr()
    mgr.load_style_definitions()

    opts = dict()
    for cat in mgr.categories.get_categories(s):
        if GENERATE_LABELS:
            opts[f"label_{cat}"] = ("STRING", {"default": cat})
        opts[cat] = ("COMBO", {"options": 
                      mgr.get_style_names_from_category(cat, s),
                      "multi_select": True,
                      "tooltip": cat,
                      "default": cat})

    return {"optional": {
            "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
        "negative_prompt": ("STRING", {"forceInput": True}),
        **opts
    }}

if __name__ == "__main__":
    pass


