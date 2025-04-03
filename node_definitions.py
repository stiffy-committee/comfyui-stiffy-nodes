from typing import Tuple, Optional, List
from constants import ENCODED_PROMPT_TYPE
from node_logic import NodeRunner
from models import Category

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
