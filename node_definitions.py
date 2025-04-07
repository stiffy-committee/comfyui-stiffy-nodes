from typing import Tuple, Optional, List

from .utils import Model, Clip
from .constants import ENCODED_PROMPT_TYPE, UNCATEGORIZED_CATEGORY_NAME
from .node_logic import NodeRunner, ALL_CATEGORIES

class StiffyPrompterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional":{
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt_text": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"

    def get_stiffy(self, prompt: str="", negative_prompt: str="", prompt_text: str="", negative_prompt_text: str="") -> Tuple[str, str]:
        return NodeRunner() \
            .process_positive_prompt_input(prompt, False) \
            .process_negative_prompt_input(negative_prompt) \
            .process_positive_prompt_input(prompt_text, False) \
            .process_negative_prompt_input(negative_prompt_text) \
            .get_decoded_prompt()
            
class StiffyPersistentPrompterNode:
    @classmethod
    def INPUT_TYPES(cls):
        runner = NodeRunner()
        return {
            "optional": {
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "category": (runner.get_categories(), {"default": ""}),
                "style_name": ("STRING", {"multiline": False, "default": "", "placeholder": "cust-"}),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt_text": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"

    def get_stiffy(self, prompt: str="", negative_prompt: str="", category: str="", style_name: str="", prompt_text: str="", negative_prompt_text: str="") -> Tuple[str, str]:
        return NodeRunner() \
            .process_positive_prompt_input(prompt, False) \
            .process_negative_prompt_input(negative_prompt) \
            .process_positive_prompt_input(prompt_text, False) \
            .process_negative_prompt_input(negative_prompt_text) \
            .save_style(style_name, category, prompt, negative_prompt) \
            .get_decoded_prompt()

class StiffyCategorizedPrompterNode:
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
            .process_positive_prompt_input(prompt, True) \
            .process_positive_prompt_input(positive_prompt, False, category) \
            .get_encoded_prompt(),
            
class StiffyCategorizedPersistentPrompterNode:
    @classmethod
    def INPUT_TYPES(cls):
        runner = NodeRunner()
        return {
            "optional": {
                "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
                "category": (runner.get_categories(), {"default": ""}),
                "style_name": ("STRING", {"multiline": False, "default": "", "placeholder": "cust-"}),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    RETURN_NAMES = ("prompt", )
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"

    def get_stiffy(self, prompt: str="", category: str="", style_name: str="", positive_prompt: str="", negative_prompt: str="") -> Tuple[str]:
        return NodeRunner() \
            .process_positive_prompt_input(prompt, True) \
            .process_positive_prompt_input(positive_prompt, False, category) \
            .process_negative_prompt_input(negative_prompt) \
            .save_style(style_name, category, positive_prompt, negative_prompt) \
            .get_encoded_prompt(),

class StiffyStylerBase:
    STYLE_CATEGORY: str = UNCATEGORIZED_CATEGORY_NAME
    @classmethod
    def INPUT_TYPES(cls):
        runner = NodeRunner()
        return {"optional":{
            "prompt": (ENCODED_PROMPT_TYPE, {"forceInput": True}),
            "styles": ("COMBO", {
                "options": runner.get_styles(cls.STYLE_CATEGORY),
                "multi_select": True,
                "tooltip": cls.STYLE_CATEGORY
            })
        }}
    
    RETURN_TYPES = (ENCODED_PROMPT_TYPE,)
    RETURN_NAMES = ("prompt", )
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"
    def get_stiffy(self, prompt: str="", styles: Optional[List[str]] = None) -> Tuple[str]:
        return NodeRunner() \
            .process_positive_prompt_input(prompt, self.STYLE_CATEGORY != "", self.STYLE_CATEGORY) \
            .apply_styles(styles, self.STYLE_CATEGORY != "") \
            .get_encoded_prompt(),

class StiffyStyler:
    @classmethod
    def INPUT_TYPES(cls):
        runner = NodeRunner()
        return {"optional":{
            "prompt": ("STRING", {"forceInput": True}),
            "negative_prompt": ("STRING", {"forceInput": True}),
            "styles": ("COMBO", {
                "options": runner.get_styles(ALL_CATEGORIES),
                "multi_select": True,
                "tooltip": ALL_CATEGORIES
            })
        }}
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt" )
    CATEGORY = "stiffy"
    FUNCTION = "get_stiffy"
    def get_stiffy(self, prompt: str="", negative_prompt: str="", styles: Optional[List[str]] = None) -> Tuple[str, str]:
        return NodeRunner() \
            .process_positive_prompt_input(prompt, False) \
            .process_negative_prompt_input(negative_prompt) \
            .apply_styles(styles, False) \
            .get_decoded_prompt()


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
                .process_positive_prompt_input(encoded_prompt, True) \
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

class StiffyLoraLoaderNode:
    @classmethod
    def INPUT_TYPES(cls): 
        runner = NodeRunner()
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRAs will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRAs will be applied to."}),
                "lora_name": ("COMBO", {
                    "options": runner.get_lora_names(),
                    "multi_select": True,
                    "tooltip": "The name of the LoRA.", 
                }),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "get_stiffy"

    CATEGORY = "stiffy"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def get_stiffy(self, model: Model, clip: Clip, lora_names: List[str], strength_model: float, strength_clip: float):
        NodeRunner().apply_loras(model, clip, lora_names, strength_model, strength_clip)
