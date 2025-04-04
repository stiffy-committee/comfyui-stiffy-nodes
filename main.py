from typing import Self, Type
import re 
from .node_definitions import *
from .utils import T
from .node_logic import NodeRunner
from .constants import ALL_CATEGORIES
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

class NodeFactory:
    def __init__(self) -> None:
        pass

    def register_node(self, node: Type[T]) -> Self:
        name = node.__name__
        display_name = re.sub(r"(?<!^)(?=[A-Z])|Node", " ", name)
        NODE_CLASS_MAPPINGS[name] = node
        NODE_DISPLAY_NAME_MAPPINGS[name] = display_name
        return self

    def register_styler_nodes(self) -> Self:
        runner = NodeRunner()
        for category in [ALL_CATEGORIES, *runner.get_categories()]:
            cat_name = "All" if category == ALL_CATEGORIES else category.capitalize()
            node_name = f"StiffyStyler{cat_name}Node"
            node = type(node_name, (StiffyStylerBase, ), {"STYLE_CATEGORY": category})
            self.register_node(node)
        return self

NodeFactory() \
    .register_node(StiffyPrompterNode) \
    .register_node(StiffyPersistentPrompterNode) \
    .register_node(StiffyDecoderNode) \
    .register_node(StiffyDebuggerNode) \
    .register_styler_nodes() 

if __name__ == "__main__":
    pass

