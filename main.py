from pprint import pprint
from typing import Self, Type
import re
import inspect


from threadpoolctl import register 
from .utils import T
from .node_logic import NodeRunner
from .constants import ALL_CATEGORIES
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import node_definitions

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
        for category in ["", ALL_CATEGORIES, *runner.get_categories()]:
            cat_name = "All" if category == ALL_CATEGORIES else category.capitalize()
            node_name = f"StiffyStyler{cat_name}Node"
            node = type(node_name, (node_definitions.StiffyStylerBase, ), {"STYLE_CATEGORY": category})
            self.register_node(node)
        return self
    
    def dynamically_register_nodes(self) -> Self:
        for name, obj in vars(node_definitions).items():
            try:
                if inspect.isclass(obj) and \
                    obj.__module__ == node_definitions.__name__ and \
                    name != node_definitions.StiffyStylerBase.__name__:
                    self.register_node(obj)
                    print(f"registered: {name}")
                else:
                    print(f"did not register: {name}")
            except Exception as e:
                print(e)
                break
        return self

NodeFactory() \
    .dynamically_register_nodes() \
    .register_styler_nodes() 



