import os
from pathlib import Path

ENCODED_PROMPT_TYPE = "ENCODED_PROMPT"
UNCATEGORIZED_CATEGORY_NAME = "uncategorized"
TEMPLATE_STYLE_NAME = "template"
ALL_CATEGORIES = "*"

ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
CUST_STYLES_PATH = ROOT_DIR.joinpath("styles").joinpath("cust-styles.yaml")
