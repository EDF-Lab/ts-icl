# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

tsicl_path = str(Path.cwd().parent / "src/tsicl")

print(tsicl_path)

project = 'TS-ICL'
copyright = f"{datetime.now().year}, EDF"
author = 'Etienne Le Naour (EDF), Tahar Nabil (EDF), Adrien Petralia (EDF)'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_design",
    "alabaster",
    "myst_parser",
    "sphinx_design",
    "nbsphinx",          # renders .ipynb files
    "nbsphinx_link",     # allows notebooks outside source root
    "sphinxcontrib.lightbox2",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# language = "fr"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo-tsicl.jpeg"
html_favicon = "_static/logo-tsicl.jpeg"

autoapi_type = "python"
autoapi_dirs = [tsicl_path]
autoapi_ignore = [
    f"{tsicl_path}/model/*",
    "*__about__*",
]

autoapi_root = "autoapi"
autoapi_member_order = "groupwise"
autoapi_python_class_content = "class"
autoapi_own_page_level = "function"

autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_generate_api_docs = True

autoapi_options = [
    "members",  # Public members
    "undoc-members",  # Undocumented members
    # "private-members",    # Private members
    "special-members",  # Special members (e.g., __init__)
    "inherited-members",  # Inherited members (e.g., parent class)
    "imported_members",  # Imported members
    "show-module-summary",  # Module summary at the top of page
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

html_theme_options = {
    "header_links_before_dropdown": 6,
    "secondary_sidebar_items": {
        "**": ["page-toc"],
        "index": [],
    },
    # "navbar_align": "left"
}

html_css_files = ["custom.css"]

html_sidebars = {"**": [], "autoapi/index": ["sidebar-nav-bs"]}

pygments_style = "sphinx"