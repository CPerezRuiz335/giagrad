# Configuration file for the Sphinx documentation builder.
import os
import sys
# Need this so sphinx can find lumache.py. Change is .py files are elsewhere than root.
sys.path.insert(0, os.path.abspath('../../giagrad'))

# -- Project information
project = 'giagrad'
copyright = '2023, Carlos Perez'
author = 'Carlos Perez'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions =  [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../../giagrad']


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_context = {
    "display_github": True, # Integrate GitHub
    "github_repo": "CPerezRuiz335/giagrad", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "docs/source/", # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------

html_short_title = "topobathy"
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
repository_url = f"https://github.com/CPerezRuiz335/giagrad"
html_context = {
    "menu_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            repository_url,
        ),
        (
            '<i class="fa fa-book fa-fw"></i> License',
            f"{repository_url}/blob/main/LICENSE",
        ),
    ],
}