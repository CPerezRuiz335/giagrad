## Configuration file for the Sphinx documentation builder.
import os
import sys
# Need this so sphinx can find lumache.py. Change is .py files are elsewhere than root.
sys.path.insert(0, os.path.abspath('../../giagrad'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'giagrad'
copyright = '2023, Carlos Perez'
author = 'Carlos Perez'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.intersphinx',
    'autoapi.extension',
    ]

autoapi_type = 'python'
autoapi_dirs = ['../../giagrad']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
