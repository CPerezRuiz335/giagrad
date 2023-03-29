## Configuration file for the Sphinx documentation builder.
import os
import sys
# Need this so sphinx can find source files. Change is .py files are elsewhere than root.
sys.path.insert(0, os.path.abspath('../../giagrad'))

def linkcode_resolve(domain, info, linkcode_url=None):
    import os
    import sys
    import inspect
    import pkg_resources

    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    topmodulename = modname.split(".")[0]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    if linkcode_url is None:
        linkcode_url = (
            f"https://github.com/inducer/{project}/blob/"
            + "main"
            + "/{filepath}#L{linestart}-L{linestop}"
        )

    return linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop
    )

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'giagrad'
copyright = '2023, Carlos Pérez'
author = 'Carlos Pérez'
# release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.duration',
        "sphinx.ext.autodoc",
        "sphinx.ext.intersphinx",
        "sphinx.ext.linkcode",
        "sphinx.ext.doctest",
        "sphinx.ext.mathjax",
        "sphinx_copybutton",
        'sphinx.ext.githubpages',
        "sphinx.ext.napoleon",
        'sphinx.ext.autosummary',
        ]

autoapi_type = 'python'
autoapi_dirs = ['../../giagrad']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_show_sourcelink = True
autoclass_content = "class"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_context = {
    "display_github": True, # Integrate GitHub
    "github_repo": "CPerezRuiz335/giagrad", # Repo name
    "github_version": "main", # Version
    "conf_py_path": ".docs/", # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True