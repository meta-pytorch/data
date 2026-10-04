# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme2
import torchdata

# sys.path.insert(0, os.path.abspath('.'))

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)


# -- Project information -----------------------------------------------------

project = "TorchData"
copyright = "2021 - Present, Torch Contributors"
author = "Torch Contributors"

# The short X.Y version
version = "main (" + torchdata.__version__ + " )"

# The full version, including alpha/beta/rc tags
release = "main"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx_design",
    "sphinx_sitemap",
    "pytorch_sphinx_theme2",
    "sphinxext.opengraph",
]

# Do not execute standard reST doctest blocks so that documentation can
# be successively migrated to sphinx's doctest directive.
doctest_test_doctest_blocks = ""

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]

# OpenGraph settings
ogp_site_url = "https://pytorch.org/data/"
ogp_image = "https://pytorch.org/assets/images/social-share.jpg"

# Theme options
theme_variables = pytorch_sphinx_theme2.get_theme_variables()

html_theme_options = {
    "navigation_with_keys": False,
    "analytics_id": "GTM-T8XT4PS",
    "logo": {
        "text": "TorchData",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pytorch/data",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/torchdata/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
    "navbar_start": ["pytorch_version"],
    "display_version": True,
}

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "pytorch",
    "github_repo": "data",
    "github_version": "main",
    "doc_path": "docs/source",
    "library_links": theme_variables.get("library_links", []),
    "community_links": theme_variables.get("community_links", []),
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Sitemap settings
html_baseurl = f"https://pytorch.org/data/{version}/"
sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
]
sitemap_url_scheme = "{link}"

signature_replacements = {}


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Replacing long type annotations in signature with more succinct ones."""
    if isinstance(signature, str):
        for old, new in signature_replacements.items():
            if old in signature:
                signature = signature.replace(old, new)
        return signature, return_annotation


def setup(app):

    app.connect("autodoc-process-signature", process_signature)


intersphinx_mapping = {
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
}
