"""
sklearn_nodes — Machine Learning plugin for Synapse

Provides preprocessing, classification, regression, evaluation, clustering,
visualization, grid search, SHAP analysis, and model I/O nodes using
scikit-learn (plus optional xgboost / shap).

Vendor setup
------------
The packaged ``.synpkg`` ships scikit-learn, xgboost, and shap inside
``vendor/`` so the plugin works on any host machine without those packages
pre-installed.  numpy, scipy, pandas, and matplotlib are expected to already
be present in the Synapse runtime.
"""
from __future__ import annotations

import pathlib
import sys

# ── Vendor injection ──────────────────────────────────────────────────────────
_vendor = pathlib.Path(__file__).parent / 'vendor'
if _vendor.is_dir() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

# ── Node imports ──────────────────────────────────────────────────────────────
from .ml_data import *
from .preprocess_nodes import *
from .classifier_nodes import *
from .regressor_nodes import *
from .gridsearch_nodes import *
from .eval_nodes import *
from .shap_nodes import *
from .model_io_nodes import *
from .cluster_nodes import *
from .plot_nodes import *
from .embedding_nodes import *
