"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest

_SYNAPSE = Path('/Users/s/Desktop/demo/PySide_Node/synapse')
if str(_SYNAPSE) not in sys.path:
    sys.path.insert(0, str(_SYNAPSE))


@pytest.fixture(scope='session')
def qapp():
    """Singleton QApplication for Qt-dependent tests."""
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app
