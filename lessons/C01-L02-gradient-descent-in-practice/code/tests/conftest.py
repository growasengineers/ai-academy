"""Let the tests import the lesson modules (data, model, ...) that live one folder up.

The lesson's code folder is not a Python package, so we put it on sys.path once here.
"""
import pathlib
import sys

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
